#!/usr/bin/env python3
"""트랙2 채점·표 생성 — `docs/TRACK2_STT_PIPELINE_SPEC.md` §6 B1-④

  저장된 결과를 채점해 표를 만든다. 오디오를 열지 않는다 · 모델을 부르지 않는다.
  GPU 가 필요 없다.

산출물 2·3 (inference 명세 §1)
  registry/benchmark/track2_tempbench_v1_stt.json

사용
  python src/stt/score_track2.py
  python src/stt/score_track2.py --in-dir work/scratch/track2_stt_smoke \
         --out work/scratch/track2_stt_smoke/stt.json

② 보강 세트 (`docs/TRACK2_STT_ADDON_SETS_SPEC.md` §6) — 조건 A 만 · 세트 4개 · ②의 조각에서
비교 칸 5개를 다시 계산해 원천 표·스타일 표를 낸다
  python src/stt/score_track2.py --bench track2_tempbench_v1_addon
  → registry/benchmark/track2_tempbench_v1_addon_stt.json

③ 벤치마크 v2 시험지 (`docs/TRACK2_BENCHMARK_V2_SPEC.md` §3) — 구현은 score_bench_v2.py
  python src/stt/score_track2.py --bench track2_bench_v2
  → registry/benchmark/track2_bench_v2_stt.json
"""
import argparse
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.asr_metrics import aggregate, score_one       # 복제 금지 — 그대로 쓴다
from stt.bootstrap_ci import boot_ci

ROOT = Path(__file__).resolve().parents[2]
IN_DEFAULT = ROOT / "registry/benchmark/track2_stt"
OUT_DEFAULT = ROOT / "registry/benchmark/track2_tempbench_v1_stt.json"
LEVEL_AUDIT = ROOT / "registry/benchmark/track2_tempbench_v1_level_audit.json"
COHORT = ROOT / "registry/asr/cohort_summary.json"
KSPON = ROOT / "registry/asr/whisper-large-v3__kspon_eval_clean.json"

# 재표집 단위는 대화다 (inference 명세 §8 결정 2026-08-26).
BY_CLUSTER, ITERS, SEED = True, 400, 20260826

AXES = [("연령", "age"), ("지역", "region"), ("스타일", "style"), ("원천", "source")]

# ② 보강 (addon 명세 §6) — ②의 A 조각에서 다시 뽑는 비교 칸. (원천, 연령대, 스타일)
ADDON_CELLS = {
    "②_신규_50대_free":   ("신규_013-017", "50대", "free_dialogue"),
    "②_기존_50대_free":   ("기존_139", "50대", "free_dialogue"),
    "②_기존_60대_free":   ("기존_139", "60대", "free_dialogue"),
    "②_기존_60대_read":   ("기존_139", "60대", "read"),
    "②_기존_60대_semi":   ("기존_139", "60대", "semi_dialogue"),
}
ADDON_TABLES = {   # 표 이름 → [(행 이름, addon 세트 또는 ② 칸)]
    "원천 (50대 · 자유대화 고정)": [("set17 기존 50대 free (신규 세트)", "set17_cell_legacy139_50s_free"),
                               ("② 기존 50대 free (얇은 칸)", "②_기존_50대_free"),
                               ("② 신규 50대 free (비교 짝)", "②_신규_50대_free")],
    "스타일 (기존 · 60대 고정)": [("set18 free (신규 세트)", "set18_cell_legacy139_60s_free"),
                              ("set19 read (신규 세트)", "set19_cell_legacy139_60s_read"),
                              ("set20 semi (신규 세트)", "set20_cell_legacy139_60s_semi"),
                              ("② 60대 free", "②_기존_60대_free"),
                              ("② 60대 read", "②_기존_60대_read"),
                              ("② 60대 semi", "②_기존_60대_semi")],
}


def expected_sets(bench: str = None) -> list:
    """기대 세트 16개. 세트 폴더가 정본이다 (§9 수용 기준 1)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stt import track2_data as D
    if bench:
        D.set_bench(bench)
    return D.list_sets()


def check_meta(in_dir: Path, shards: list) -> tuple:
    """채점 대상 조각 전부에 실행 메타(model·device·dtype)가 있어야 한다 (코덱스 3판 2026-08-27).
    (shards_meta, devices, dtypes, warnings) 를 돌려준다."""
    run_meta = {}
    rm = in_dir / "_run_meta.json"
    if rm.exists():
        run_meta = json.loads(rm.read_text(encoding="utf-8"))
    sh = run_meta.get("shards", {})
    if not isinstance(sh, dict):
        sh = {}
    no_meta = [k for k in shards
               if not all(sh.get(k, {}).get(f) for f in ("model", "device", "dtype"))]
    if no_meta:
        print(f"실행 메타가 없는 조각 {len(no_meta)}개:", file=sys.stderr)
        for k in no_meta[:10]:
            print(f"  메타없음 {k}.csv.gz", file=sys.stderr)
        sys.exit("_run_meta.json 이 채점 대상 조각을 모두 담아야 한다 "
                 "(어느 device·dtype 으로 만든 것인지 기록이 없다)")
    devices = sorted({sh[k]["device"] for k in shards})
    dtypes = sorted({sh[k]["dtype"] for k in shards})
    warn = []
    if len(devices) > 1 or len(dtypes) > 1:
        warn.append(f"🚨 혼합 실행: device={devices} dtype={dtypes} "
                    "— 한 방법으로 끝까지 간다 (inference 명세 §9)")
    return sh, devices, dtypes, warn


def addon_base_cells(base_sets_dir: Path, base_shards: Path) -> dict:
    """②의 A 조각 16개를 합쳐 발화 단위로 중복을 없애고, ② test manifest 로 (원천·연령대·스타일)을
    붙인 뒤 ADDON_CELLS 칸마다 채점한다. 노션 「무튜닝 STT 결과」의 「나이를 고정하고」 절이 손으로
    했던 계산을 재현 가능하게 코드로 옮긴 것이다 (addon 명세 §6)."""
    attr = {}
    for m in sorted(base_sets_dir.glob("set*/manifest.csv")):
        for r in csv.DictReader(open(m, encoding="utf-8")):
            attr[r["utterance_id"]] = (r["source_dataset"], r["age_at_recording"], r["speech_style"],
                                       float(r.get("duration_label_sec") or 0))
    rows = {}
    for p in sorted(base_shards.glob("A__set*.csv.gz")):
        for r in read_shard(p):
            rows.setdefault(r["utterance_id"], r)      # 같은 클립·같은 모델 — 첫 조각 것을 쓴다
    out = {}
    for name, cell in ADDON_CELLS.items():
        sel = [r for u, r in rows.items() if u in attr and attr[u][:3] == cell]
        m = score_combo(sel, "ref_dialect", want_ci=True) if sel else {"cer_micro": None, "ci": None,
                                                                          "scoreable": False}
        m["standard_cer_micro"] = score_combo(sel, "ref_standard", want_ci=False)["cer_micro"] if sel else None
        m["칸"] = {"원천": cell[0], "연령대": cell[1], "스타일": cell[2]}
        m["n_rows"] = len(sel)
        m["시간_h"] = round(sum(attr[r["utterance_id"]][3] for r in sel) / 3600, 3)
        m["출처"] = "②의 A 조각 16개 합집합(발화 중복 제거) — registry/benchmark/track2_stt/"
        out[name] = m
    return out


def build_addon_tables(per_set: dict, base_cells: dict, low: dict) -> dict:
    tables = {}
    for label, rows in ADDON_TABLES.items():
        lines = ["| 행 | 방언 A | CI A | 발화 | 시간h | 저레벨% | 표준어 A |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for title, key in rows:
            if key in per_set:
                m = per_set[key]["dialect_A"]; std = per_set[key]["standard_A"]["cer_micro"]
                n, h, lw = m.get("n_input", 0), per_set[key].get("시간_h", "—"), low.get("test/" + key, "—")
            elif key in base_cells:
                m = base_cells[key]; std = m["standard_cer_micro"]
                n, h, lw = m["n_rows"], m["시간_h"], "—"
            else:
                continue
            ci = m.get("ci")
            lines.append(f"| {title} | {fmt(m['cer_micro'])} | "
                         f"{'—' if not ci else '%.2f~%.2f' % (ci['lo'], ci['hi'])} | "
                         f"{n:,} | {h} | {lw} | {fmt(std)} |")
        tables[label] = "\n".join(lines) + "\n\n판정은 사람이 한다 (addon 명세 §7)."
    return tables


def main_addon(a):
    """② 보강 세트 채점 — 조건 A 만 · 세트 4개 · ② 비교 칸 5개 · 표 둘."""
    bench = a.bench
    suffix = bench.removeprefix("track2_tempbench_v1_")
    in_dir = Path(a.in_dir) if a.in_dir else IN_DEFAULT.with_name(f"track2_stt_{suffix}")
    out = Path(a.out) if a.out else ROOT / f"registry/benchmark/{bench}_stt.json"
    in_dir = in_dir if in_dir.is_absolute() else ROOT / in_dir
    out = out if out.is_absolute() else ROOT / out
    want = expected_sets(bench)
    gaps = [f"A__{s}" for s in want if not (in_dir / f"A__{s}.csv.gz").exists()]
    if gaps and not a.allow_partial:
        for g in gaps:
            print(f"  없음 {g}.csv.gz", file=sys.stderr)
        sys.exit(f"조각 {len(want)}개 중 {len(gaps)}개가 없다. 스모크면 --allow-partial")
    sets = [s for s in want if (in_dir / f"A__{s}.csv.gz").exists()]
    if not sets:
        sys.exit(f"조각이 없다: {in_dir}")

    from stt import track2_data as D
    per_set, warn_fail, warn_outlier = {}, [], []
    for s in sets:
        rows = read_shard(in_dir / f"A__{s}.csv.gz")
        m = {"dialect_A": score_combo(rows, "ref_dialect", want_ci=True),
             "standard_A": score_combo(rows, "ref_standard", want_ci=False),
             "시간_h": round(sum(float(r.get("dur") or 0) for r in rows) / 3600, 3)}
        per_set[s] = m
        for k in ("dialect_A", "standard_A"):
            v = m[k]
            if v.get("failure_rate"):
                warn_fail.append(f"{s} · {k}: 실패율 {v['failure_rate']}%")
            if v.get("n_outlier", 0) > 0:
                warn_outlier.append(f"{s} · {k}: CER>100% {v['n_outlier']}건")
            if not v.get("scoreable"):
                warn_fail.append(f"🚨 {s} · {k}: 채점 가능 행 0 — 다시 돌린다")

    shards = [f"A__{s}" for s in sets]
    sh, devices, dtypes, warn = check_meta(in_dir, shards)
    warn_fail += warn
    la = ROOT / f"registry/benchmark/{bench}_level_audit.json"
    low = json.load(open(la, encoding="utf-8"))["세트별_저레벨_시간_비중_pct"] if la.exists() else {}

    base_sets_dir = D.V / "work/benchmark" / D.BENCH_DEFAULT / "sets"
    base_cells = addon_base_cells(base_sets_dir, IN_DEFAULT) if IN_DEFAULT.exists() and base_sets_dir.exists() else {}
    if not base_cells:
        warn_fail.append("② 조각 또는 ② 세트 폴더가 없어 비교 칸을 못 냈다")

    doc = {
        "생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "명세": "docs/TRACK2_STT_ADDON_SETS_SPEC.md",
        "meta": {"by_cluster": BY_CLUSTER, "iters": ITERS, "seed": SEED, "cond": ["A"],
                 "model": sorted({sh[k]["model"] for k in shards}),
                 "device": devices, "dtype": dtypes,
                 "limit": sorted({sh[k].get("limit") or 0 for k in shards}),
                 "n_shards": len(shards),
                 "in_dir": str(in_dir.relative_to(ROOT)) if in_dir.is_relative_to(ROOT) else str(in_dir)},
        "세트별": per_set,
        "②_비교_칸": base_cells,
        "표": build_addon_tables(per_set, base_cells, low),
        "경고": {"실패율_0_아님": warn_fail, "CER_100_초과": warn_outlier, "조각_없음": gaps},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, t in doc["표"].items():
        print(f"\n### {label}\n{t}", flush=True)
    print(f"경고 {len(warn_fail) + len(warn_outlier) + len(gaps)}건 → {out}", flush=True)


def read_shard(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ci_rows(rows: list, scored: list) -> list:
    """`boot_ci` 어댑터. 🚨 네 키를 다 채운다 — 둘만 맞추면 KeyError 로 죽는다.

      chars      ← score_one() 의 ref_chars   (bootstrap_ci.py:99)
      dist       ← score_one() 의 cer_dist    (같음)
      uid        ← Row 의 utterance_id        (bootstrap_ci.py:104)
      script_key ← Row 의 dialogue_id         (같음 · 재표집 단위)

    include=True 인 행만 넘긴다. 실패·빈 정답 행은 `cer_dist` 자체가 없다
    (`asr_metrics.py:159~163`). `bootstrap_ci.py` 는 고치지 않는다.
    """
    out = []
    for raw, s in zip(rows, scored):
        if not s.get("include"):
            continue
        out.append({"chars": s["ref_chars"], "dist": s["cer_dist"],
                    "uid": raw["utterance_id"], "script_key": raw.get("dialogue_id", "")})
    return out


def score_combo(rows: list, ref_col: str, want_ci: bool) -> dict:
    """한 조합(조건 × 정답) 을 채점한다."""
    scored = [score_one(r[ref_col], r["hypothesis"], status=r["status"], kspon=False)
              for r in rows]                      # kspon=False 고정 (inference 명세 §6)
    agg = aggregate(scored)
    used = ci_rows(rows, scored)
    # 🚨 채점 가능 행이 0이면 null 이다. boot_ci([]) 는 lo=hi=0.00 을 돌려주는데,
    #    0.00 은 「완벽하다」가 아니라 「잰 것이 없다」다 (코덱스 지적 2026-08-26).
    if not used:
        # 키를 없애지 않는다 — 스키마가 조합마다 달라지면 표·후속 분석이 조용히 갈린다
        for k in ("cer_micro", "cer_macro", "wer_micro", "wer_macro", "ser",
                  "cer_micro_no_outlier", "ci"):
            agg[k] = None
        agg["scoreable"] = False
        return agg
    agg["scoreable"] = True
    if want_ci:
        agg["ci"] = boot_ci(used, by_cluster=BY_CLUSTER, iters=ITERS, seed=SEED)
    return agg


def axis_of(set_name: str) -> str:
    return set_name.split("_")[1]


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def ref_lines(ref: dict) -> str:
    """각 표 아래에 세 줄. 해석 없이 숫자만 (inference 명세 §10)."""
    return ("\n\n| 참고선 | CER | 출처 |\n|---|---:|---|\n"
            f"| 서울 표준어 | {ref['서울_표준어_CER']} | `registry/asr/cohort_summary.json` |\n"
            f"| KsponSpeech eval_clean | {ref['KsponSpeech_eval_clean_CER']} | "
            "`registry/asr/whisper-large-v3__kspon_eval_clean.json` |\n"
            f"| 기존 경남 | {ref['기존_경남_CER']} | `registry/asr/cohort_summary.json` |\n"
            "\n⚠️ 전부 whisper-large-v3 실측이지만 데이터가 달라 직접 비교가 아니다.")


def build_tables(per_set: dict, low: dict, ref: dict) -> dict:
    tables = {}
    for label, key in AXES:
        # 백분위 CI 는 점추정에 대칭이 아니다 — 폭을 반으로 접지 않고 두 끝을 그대로 적는다
        # (코덱스 2026-08-30 지적 4번: `±폭/2` 는 실제 끝점에서 최대 0.33%p 어긋났다).
        lines = ["| 세트 | 방언 A | 방언 B | CI A | CI B | 발화수 | 저레벨% | 표준어 B |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in sorted(x for x in per_set if axis_of(x) == key):
            m = per_set[s]
            da, db = m["dialect_A"], m["dialect_B"]
            ca, cb = da.get("ci"), db.get("ci")
            lines.append(
                f"| {s} | {fmt(da['cer_micro'])} | {fmt(db['cer_micro'])} | "
                f"{'—' if not ca else '%.2f~%.2f' % (ca['lo'], ca['hi'])} | "
                f"{'—' if not cb else '%.2f~%.2f' % (cb['lo'], cb['hi'])} | "
                f"{da.get('n_input', 0):,} | {low.get('test/' + s, '—')} | "
                f"{fmt(m['standard_B']['cer_micro'])} |")
        tables[label] = "\n".join(lines) + ref_lines(ref)
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir")
    ap.add_argument("--out")
    ap.add_argument("--allow-partial", action="store_true",
                    help="조각이 32개가 아니어도 채점한다 (스모크 전용)")
    ap.add_argument("--bench", default="track2_tempbench_v1",
                    help="track2_tempbench_v1_addon 이면 ② 보강 모드 (A 만 · 표 둘)")
    a = ap.parse_args()
    if a.bench == "track2_bench_v2":                       # ③ 시험지 (코덱스 2026-09-03 2번 — addon 분기로 새지 않게)
        from stt.score_bench_v2 import main_v2
        return main_v2(a)
    if a.bench != "track2_tempbench_v1":
        return main_addon(a)
    in_dir = Path(a.in_dir) if a.in_dir else IN_DEFAULT
    if not in_dir.is_absolute():
        in_dir = ROOT / in_dir
    out = Path(a.out) if a.out else OUT_DEFAULT
    if not out.is_absolute():
        out = ROOT / out

    found = sorted({p.name.split("__", 1)[1].removesuffix(".csv.gz")
                    for p in in_dir.glob("*__*.csv.gz")})
    if not found:
        sys.exit(f"조각이 없다: {in_dir}")

    # 🚨 있는 조각만 세면 빠진 세트가 조용히 사라지고, 한 조건만 없으면 KeyError 로
    #    죽는다. 기대 16세트 × A/B 를 먼저 검사한다 (코덱스 지적 2026-08-27).
    #    §9 수용 기준 1~3 (조각 32 · CER 64 · 축별 9/2/3/2행) 이 이것 없이는 안 선다.
    want = expected_sets()
    gaps = [f"{c}__{s}" for s in want for c in ("A", "B")
            if not (in_dir / f"{c}__{s}.csv.gz").exists()]
    if gaps and not a.allow_partial:
        print(f"조각 {len(want)*2}개 중 {len(gaps)}개가 없다:", file=sys.stderr)
        for g in gaps:
            print(f"  없음 {g}.csv.gz", file=sys.stderr)
        sys.exit("전량 채점 전에 빠진 조각을 채운다. 스모크면 --allow-partial")
    sets = found if a.allow_partial else want

    per_set, warn_fail, warn_outlier, missing = {}, [], [], []
    for s in sets:
        m = {}
        for cond in ("A", "B"):
            p = in_dir / f"{cond}__{s}.csv.gz"
            if not p.exists():
                missing.append(p.name)
                # 조합 넷을 빈 자리로 채운다 — 표 생성이 KeyError 로 죽지 않게 한다
                for ref in ("dialect", "standard"):
                    m[f"{ref}_{cond}"] = {"cer_micro": None, "ci": None,
                                          "scoreable": False, "n_scored": 0,
                                          "reason": "shard_missing"}
                continue
            rows = read_shard(p)
            for ref, col in (("dialect", "ref_dialect"), ("standard", "ref_standard")):
                # CI 는 방언 A·B 각각만 낸다 (inference 명세 §10). 표준어 값도 JSON 에는 다 담는다.
                m[f"{ref}_{cond}"] = score_combo(rows, col, want_ci=(ref == "dialect"))
        if not m:
            continue
        per_set[s] = m
        for k, v in m.items():
            if v.get("failure_rate"):
                warn_fail.append(f"{s} · {k}: 실패율 {v['failure_rate']}%")
            # `aggregate()` 가 `n_outlier` 를 직접 준다 (`asr_metrics.py:211`).
            # 빼서 계산하면 전량이 CER>100% 일 때 `n_no_outlier` 키가 없어 0이 된다
            # (코덱스 2판 지적 2026-08-27).
            n_out = v.get("n_outlier", 0)
            if n_out > 0:
                warn_outlier.append(f"{s} · {k}: CER>100% {n_out}건")
            if not v.get("scoreable"):
                warn_fail.append(f"🚨 {s} · {k}: 채점 가능 행 0 — 다시 돌린다")

    low = json.load(open(LEVEL_AUDIT, encoding="utf-8"))["세트별_저레벨_시간_비중_pct"]
    coh = json.load(open(COHORT, encoding="utf-8"))["whisper-large-v3"]
    ksp = json.load(open(KSPON, encoding="utf-8"))
    ref_block = {
        "서울_표준어_CER": coh["seoul"]["cer_micro"],
        "KsponSpeech_eval_clean_CER": ksp["cer_micro"],
        "기존_경남_CER": coh["gyeongnam"]["cer_micro"],
        "주의": "전부 whisper-large-v3 실측이지만 데이터가 달라 직접 비교가 아니다.",
    }
    run_meta, devices, dtypes = {}, [], []
    rm = in_dir / "_run_meta.json"
    if rm.exists():
        run_meta = json.loads(rm.read_text(encoding="utf-8"))
    sh = run_meta.get("shards", {})
    if not isinstance(sh, dict):            # 옛 형식(리스트)은 조각별 기록이 없다
        sh, run_meta["shards"] = {}, {}

    # 🚨 채점에 쓰는 조각은 전부 실행 메타가 있어야 한다. 하나라도 없으면 그 조각이
    #    어떤 device·dtype 으로 만들어졌는지 알 수 없고, 혼합 실행이 조용히 통과한다
    #    (코덱스 3판 지적 2026-08-27 — 메타 31/32 로도 exit 0 이었다).
    scored_shards = [f"{c}__{x}" for x in sets for c in ("A", "B")
                     if (in_dir / f"{c}__{x}.csv.gz").exists()]
    no_meta = [k for k in scored_shards
               if not all(sh.get(k, {}).get(f) for f in ("model", "device", "dtype"))]
    if no_meta:
        print(f"실행 메타가 없는 조각 {len(no_meta)}개:", file=sys.stderr)
        for k in no_meta[:10]:
            print(f"  메타없음 {k}.csv.gz", file=sys.stderr)
        sys.exit("_run_meta.json 이 채점 대상 조각을 모두 담아야 한다 "
                 "(어느 device·dtype 으로 만든 것인지 기록이 없다)")

    devices = sorted({sh[k]["device"] for k in scored_shards})
    dtypes = sorted({sh[k]["dtype"] for k in scored_shards})
    if len(devices) > 1 or len(dtypes) > 1:
        warn_fail.append(f"🚨 혼합 실행: device={devices} dtype={dtypes} "
                         "— 한 방법으로 끝까지 간다 (inference 명세 §9)")

    doc = {
        "생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "명세": "docs/TRACK2_STT_INFERENCE_SPEC.md",
        "meta": {"by_cluster": BY_CLUSTER, "iters": ITERS, "seed": SEED,
                 "model": sorted({sh[k]["model"] for k in scored_shards}),
                 "device": devices, "dtype": dtypes,
                 "limit": sorted({sh[k].get("limit") or 0 for k in scored_shards}),
                 "n_shards": len(scored_shards),
                 "in_dir": str(in_dir.relative_to(ROOT)) if in_dir.is_relative_to(ROOT) else str(in_dir)},
        "참고선": ref_block,
        "세트별": per_set,
        "축별_표": build_tables(per_set, low, ref_block),
        "경고": {"실패율_0_아님": warn_fail, "CER_100_초과": warn_outlier, "조각_없음": missing},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, _ in AXES:
        print(f"\n### {label}\n{doc['축별_표'][label]}", flush=True)
    print(f"\n참고선  서울 {doc['참고선']['서울_표준어_CER']} · "
          f"kspon {doc['참고선']['KsponSpeech_eval_clean_CER']} · "
          f"기존 경남 {doc['참고선']['기존_경남_CER']}", flush=True)
    print(f"경고 {len(warn_fail) + len(warn_outlier) + len(missing)}건 → {out}", flush=True)


if __name__ == "__main__":
    main()
