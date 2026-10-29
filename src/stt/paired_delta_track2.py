#!/usr/bin/env python3
"""짝지은 CER 차이 Δ 와 group_id 클러스터 부트스트랩 — 마진 τ 산출기 (`docs/TRACK2_BENCHMARK_V2_SPEC.md` §4.3)

  왜  체크포인트 선택 규칙이 쓰는 마진 τ 는 **절대 CER 의 CI 가 아니라** 「같은 발화에 대한 두 모델의
      차이(paired Δ)」의 CI 폭이다 (§4.3 「마진 τ」 절 — 코덱스 11번). 두 모델이 공유하는 발화·화자
      변동을 절대 CI 는 두 번 세고, 그만큼 τ 가 부풀어 나쁜 체크포인트를 통과시킨다.
  뭘  두 추론 조각 묶음(무튜닝 baseline · 체크포인트)을 **같은 발화** 위에서 채점해
      Δ = CER_micro(B) − CER_micro(A) 의 점추정과 95% CI 를 낸다. Δ 가 **양수면 B 가 나쁘다.**
      절대 CER 의 group 클러스터 CI(`boot_ci_group`)도 같이 제공한다 — val 표가 쓴다.
  입력  --a-dir / --b-dir  : run_track2_stt.py 가 남긴 `<cond>__<set>.csv.gz` + `_run_meta.json`
        --sets-dir         : 세트 폴더 (`.../sets/<set>/manifest.csv`) — `group_id` 열이 재표집 단위
  출력  stdout 표 · --out 을 주면 JSON
  주의
    - **재표집 단위는 `group_id`(화자 묶음)다. 대화가 아니다.** 경상 60·70·80대+ val 은 대화가
      151·385·338 개인데 group 은 10·31·21 개다 — 대화로 재표집하면 같은 화자의 여러 대화를
      독립으로 세서 CI 가 실제보다 좁아진다 (§4.3 한계 · 명세가 group_id 를 화자 단위로 정한 이유).
    - 부트스트랩 2000회 · seed 0 · 2.5~97.5 백분위. `registry/benchmark/track2_seoul_native_speed_v1.json`
      의 CI 관례와 같다 (`src/stt/seoul_native_speed_track2.py:131` `boot_ci`).
    - 채점 규칙은 새로 만들지 않는다 — `asr_metrics.score_one(kspon=False)` · `include` 인 행만,
      `score_track2.score_combo` 와 같은 호출이다 (inference 명세 §6).
    - 두 모델 중 한쪽만 채점 가능한 발화(모델 실패·빈 정답)는 **교집합에서 빠진다.**
      `bootstrap_ci.paired` 와 같은 관례다. 빠진 건수는 `n_a_only`·`n_b_only` 로 보고한다.
    - GPU 를 쓰지 않는다 · 추론을 돌리지 않는다 · 오디오를 열지 않는다.

  ┌ 미결 1 — τ 를 언제 정하나 ──────────────────────────────────────────────────────────┐
  │ §4.3 은 「마진 τ 는 val 무튜닝 채점 직후에 정한다 … 학습 전에 정해지므로 순환하지 않는다」고    │
  │ 적었는데, paired Δ 는 정의상 **두 모델**이 있어야 잴 수 있다. 무튜닝만 있는 시점에는 Δ 자체가  │
  │ 없다. 명세의 두 문장이 그대로는 함께 설 수 없다.                                          │
  │   이 스크립트는 **τ 를 정하는 절차만** 제공하고 값을 고르지 않는다. 첫 체크포인트가 나오면       │
  │   `--a-dir <무튜닝> --b-dir <첫 체크포인트>` 로 바로 τ 가 나온다.                          │
  │   어느 체크포인트의 Δ CI 를 τ 로 쓸지는 **박재영 판단 대기**다. 후보:                        │
  │     (가) 첫(가장 이른) 체크포인트의 Δ CI — 선택 결과에 가장 덜 물든다. score_val_track2.py 기본값 │
  │     (나) 체크포인트마다 그때의 Δ CI — 규칙 2·3·4·5 의 문턱이 매번 바뀌어 재현이 어렵다          │
  │     (다) 사람이 고른 고정값 — `--tau` 로 넣는다                                          │
  │   baseline 을 두 조각으로 쪼개 재는 방식은 **쓰지 않는다** (박재영 지시 2026-09-03).           │
  └──────────────────────────────────────────────────────────────────────────────────┘
  ┌ 미결 2 — τ 는 CI 「폭」인가 「반폭」인가 ────────────────────────────────────────────────┐
  │ §4.3 은 「그 폭을 τ 로 쓴다」고 적었다. 같은 절의 정밀도 추정은 「CI 폭 0.98 … ±1.49」처럼        │
  │ 폭과 ± 를 구분해 쓰므로, 글자 그대로 읽으면 τ = 폭(hi − lo)이다. 다만 Δ 한쪽 방향만 보는          │
  │ 문턱(「τ 를 넘어 나빠지면」)에 폭 전체를 쓰는 것이 의도인지는 명세에 없다.                       │
  │   기본값은 **글자 그대로 폭**이고, `--tau-rule half_width` 로 반폭도 낼 수 있다. 둘 다 출력한다.  │
  └──────────────────────────────────────────────────────────────────────────────────┘

  python3 src/stt/paired_delta_track2.py --a-dir <무튜닝 조각> --b-dir <체크포인트 조각> \
          --sets-dir "$ACCENTURY_DATA/work/finetune/track2_bench_v2_val/sets" --out work/scratch/delta.json
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.asr_metrics import score_one                      # 채점 규칙 복제 금지
from stt.score_track2 import check_meta, read_shard        # 메타 검사·조각 읽기도 복제 금지

ROOT = Path(__file__).resolve().parents[2]
V = Path(os.environ.get("ACCENTURY_DATA") or (Path.home() / "AccenturyData"))

CLUSTER_COL = "group_id"        # 재표집 단위 (§4.3) — 대화가 아니다
ITERS, SEED = 2000, 0           # track2_seoul_native_speed_v1 과 같은 관례
MIN_GROUP = 5                   # 이보다 적으면 클러스터 재표집이 사실상 몇 가지 표본만 만든다 (경고선 — 명세 값 아님)
PLO, PHI = 0.025, 0.975
TAU_RULES = ("width", "half_width")


# ── 입력 ─────────────────────────────────────────────────
def sets_root(bench: str) -> Path:
    """세트 루트를 찾는다. `track2_data.set_bench` 는 `work/benchmark/<bench>/sets` 만 보는데
    val 세트는 `work/finetune/track2_bench_v2_val/sets` 에 있다 (`build_track2_v2_val.py:294`).
    track2_data.py 를 고치지 않기 위해 여기서 두 자리를 다 본다."""
    for sub in ("work/finetune", "work/benchmark"):
        p = V / sub / bench / "sets"
        if p.is_dir():
            return p
    sys.exit(f"세트 폴더가 없다: {V}/work/{{finetune,benchmark}}/{bench}/sets — --sets-dir 로 준다")


def list_sets(sets_dir: Path) -> list:
    return sorted(p.name for p in Path(sets_dir).iterdir() if p.is_dir())


def refs_of(r: dict) -> tuple:
    """정답 두 개. `_clean` 이 비면 원문으로 떨어진다 (`track2_data.refs` 와 같은 규칙).
    track2_data 를 import 하면 numpy·soundfile 이 딸려와 채점기가 GPU 상자 밖에서 안 돈다 —
    `score_bench_v2.py` 가 같은 이유로 인라인해 뒀다."""
    d = (r.get("dialect_text_clean") or "").strip() or (r.get("dialect_text") or "")
    s = (r.get("standard_text_clean") or "").strip() or (r.get("standard_text") or "")
    return d, s


def read_manifest(sets_dir: Path, set_name: str) -> dict:
    """세트 manifest → {utterance_id: 행}."""
    p = Path(sets_dir) / set_name / "manifest.csv"
    if not p.exists():
        sys.exit(f"manifest 없음: {p}")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    attr = {r["utterance_id"]: r for r in rows}
    if len(attr) != len(rows):
        sys.exit(f"{set_name}: manifest 안에 같은 utterance_id 가 있다 ({len(rows)}행 → {len(attr)}개)")
    if CLUSTER_COL not in rows[0]:
        sys.exit(f"{set_name}: manifest 에 {CLUSTER_COL} 열이 없다 — 재표집 단위를 못 만든다")
    return attr


def read_checked(shard_dir: Path, set_name: str, cond: str, attr: dict) -> list:
    """조각을 읽고 manifest 와 대조한다. 검사 항목은 `score_bench_v2.py` 와 같다
    (코덱스 2차 3번 — ID 만 같고 내용이 낡은 조각을 걸러야 한다)."""
    p = Path(shard_dir) / f"{cond}__{set_name}.csv.gz"
    if not p.exists():
        sys.exit(f"조각 없음: {p}")
    rows = read_shard(p)
    if len(rows) != len(attr) or {r["utterance_id"] for r in rows} != set(attr):
        sys.exit(f"{set_name}: 조각 {len(rows)}행 ≠ manifest {len(attr)}행 (또는 발화 불일치) — 다시 돌린다")
    bad = []
    for r in rows:
        a = attr[r["utterance_id"]]
        rd, rs = refs_of(a)
        if (r.get("set") != set_name or r.get("cond") != cond
                or r.get("dialogue_id") != a.get("dialogue_id")
                or r.get("ref_dialect") != rd or r.get("ref_standard") != rs):
            bad.append(r["utterance_id"])
    if bad:
        sys.exit(f"{set_name}: 조각 {len(bad)}행이 manifest 와 다르다 (set·cond·dialogue_id·정답) "
                 f"예: {bad[:3]} — 다시 돌린다")
    return rows


def cluster_map(attrs: dict) -> dict:
    """{세트: {utterance_id: 행}} → {utterance_id: group_id}.
    group_id 는 분할표 전역 식별자라 세트를 합쳐도 같은 화자 묶음이 하나로 모인다."""
    return {u: r[CLUSTER_COL] for a in attrs.values() for u, r in a.items()}


# ── 채점 ─────────────────────────────────────────────────
def score_rows(rows: list, ref_col: str = "ref_dialect") -> dict:
    """조각 행 → {utterance_id: 발화별 지표}. `include=False`(모델 실패·빈 정답) 는 뺀다.

    호출은 `score_track2.score_combo` 와 같다 — `asr_metrics.score_one(..., kspon=False)`
    (inference 명세 §6). 여기서 규칙을 새로 만들지 않는다. score_combo 는 발화별 값을
    돌려주지 않아 짝짓기·클러스터 재표집에 쓸 수 없어서 이 얇은 층만 둔다.
    """
    out = {}
    for r in rows:
        s = score_one(r[ref_col], r["hypothesis"], status=r["status"], kspon=False)
        if not s.get("include"):
            continue
        out[r["utterance_id"]] = {"dist": s["cer_dist"], "chars": s["ref_chars"], "cer": s["cer"],
                                  "wdist": s["wer_dist"], "words": s["ref_words"]}
    return out


def micro_cer(sc: dict, uids) -> float:
    c = sum(sc[u]["chars"] for u in uids)
    return 100.0 * sum(sc[u]["dist"] for u in uids) / c if c else 0.0


def micro_wer(sc: dict, uids) -> float:
    w = sum(sc[u]["words"] for u in uids)
    return 100.0 * sum(sc[u]["wdist"] for u in uids) / w if w else 0.0


def macro_cer(sc: dict, uids) -> float:
    return 100.0 * sum(sc[u]["cer"] for u in uids) / len(uids) if uids else 0.0


def _round(v, nd=2):
    return None if v is None else round(v, nd)


def buckets(sc: dict, uids, cluster: dict) -> dict:
    """{group_id: [편집거리 합, 정답 글자 합]}. 클러스터가 없는 발화는 예외로 올린다 —
    조용히 빠지면 재표집 단위가 달라진다."""
    out = {}
    for u in uids:
        k = cluster.get(u)
        if k is None:
            raise KeyError(f"{u}: {CLUSTER_COL} 를 못 찾았다 — manifest 를 다 넘겼는지 본다")
        e = out.setdefault(k, [0, 0])
        e[0] += sc[u]["dist"]
        e[1] += sc[u]["chars"]
    return out


def boot_ci_group(sc: dict, uids, cluster: dict, iters: int = ITERS, seed: int = SEED) -> dict:
    """절대 CER micro 의 95% CI. 재표집 단위는 `group_id` (§4.3).

    🚨 채점 가능 발화가 0이면 None 이다. 0.00 을 돌려주면 「완벽하다」로 읽힌다
    (`score_track2.score_combo` 의 같은 주의).
    """
    uids = list(uids)
    if not uids:
        return None
    b = buckets(sc, uids, cluster)
    keys = list(b)
    rnd = random.Random(seed)
    vals = []
    for _ in range(iters):
        smp = [b[rnd.choice(keys)] for _ in keys]
        c = sum(x[1] for x in smp)
        vals.append(100.0 * sum(x[0] for x in smp) / c if c else 0.0)
    vals.sort()
    lo, hi = vals[int(PLO * iters)], vals[int(PHI * iters) - 1]
    return {"lo": round(lo, 2), "hi": round(hi, 2), "width": round(hi - lo, 2),
            "n": len(uids), "group": len(keys), "iters": iters, "seed": seed, "unit": CLUSTER_COL}


def paired_delta(a: dict, b: dict, cluster: dict, uids=None,
                 iters: int = ITERS, seed: int = SEED,
                 label_a: str = "A", label_b: str = "B") -> dict:
    """같은 발화 위에서 Δ = CER_micro(b) − CER_micro(a) 와 그 95% CI.

    **Δ 가 양수면 b(체크포인트)가 나쁘다.** §4.3 의 「무튜닝 대비 τ 를 넘어 나빠졌다」는
    Δ > τ 다.

    a·b 는 `score_rows` 의 결과, cluster 는 {utterance_id: group_id}.
    uids 를 주면 그 범위(세트·칸)로 좁힌다. 없으면 a∩b 전체.
    같은 발화 집합 위의 micro CER 차이는 점추정에서 「짝지은 Δ」와 정확히 같다 —
    다른 것은 CI 뿐이다 (재표집을 두 모델에 같이 걸기 때문).
    """
    pool = set(a) & set(b)
    if uids is not None:
        pool &= set(uids)
    common = sorted(pool)
    only_a = len((set(a) - set(b)) & (set(uids) if uids is not None else set(a)))
    only_b = len((set(b) - set(a)) & (set(uids) if uids is not None else set(b)))
    if not common:
        return {"a": label_a, "b": label_b, "n": 0, "group": 0, "n_a_only": only_a, "n_b_only": only_b,
                "cer_a": None, "cer_b": None, "delta": None, "lo": None, "hi": None,
                "width": None, "half_width": None, "crosses_zero": None, "p_b_worse_pct": None,
                "cer_macro_a": None, "cer_macro_b": None, "delta_macro": None,
                "wer_a": None, "wer_b": None, "delta_wer": None,
                "iters": iters, "seed": seed, "unit": CLUSTER_COL, "scoreable": False}

    ba, bb = buckets(a, common, cluster), buckets(b, common, cluster)
    keys = sorted(ba)
    point = micro_cer(b, common) - micro_cer(a, common)

    rnd = random.Random(seed)
    diffs = []
    for _ in range(iters):
        smp = [rnd.choice(keys) for _ in keys]
        da = sum(ba[k][0] for k in smp); ca = sum(ba[k][1] for k in smp)
        db = sum(bb[k][0] for k in smp); cb = sum(bb[k][1] for k in smp)
        diffs.append((100.0 * db / cb if cb else 0.0) - (100.0 * da / ca if ca else 0.0))
    diffs.sort()
    lo, hi = diffs[int(PLO * iters)], diffs[int(PHI * iters) - 1]
    return {"a": label_a, "b": label_b, "n": len(common), "group": len(keys),
            "n_a_only": only_a, "n_b_only": only_b,
            "cer_a": _round(micro_cer(a, common)), "cer_b": _round(micro_cer(b, common)),
            "delta": round(point, 2), "lo": round(lo, 2), "hi": round(hi, 2),
            "width": round(hi - lo, 2), "half_width": round((hi - lo) / 2, 2),
            "crosses_zero": bool(lo <= 0 <= hi),
            "p_b_worse_pct": round(100.0 * sum(d > 0 for d in diffs) / iters, 1),
            "cer_macro_a": _round(macro_cer(a, common)), "cer_macro_b": _round(macro_cer(b, common)),
            "delta_macro": round(macro_cer(b, common) - macro_cer(a, common), 2),   # 병기 — 판정은 micro 로 한다
            "wer_a": _round(micro_wer(a, common)), "wer_b": _round(micro_wer(b, common)),
            "delta_wer": round(micro_wer(b, common) - micro_wer(a, common), 2),
            "iters": iters, "seed": seed, "unit": CLUSTER_COL, "scoreable": True}


def tau_of(d: dict, rule: str = "width"):
    """Δ 의 CI 에서 마진 τ 를 뽑는다. 미결 2 참조 — 기본은 명세 글자 그대로 「폭」."""
    if rule not in TAU_RULES:
        sys.exit(f"--tau-rule 은 {TAU_RULES} 중 하나다: {rule}")
    return None if not d or d.get(rule) is None else d[rule]


# ── CLI ──────────────────────────────────────────────────
def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def load_model(shard_dir: Path, sets: list, attrs: dict, cond: str, ref_col: str) -> dict:
    """조각 검사 + 채점. {utterance_id: 지표} 와 실행 메타를 돌려준다."""
    shards = [f"{cond}__{s}" for s in sets]
    sh, devices, dtypes, warn = check_meta(Path(shard_dir), shards)      # 코덱스 — 메타 없는 조각 금지
    sc = {}
    for s in sets:
        rows = read_checked(Path(shard_dir), s, cond, attrs[s])
        sc.update(score_rows(rows, ref_col))
    meta = {"model": sorted({sh[k]["model"] for k in shards}), "device": devices, "dtype": dtypes,
            "n_shards": len(shards),
            "in_dir": str(Path(shard_dir).relative_to(ROOT)) if Path(shard_dir).is_relative_to(ROOT) else str(shard_dir)}
    return {"scored": sc, "meta": meta, "warn": warn}


def main():
    ap = argparse.ArgumentParser(description="짝지은 Δ CER 과 group 클러스터 부트스트랩 CI (명세 §4.3)")
    ap.add_argument("--a-dir", required=True, help="기준 모델(무튜닝) 조각 폴더")
    ap.add_argument("--b-dir", required=True, help="비교 모델(체크포인트) 조각 폴더")
    ap.add_argument("--a-label", default="baseline")
    ap.add_argument("--b-label", default="ckpt")
    ap.add_argument("--bench", default="track2_bench_v2_val", help="--sets-dir 를 안 주면 이 이름으로 찾는다")
    ap.add_argument("--sets-dir")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="세트 이름 (반복·쉼표). 없으면 전부")
    ap.add_argument("--cond", default="A")
    ap.add_argument("--ref-col", default="ref_dialect", choices=["ref_dialect", "ref_standard"])
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tau-rule", default="width", choices=list(TAU_RULES))
    ap.add_argument("--out", help="JSON 저장 경로")
    a = ap.parse_args()

    sets_dir = Path(a.sets_dir) if a.sets_dir else sets_root(a.bench)
    sets = [x for v in a.sets for x in v.split(",") if x] or list_sets(sets_dir)
    attrs = {s: read_manifest(sets_dir, s) for s in sets}
    cluster = cluster_map(attrs)

    A = load_model(Path(a.a_dir), sets, attrs, a.cond, a.ref_col)
    B = load_model(Path(a.b_dir), sets, attrs, a.cond, a.ref_col)

    per_set, lines = {}, []
    for s in sets:
        d = paired_delta(A["scored"], B["scored"], cluster, uids=set(attrs[s]),
                         iters=a.iters, seed=a.seed, label_a=a.a_label, label_b=a.b_label)
        per_set[s] = d
        lines.append(f"| {s} | {fmt(d['cer_a'])} | {fmt(d['cer_b'])} | {fmt(d['delta'])} | "
                     f"{fmt(d['lo'])}~{fmt(d['hi'])} | {fmt(d['width'])} | {fmt(d['half_width'])} | "
                     f"{d['n']:,} | {d['group']} | {'예' if d['crosses_zero'] else '아니오'} |")
    allu = {u for s in sets for u in attrs[s]}
    pooled = paired_delta(A["scored"], B["scored"], cluster, uids=allu,
                          iters=a.iters, seed=a.seed, label_a=a.a_label, label_b=a.b_label)
    lines.append(f"| **합산({len(sets)}세트)** | {fmt(pooled['cer_a'])} | {fmt(pooled['cer_b'])} | "
                 f"{fmt(pooled['delta'])} | {fmt(pooled['lo'])}~{fmt(pooled['hi'])} | {fmt(pooled['width'])} | "
                 f"{fmt(pooled['half_width'])} | {pooled['n']:,} | {pooled['group']} | "
                 f"{'예' if pooled['crosses_zero'] else '아니오'} |")

    tbl = ("| 세트 | CER A | CER B | Δ(B−A) | 95% CI | 폭 | 반폭 | 발화 | group | 0 포함 |\n"
           "|---|---:|---:|---:|---|---:|---:|---:|---:|---|\n" + "\n".join(lines))
    print(f"A={a.a_label} ({a.a_dir})\nB={a.b_label} ({a.b_dir})\n")
    print(tbl)
    print(f"\nΔ 는 양수면 B 가 나쁘다 · 재표집 {CLUSTER_COL} · {a.iters}회 · seed {a.seed} · 정답 {a.ref_col}")
    print(f"τ 후보 (--tau-rule {a.tau_rule}) — 합산 {fmt(tau_of(pooled, a.tau_rule))} · "
          + " · ".join(f"{s} {fmt(tau_of(per_set[s], a.tau_rule))}" for s in sets))
    print("τ 를 무엇으로 확정할지는 사람이 정한다 (파일 머리말 미결 1·2).")
    warn = list(A["warn"] + B["warn"])
    for s in sets:            # group 이 적으면 재표집이 만들 수 있는 표본이 몇 가지뿐이다 (§4.3 한계)
        if per_set[s]["group"] < MIN_GROUP:
            warn.append(f"{s}: group {per_set[s]['group']}개 — 서로 다른 재표본이 몇 가지뿐이라 "
                        f"CI 폭·τ 를 믿을 수 없다 (§4.3 한계 「화자 묶음이 적다」)")
    for w in warn:
        print(f"경고: {w}")

    if a.out:
        p = Path(a.out)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
               "스크립트": "src/stt/paired_delta_track2.py",
               "명세": "docs/TRACK2_BENCHMARK_V2_SPEC.md §4.3",
               "규칙": {"Δ": "CER_micro(B) − CER_micro(A) · 양수면 B 가 나쁘다",
                      "재표집": CLUSTER_COL, "iters": a.iters, "seed": a.seed,
                      "백분위": "2.5~97.5", "정답": a.ref_col, "cond": a.cond,
                      "채점": "asr_metrics.score_one(kspon=False) · include 만 — score_track2.score_combo 와 같은 호출",
                      "τ_rule": a.tau_rule, "τ_미결": "머리말 미결 1·2 — 사람이 정한다"},
               "모델": {a.a_label: A["meta"], a.b_label: B["meta"]},
               "세트별": per_set, "합산": pooled, "표": tbl,
               "경고": warn,
               "판정": "판정은 사람이 한다. 표는 숫자만 낸다."}
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"→ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
