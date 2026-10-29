#!/usr/bin/env python3
"""트랙 2 벤치마크 v2 시험지 채점 — `docs/TRACK2_BENCHMARK_V2_SPEC.md` §3 · §5 3번 (코덱스 2026-09-03 2번)

  입력  registry/benchmark/track2_stt_bench_v2/A__v2_test_age_*.csv.gz
        (run_track2_stt.py --bench track2_bench_v2 --all --cond A --out-dir registry/benchmark/track2_stt_bench_v2)
        $ACCENTURY_DATA/work/benchmark/track2_bench_v2/sets/*/manifest.csv     (길이·성별·원천·수집자)
        registry/benchmark/track2_bench_v2_level_audit.json                    (세트별 저레벨 시간 비중)
  출력  registry/benchmark/track2_bench_v2_stt.json · 표는 stdout

  채점 규칙은 score_track2.py 그대로 (score_combo = asr_metrics.score_one/aggregate + 대화 단위 CI 400회).
  세트마다 전체 · 2초 이상 · 자유발화(`speech_style == free_dialogue` · 명세 §1 병기, 2026-09-11) · 남 · 여 를 낸다 — 길이·스타일 필터는 세트에 없고 여기서만 가른다 (명세 §2 「포함」).
  조각 다섯 개의 `adapter_sha256`(`_run_meta.json`)이 하나여야 하고 결과 `meta.adapter_sha256` 에 남긴다 (무튜닝은 None · 코덱스 6차 10번).
  성별 잣대는 다섯 세트의 남/여 조각을 합쳐 두 장으로 낸다 (명세 §3 「성별 잣대는 따로 만들지 않는다」).

  python3 src/stt/score_bench_v2.py            # 또는  score_track2.py --bench track2_bench_v2
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.score_track2 import read_shard, score_combo, fmt, check_meta      # 채점 규칙 복제 금지

ROOT = Path(__file__).resolve().parents[2]
V = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))
BENCH = "track2_bench_v2"
IN = ROOT / "registry/benchmark/track2_stt_bench_v2"
SETS = V / "work/benchmark" / BENCH / "sets"
LEVEL_AUDIT = ROOT / f"registry/benchmark/{BENCH}_level_audit.json"
SETS_JSON = ROOT / f"registry/benchmark/{BENCH}_test_sets.json"
OUT = ROOT / f"registry/benchmark/{BENCH}_stt.json"
CELLS = ["10to40s", "50s", "60s", "70s", "80s_plus"]
EXPECTED = [f"v2_test_age_{c}" for c in CELLS]
COLL = re.compile(r"collector(g[sw]\d+)_")
SHORT = 2.0
FREE = "free_dialogue"                       # manifest `speech_style` — 자유발화 전용 CER 의 기준 (시험지 고령 칸 자유발화 시간 비율 40.6/10.7/12.5%, 명세 §1)

BLIND_SPOT = ("시험지의 사각지대 (명세 §2.1 ④): 이름 포함 발화(신규 2.49%)·안 들리는 발화(2.05%)는 시험지 밖이다. "
              "잘 안 들리는 발화가 빠지므로 점수는 실제 서비스보다 약간 좋게 나온다. 2초 미만은 남겼다 — 「2초 이상」 열을 병기한다.")
REF_NOTE = ("참고선(서울 2.07 · kspon 11.38 · 기존 경남 16.25)은 8월 정규화 규칙으로 잰 값이라 여기 붙이지 않는다 — "
            "inference 명세 §14 14번(재채점) 뒤에 단다.")


def pack(rows, attr, label):
    """한 구간의 채점 묶음. 방언 정답은 CI 포함, 표준어 정답은 병기(CI 없음)."""
    d = score_combo(rows, "ref_dialect", want_ci=True)
    s = score_combo(rows, "ref_standard", want_ci=False)
    h = sum(attr[r["utterance_id"]]["dur"] for r in rows) / 3600
    colls = {attr[r["utterance_id"]]["coll"] for r in rows} - {""}
    tot_sec = sum(attr[r["utterance_id"]]["dur"] for r in rows)
    old_sec = sum(attr[r["utterance_id"]]["dur"] for r in rows if attr[r["utterance_id"]]["src"] == "기존_139")
    return {"구간": label, "cer": d["cer_micro"], "ci": d.get("ci"), "cer_no_outlier": d.get("cer_micro_no_outlier"),
            "n_outlier": d.get("n_outlier"), "failure_rate": d.get("failure_rate"), "scoreable": d.get("scoreable"),
            "wer": d.get("wer_micro"), "표준어_cer": s["cer_micro"],
            "방언_집계": d, "표준어_집계": s,                       # score_combo 반환 전체 (macro·SER·빈 정답 수 — 코덱스 2차 2번)
            "n": len(rows), "시간_h": round(h, 3),
            "대화": len({r["dialogue_id"] for r in rows}), "기존_수집자_수": len(colls) if old_sec else None,
            "기존_pct": round(100 * old_sec / tot_sec, 1) if tot_sec else None,      # 시간 기준 — 세트 정본과 같은 정의 (코덱스 2차 1번)
            "2초미만_pct": round(100 * sum(attr[r["utterance_id"]]["dur"] < SHORT for r in rows) / max(1, len(rows)), 1)}


def ci_str(p):
    return "—" if not p.get("ci") else f"{p['ci']['lo']:.2f}~{p['ci']['hi']:.2f}"


def main_v2(a):
    in_dir = Path(a.in_dir) if a.in_dir else IN
    if not in_dir.is_absolute():
        in_dir = ROOT / in_dir
    out = Path(a.out) if a.out else OUT
    if not out.is_absolute():
        out = ROOT / out
    gaps = [s for s in EXPECTED if not (in_dir / f"A__{s}.csv.gz").exists()]
    if gaps and not a.allow_partial:
        sys.exit(f"조각이 빠졌다 (5세트 × A): {gaps} — --allow-partial 은 스모크 전용")
    if len(gaps) == len(EXPECTED):
        sys.exit(f"조각이 하나도 없다: {in_dir}")
    if a.allow_partial and (not a.out or out.resolve() == OUT.resolve()):
        sys.exit("--allow-partial 은 정본 경로에 쓰지 않는다 — --out 을 registry 밖으로 준다")   # 코덱스 2차 4번
    present = [f"A__{s}" for s in EXPECTED if s not in gaps]
    sh_meta, devices, dtypes, meta_warn = check_meta(in_dir, present)               # 코덱스 2차 5번 — ②와 같은 메타 강제
    # 어댑터 동일성 (코덱스 6차 10번) — 다섯 조각이 같은 어댑터(또는 전부 무튜닝)로 만든 것이어야 한다. 결과 JSON 에 sha256 을 남긴다
    adapters = {sh_meta[k].get("adapter_sha256") for k in present}
    if len(adapters) > 1:
        sys.exit(f"조각의 adapter_sha256 이 하나가 아니다 {sorted(map(str, adapters))} — 무튜닝·어댑터 조각이 섞였다 (inference 명세 §9)")
    adapter_sha = adapters.pop() if adapters else None
    low = json.load(open(LEVEL_AUDIT, encoding="utf-8"))["세트별_저레벨_시간_비중_pct"] if LEVEL_AUDIT.exists() else {}
    sets_json = json.load(open(SETS_JSON, encoding="utf-8"))["세트"] if SETS_JSON.exists() else {}

    per_set, table, warn = {}, [], []
    pooled = {"m": ([], {}), "f": ([], {})}
    for s in EXPECTED:
        if s in gaps:
            continue
        m = SETS / s / "manifest.csv"
        if not m.exists():
            sys.exit(f"manifest 없음: {m}")
        attr = {}
        for r in csv.DictReader(open(m, encoding="utf-8")):
            mm = COLL.search(r["utterance_id"]) if r["source_dataset"] == "기존_139" else None
            attr[r["utterance_id"]] = {"dur": float(r["duration_label_sec"]), "gender": r["gender"], "src": r["source_dataset"],
                                       "style": r.get("speech_style", ""),                        # 자유발화 전용 CER (명세 §1·§5 7번 병기 — 코덱스 6차 11번)
                                       "coll": mm.group(1) if mm else "", "age": r["age_at_recording"], "dlg": r["dialogue_id"],
                                       "ref_d": (r.get("dialect_text_clean") or "").strip() or (r.get("dialect_text") or ""),
                                       "ref_s": (r.get("standard_text_clean") or "").strip() or (r.get("standard_text") or "")}
        rows = read_shard(in_dir / f"A__{s}.csv.gz")
        if len(rows) != len(attr) or {r["utterance_id"] for r in rows} != set(attr):
            sys.exit(f"{s}: 조각 {len(rows)}행 ≠ manifest {len(attr)}행 (또는 발화 불일치) — 다시 돌린다")
        bad = [r["utterance_id"] for r in rows                                       # 코덱스 2차 3번 — ID 만 같고 내용이 낡은 조각
               if r.get("set") != s or r.get("cond") != "A" or r["dialogue_id"] != attr[r["utterance_id"]]["dlg"]
               or r["ref_dialect"] != attr[r["utterance_id"]]["ref_d"] or r["ref_standard"] != attr[r["utterance_id"]]["ref_s"]]
        if bad:
            sys.exit(f"{s}: 조각 {len(bad)}행이 manifest 와 다르다 (set·cond·dialogue_id·정답) 예: {bad[:3]} — 다시 돌린다")
        by_g = {g: [r for r in rows if attr[r["utterance_id"]]["gender"] == g] for g in ("m", "f")}
        splits = [pack(rows, attr, "전체"),
                  pack([r for r in rows if attr[r["utterance_id"]]["dur"] >= SHORT], attr, "2초 이상"),
                  pack([r for r in rows if attr[r["utterance_id"]]["style"] == FREE], attr, "자유발화"),   # 목표 문장의 스타일 — 명세 §1 「칸 전체 + 자유발화 전용 병기」
                  pack(by_g["m"], attr, "남"), pack(by_g["f"], attr, "여")]
        for g in ("m", "f"):
            pooled[g][0].extend(by_g[g]); pooled[g][1].update({u: attr[u] for u in attr if attr[u]["gender"] == g})
        e, e2, efree, em, ef = splits
        for p in splits:
            if not p["scoreable"]:
                warn.append(f"🚨 {s} · {p['구간']}: 채점 가능 행 0")
            if p["failure_rate"]:
                warn.append(f"{s} · {p['구간']}: 실패율 {p['failure_rate']}%")
            if p["n_outlier"]:
                warn.append(f"{s} · {p['구간']}: CER>100% {p['n_outlier']}건")
        per_set[s] = {"연령칸": sets_json.get(s, {}).get("연령칸", s.removeprefix("v2_test_age_")), "구간": splits,
                      "저레벨_시간_pct": low.get(f"test/{s}"), "연령_구성": sorted({v["age"] for v in attr.values()})}
        table.append(f"| {s} | {fmt(e['cer'])} | {ci_str(e)} | {fmt(e2['cer'])} | {fmt(efree['cer'])} ({efree['n']:,}) | {fmt(em['cer'])} / {fmt(ef['cer'])} | "
                     f"{fmt(e['표준어_cer'])} | {e['n']:,} | {e['시간_h']:.2f} | "
                     f"{'메타 없음(신규만)' if e['기존_수집자_수'] is None else e['기존_수집자_수']} | {fmt(e['기존_pct'], 1)} | "
                     f"{e['2초미만_pct']} | {low.get('test/' + s, '—')} |")

    gender = {}
    for g, (rows, attr) in pooled.items():
        if len({r["utterance_id"] for r in rows}) != len(rows):                        # 코덱스 2차 6번
            sys.exit(f"성별 합산 {g}: 세트 간 같은 발화가 겹친다 — 합산하지 않는다")
        if rows:
            gender["남" if g == "m" else "여"] = [pack(rows, attr, "전체"),
                                                pack([r for r in rows if attr[r["utterance_id"]]["dur"] >= SHORT], attr, "2초 이상")]

    hdr = ("| 세트 | 방언 CER | CI | 2초 이상 | 자유발화 (발화) | 남 / 여 | 표준어 CER | 발화 | 시간h | 기존 수집자 | 기존%(시간) | 2초미만%(발화) | 저레벨% |\n"
           "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    age_tbl = hdr + "\n" + "\n".join(table)
    g_lines = ["| 성별 (5칸 합산) | 방언 CER | CI | 2초 이상 | 발화 | 시간h |", "|---|---:|---|---:|---:|---:|"]
    for g, (e, e2) in gender.items():
        g_lines.append(f"| {g} | {fmt(e['cer'])} | {ci_str(e)} | {fmt(e2['cer'])} | {e['n']:,} | {e['시간_h']:.2f} |")
    g_tbl = "\n".join(g_lines)

    doc = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "스크립트": "src/stt/score_bench_v2.py",
           "명세": "docs/TRACK2_BENCHMARK_V2_SPEC.md §3", "벤치마크": BENCH, "조건": "A",
           "규칙": "score_track2.score_combo · 정답 ref_dialect(CI 대화 단위 400회 seed 20260826) · ref_standard 병기 · 2초 이상은 라벨 길이 기준",
           "in_dir": str(in_dir.relative_to(ROOT)) if in_dir.is_relative_to(ROOT) else str(in_dir),
           "partial": bool(gaps), "빠진_세트": gaps,
           "meta": {"model": sorted({sh_meta[k]["model"] for k in present}), "device": devices, "dtype": dtypes,
                    "adapter_sha256": adapter_sha,                                   # None 이면 무튜닝
                    "n_shards": len(present), "by_cluster": True, "iters": 400, "seed": 20260826},
           "세트": per_set, "성별_합산": gender, "표": {"연령": age_tbl, "성별": g_tbl},
           "사각지대": BLIND_SPOT, "참고선": REF_NOTE, "경고": meta_warn + warn,
           "판정": "판정은 사람이 한다. 표는 숫자만 낸다."}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(age_tbl + "\n\n" + g_tbl + "\n\n" + BLIND_SPOT + "\n" + REF_NOTE)
    if warn:
        print("\n경고:\n  " + "\n  ".join(warn))
    print(f"→ {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir")
    ap.add_argument("--out")
    ap.add_argument("--allow-partial", action="store_true", help="조각이 5개가 아니어도 채점한다 (스모크 전용)")
    main_v2(ap.parse_args())


if __name__ == "__main__":
    main()
