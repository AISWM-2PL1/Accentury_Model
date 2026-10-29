#!/usr/bin/env python3
"""문장 클러스터 부트스트랩 CI · 문장 쏠림 검사 — 재현 스크립트.

2026-08-14 신규. 위키 `stt-benchmark-results` 의 두 절이 인용하는 수치인데
**산출 코드가 어디에도 없어** 새로 쓴 것이다. 원래 값과 다르면 그 사실을 기록한다.

무엇을 재나
  ① 문장 쏠림 — 소수 문장이 전체 오류를 끌고 가는가
       순위 기준 둘을 함께 낸다. **오류 총량 순이 이 질문의 답**이고,
       오류율 순은 원문이 쓴 축이라 대조용으로 남긴다.
  ② CI 폭 비교 — 발화 단위 부트스트랩 vs 문장 클러스터 부트스트랩
  ③ 짝지은 모델 비교 — 같은 클러스터 위에서 두 모델의 CER 차이

왜 클러스터인가
  방언 코호트는 **같은 문장을 여러 화자가 돌려 읽은 것**이라 발화가 독립이
  아니다. 발화 단위로 재표본하면 오차범위가 실제보다 좁게 나온다.

🚨 서울에는 문장 클러스터가 없다
  `script_key` 가 빈 값이고 `script_text` 가 발화마다 다르다. 같은 문장을
  여러 화자가 읽은 구조가 아니므로 **클러스터 = 발화**가 되어 두 방식이 같아진다.
  방언과 같은 방식으로 묶으면 안 된다.

입력 (전부 repo 안)
  registry/benchmark/benchmark_v2_FROZEN.csv   uid ↔ script_key/script_text
  registry/asr/<model>__benchmark_all.csv      원시 전사 (CER 은 여기 없다)

채점은 `asr_metrics.score_one` 을 다시 돌린다 — CSV 에 발화별 지표가 없다.
**서울 코호트에는 `kspon=True`** (프로토콜 §10 — KsponSpeech 전용 전처리는 서울·Kspon 에만).

사용
  python3 src/stt/bootstrap_ci.py                      # 요약
  python3 src/stt/bootstrap_ci.py --json out.json      # 전체 결과 저장
  python3 src/stt/bootstrap_ci.py --iters 400 --seed 20260806
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # src/stt/ → src/ → repo 루트
REGISTRY = ROOT / "registry"

sys.path.insert(0, str(HERE))
from asr_metrics import score_one            # noqa: E402  (src/stt/ 안에 있다)

FROZEN = REGISTRY / "benchmark" / "benchmark_v2_FROZEN.csv"
ASR_DIR = REGISTRY / "asr"

# 위키가 인용하는 절차값. 바꾸면 결과가 달라지므로 기본값을 원문에 맞춘다.
DEFAULT_ITERS = 400
DEFAULT_SEED = 20260806

COHORTS = ("gyeongnam", "gyeongbuk", "seoul")


# ── 입력 ─────────────────────────────────────────────────
def load_frozen() -> dict[str, dict]:
    """uid → {cohort, script_key, script_text}."""
    with open(FROZEN, encoding="utf-8") as f:
        return {r["file_id"]: {"cohort": r["cohort"],
                               "script_key": r["script_key"],
                               "script_text": r["script_text"]}
                for r in csv.DictReader(f)}


def models() -> list[str]:
    return sorted(p.name.split("__")[0]
                  for p in ASR_DIR.glob("*__benchmark_all.csv"))


def score_model(name: str, frozen: dict) -> list[dict]:
    """발화별 채점 결과. include=False 인 것은 버린다(§4.4)."""
    rows = []
    with open(ASR_DIR / f"{name}__benchmark_all.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta = frozen.get(r["uid"])
            if meta is None:
                continue
            # 🚨 서울 코호트에는 KsponSpeech 전용 전처리를 건다 (프로토콜 §4.1b·§10).
            #    이걸 빼면 서울 CER 이 2.07 → 3.50 으로 부풀고 코호트 격차가 줄어 보인다.
            s = score_one(r["reference"], r["hypothesis"], r["status"],
                          kspon=(meta["cohort"] == "seoul"))
            if not s.get("include"):
                continue
            rows.append({"uid": r["uid"], "cohort": meta["cohort"],
                         "script_key": meta["script_key"],
                         "script_text": meta["script_text"],
                         "dist": s["cer_dist"], "chars": s["ref_chars"]})
    return rows


# ── 집계 ─────────────────────────────────────────────────
def micro(rows) -> float:
    c = sum(r["chars"] for r in rows)
    return 100.0 * sum(r["dist"] for r in rows) / c if c else 0.0


def cid(r) -> str:
    """재표본 단위. `script_key` 가 비면 그 발화 하나를 자기 클러스터로 둔다.

    🚨 이걸 안 하면 서울 4,084건이 빈 문자열 하나로 묶여 **거대한 단일
    클러스터**가 된다. 코호트를 섞어 재표본할 때 특히 조용히 틀린다.
    """
    return r["script_key"] or f"__utt__{r['uid']}"


def cluster_key(rows, cohort: str) -> str | None:
    """이 코호트에서 문장 클러스터를 정의할 수 있나.

    `script_key` 가 비어 있거나 값이 하나뿐이면 클러스터가 성립하지 않는다.
    """
    keys = {r["script_key"] for r in rows}
    if len(keys) <= 1 or keys == {""}:
        return None
    return "script_key"


# ── ① 문장 쏠림 ──────────────────────────────────────────
def concentration(rows) -> dict:
    """문장별 오류 점유율. 축은 `script_text`(위키 표와 같다)."""
    by = {}
    for r in rows:
        b = by.setdefault(r["script_text"], {"dist": 0, "chars": 0, "n": 0})
        b["dist"] += r["dist"]; b["chars"] += r["chars"]; b["n"] += 1

    total = sum(b["dist"] for b in by.values())
    cers = [100.0 * b["dist"] / b["chars"] for b in by.values() if b["chars"]]
    top5pct = max(1, round(len(by) * 0.05))

    base = micro(rows)
    worst = max(by.items(), key=lambda kv: kv[1]["dist"])[0]
    wo = [r for r in rows if r["script_text"] != worst]

    # 🚨 순위 기준을 둘 다 낸다.
    #   mass — 오류 총량 순. 「소수 문장이 전체 오류를 끌고 가는가」에 답한다
    #   rate — 오류율(CER) 순. 원문이 쓴 축인데, 오류율이 높아도 문장이 짧으면
    #          총량이 작아 **집중도를 과소평가한다** (경남 2.6% < 균등 기대 3.2%)
    out = {
        "n_sentences": len(by),
        "utts_per_sentence": round(len(rows) / len(by), 1),
        "cer_min": round(min(cers), 2), "cer_max": round(max(cers), 2),
        "cer_sd": round(st.pstdev(cers), 2) if len(cers) > 1 else 0.0,
        "uniform_top10_pct": round(100.0 * 10 / len(by), 1),   # 균등 분포 기대치
        "drop_worst_delta_pp": round(micro(wo) - base, 2),
    }
    for tag, key in (("mass", lambda b: -b["dist"]),
                     ("rate", lambda b: -(b["dist"] / b["chars"] if b["chars"] else 0))):
        rk = sorted(by.values(), key=key)
        out[f"top10_share_{tag}_pct"] = round(100.0 * sum(b["dist"] for b in rk[:10]) / total, 1)
        out[f"top5pct_share_{tag}_pct"] = round(100.0 * sum(b["dist"] for b in rk[:top5pct]) / total, 1)
    return out


# ── ②③ 부트스트랩 ───────────────────────────────────────
def _resample(units, rng):
    return [u for _ in range(len(units)) for u in (rng.choice(units),)]


def boot_ci(rows, by_cluster: bool, iters: int, seed: int) -> dict:
    """CER micro 의 95% CI. by_cluster=True 면 문장 단위로 재표본한다."""
    if by_cluster:
        groups = {}
        for r in rows:
            groups.setdefault(cid(r), []).append(r)
        units = list(groups.values())
    else:
        units = [[r] for r in rows]

    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        flat = [r for g in _resample(units, rng) for r in g]
        vals.append(micro(flat))
    vals.sort()
    lo, hi = vals[int(0.025 * iters)], vals[int(0.975 * iters) - 1]
    return {"lo": round(lo, 2), "hi": round(hi, 2), "width": round(hi - lo, 2)}


def paired(rows_a, rows_b, by_cluster: bool, iters: int, seed: int) -> dict:
    """두 모델을 **같은 재표본** 위에서 비교한다 (paired)."""
    a = {r["uid"]: r for r in rows_a}
    b = {r["uid"]: r for r in rows_b}
    common = sorted(set(a) & set(b))

    if by_cluster:
        groups = {}
        for u in common:
            groups.setdefault(cid(a[u]), []).append(u)
        units = list(groups.values())
    else:
        units = [[u] for u in common]

    point = micro([b[u] for u in common]) - micro([a[u] for u in common])
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        flat = [u for g in _resample(units, rng) for u in g]
        diffs.append(micro([b[u] for u in flat]) - micro([a[u] for u in flat]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]
    return {"n": len(common), "diff_pp": round(point, 2),
            "lo": round(lo, 2), "hi": round(hi, 2),
            "p_a_better_pct": round(100.0 * sum(d > 0 for d in diffs) / iters, 1),
            "crosses_zero": lo <= 0 <= hi}


# ── 실행 ─────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ref-model", default="whisper-large-v3",
                    help="쏠림·CI 폭을 잴 기준 모델")
    ap.add_argument("--json", type=Path, help="전체 결과를 여기에 저장")
    args = ap.parse_args()

    frozen = load_frozen()
    names = models()
    print(f"모델 {len(names)}개 · 동결본 {len(frozen)}건 "
          f"· iters={args.iters} · seed={args.seed}\n")

    scored = {}
    for n in names:
        scored[n] = score_model(n, frozen)
        print(f"  채점 {n:26} {len(scored[n]):6}건", flush=True)

    out = {"meta": {"iters": args.iters, "seed": args.seed,
                    "ref_model": args.ref_model, "models": names}}

    # ① 쏠림 + ② CI 폭
    ref = scored[args.ref_model]
    print(f"\n① 문장 쏠림 · ② CI 폭  (기준 모델 {args.ref_model})\n")
    hdr = f"{'코호트':<11}{'문장':>6}{'발화/문장':>9}{'CER범위':>18}{'sd':>7}" \
          f"{'상위10(총량)':>13}{'균등':>7}{'상위10(율)':>11}{'최악제거':>9}{'발화CI':>8}{'클러스터CI':>11}{'배':>6}"
    print(hdr); print("-" * len(hdr))
    out["cohorts"] = {}
    for co in COHORTS:
        rows = [r for r in ref if r["cohort"] == co]
        if not rows:
            continue
        c = concentration(rows)
        ck = cluster_key(rows, co)
        u = boot_ci(rows, False, args.iters, args.seed)
        g = boot_ci(rows, True, args.iters, args.seed) if ck else dict(u)
        ratio = round(g["width"] / u["width"], 1) if u["width"] else float("nan")
        c.update(cluster_axis=ck, ci_utterance=u, ci_cluster=g, ci_ratio=ratio,
                 cer_micro=round(micro(rows), 2))
        out["cohorts"][co] = c
        print(f"{co:<11}{c['n_sentences']:>6}{c['utts_per_sentence']:>9}"
              f"{c['cer_min']:>9.2f}~{c['cer_max']:<8.2f}{c['cer_sd']:>7}"
              f"{c['top10_share_mass_pct']:>12}%{c['uniform_top10_pct']:>6}%"
              f"{c['top10_share_rate_pct']:>10}%"
              f"{c['drop_worst_delta_pp']:>8}p{u['width']:>8}{g['width']:>11}{ratio:>6}")
        if ck is None:
            print(f"{'':11}└ 🚨 클러스터 축 없음 — script_key 가 비어 있다. "
                  f"클러스터 CI = 발화 CI")

    # ③ 짝지은 모델 비교 — 두 범위로 돌린다
    out["paired"] = {}
    for scope, keep in (("all", lambda r: True),
                        ("dialect", lambda r: r["cohort"] != "seoul")):
        sub = {n: [r for r in scored[n] if keep(r)] for n in names}
        order = sorted(names, key=lambda n: micro(sub[n]))
        label = "세 코호트 전체" if scope == "all" else "방언만 (경남+경북)"
        print(f"\n③ 짝지은 모델 비교 — {label} · 문장 클러스터 {args.iters}회\n")
        print("  CER micro 순위: " +
              " < ".join(f"{n}({micro(sub[n]):.2f})" for n in order))
        hdr2 = f"\n{'A → B (B 가 더 나쁘면 +)':<46}{'차이':>8}{'95% CI':>20}{'A 우세':>8}{'판정':>10}"
        print(hdr2); print("-" * 92)
        rows_out = []
        for a, b in zip(order, order[1:]):
            p = paired(sub[a], sub[b], True, args.iters, args.seed)
            verdict = "🚨 순위 없음" if p["crosses_zero"] else "✅ 유의"
            rows_out.append({"a": a, "b": b, **p})
            print(f"{a + ' → ' + b:<46}{p['diff_pp']:>+8.2f}"
                  f"{'[' + f'{p[chr(108) + chr(111)]:+.2f}' + ', ' + f'{p[chr(104) + chr(105)]:+.2f}' + ']':>20}"
                  f"{p['p_a_better_pct']:>7}%{verdict:>12}")
        out["paired"][scope] = rows_out

    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    main()
