#!/usr/bin/env python3
"""벤치마크 v2 시험지 결과의 방언 어절 분해 — 「CER 을 가르는 것은 지역이 아니라 방언 어절 밀도」의 근거 (2026-09-03)

  왜: v2 무튜닝 재측정(연령 10~40대 10.2 → 80대+ 21.2)과 NIKL 참고선(서울 ≈ 경상 ≈ 전라)이 나란히 나와
      「고령에서만 왜 높은가」를 물었다. 답: 고령일수록 방언 어절이 많고, 방언 어절은 나이와 무관하게 대부분 못 알아듣는다.
  무엇 (표 넷):
    A. 연령칸 × 원천 × 방언 어절 비율 버킷 → CER      (방언 어절 = 방언층과 표준층이 다른 어절, 대리 지표)
    B. 연령칸 × 방언 자리 오류 유형 (어절 단위)      정답 / 표준형으로 씀 / 방언형에 가까움 / 표준형에 가까움 / 둘 다 멂 / 누락
    C. 연령칸 × 초당 어절 버킷 → CER
    D. 신규 013-017 자유대화 연령대별 방언 어절 밀도 (분할표 전량 — 시험지 밖 근거)
    E. 오라클 — 방언 자리 어절을 전부 정답으로 바꿨을 때의 CER (개선 상한) · 「표준형으로 쓴」 자리만 고쳤을 때
  입력  registry/benchmark/track2_stt_bench_v2/A__v2_test_age_*.csv.gz · $ACCENTURY_DATA/work/benchmark/track2_bench_v2/sets/*/manifest.csv
        $ACCENTURY_DATA/work/benchmark/track2_split_manifest.csv.gz (D)
  출력  registry/benchmark/track2_bench_v2_dialect_pos.json · 표는 stdout

  한계 (해석 주의):
    - 「방언 어절」은 방언층≠표준층 어절이다. 기존 139 는 두 층이 같은 행이 33.8% 라 방언 발음을 표준 철자로 적은 어절은 0% 쪽에 섞인다.
    - 방언층↔표준층 정렬은 어절 단위 difflib. 방언층↔출력 정렬은 **공백 무시 글자 정렬**(`align_chars`, 2026-09-09 박재영 결정) —
      어절 정렬에서는 띄어쓰기 차이(`있다카던디`/`있다 카던디`)가 「누락·둘 다 멂」으로 세어졌다 (val 표본 30 중 15). 그 전 값과 표 B 는 비교 불가.
    - 2~3글자 어절은 한 글자 오류로도 어절 CER 이 NEAR(0.34)를 넘어 「둘 다 멂」이 된다. 미수정.
    - 청취 검증 없음.

  python3 src/stt/dialect_pos_track2.py
"""
import collections
import csv
import difflib
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.asr_metrics import normalize, score_one, edit_counts      # 채점 규칙 복제 금지

ROOT = Path(__file__).resolve().parents[2]
V = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))
BENCH = "track2_bench_v2"
IN = ROOT / "registry/benchmark/track2_stt_bench_v2"
SETS = V / "work/benchmark" / BENCH / "sets"
SPLIT = V / "work/benchmark/track2_split_manifest.csv.gz"
OUT = ROOT / f"registry/benchmark/{BENCH}_dialect_pos.json"
CELLS = ["10to40s", "50s", "60s", "70s", "80s_plus"]
RATIO_BUCKETS = ["0%", "1~25%", "26~50%", "51%+"]
RATE_BUCKETS = ["<1.5", "1.5~2.5", "2.5+"]
KINDS = ["정답", "표준형으로_씀", "방언형에_가까움", "표준형에_가까움", "둘_다_멂", "누락"]
NEAR = 0.34          # 어절 CER 이 1/3 이하면 「가까움」


def ratio_bucket(x):
    return "0%" if x == 0 else "1~25%" if x <= 0.25 else "26~50%" if x <= 0.5 else "51%+"


def rate_bucket(w):
    return "<1.5" if w < 1.5 else "1.5~2.5" if w < 2.5 else "2.5+"


def align(ref, hyp):
    """어절 정렬: ref 인덱스 → hyp 어절(또는 None). 같은 것을 먼저 잡고 치환 구간은 순서대로 짝짓는다."""
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = hyp[j1 + k]
        elif tag == "replace":
            for k in range(i2 - i1):
                m[i1 + k] = hyp[j1 + k] if k < (j2 - j1) else None
        elif tag == "delete":
            for k in range(i1, i2):
                m[k] = None
    return m


def align_chars(ref, hyp):
    """`align` 의 공백 무시판 (2026-09-09 · 박재영 결정). 같은 모양의 dict(ref 인덱스 → 출력 글자열 또는 None)를 돌려준다.

    어절 단위 `align` 은 띄어쓰기가 다르면(`있다카던디` / `있다 카던디`) 소리가 맞아도 「둘 다 멂·누락」으로 센다 —
    val 고령 3칸 「둘 다 멂」 표본 30개 중 15개가 이 오차였다. 여기서는 두 쪽의 공백을 지운 글자열을 정렬하고,
    ref 어절이 차지하는 글자 구간에 대응하는 hyp 글자 구간을 그 어절의 출력으로 삼는다.
    치환 구간은 글자 수 비례로 나눈다. 대응 글자가 하나도 없으면 None(누락).
    시험지 표 B(`main`)와 val(`dialect_pos_val_track2.py`) 둘 다 이것을 쓴다. `oracle_hyp` 는 CER 용이라 어절 정렬 그대로다."""
    rc, hc = "".join(ref), "".join(hyp)
    lo, hi = [None] * len(rc), [None] * len(rc)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=list(rc), b=list(hc), autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                lo[i1 + k], hi[i1 + k] = j1 + k, j1 + k + 1
        elif tag == "replace":
            n, m_ = i2 - i1, j2 - j1
            for k in range(n):
                lo[i1 + k], hi[i1 + k] = j1 + (k * m_) // n, j1 + ((k + 1) * m_) // n
    m, pos = {}, 0
    for i, w in enumerate(ref):
        span = [(lo[k], hi[k]) for k in range(pos, pos + len(w)) if lo[k] is not None and hi[k] > lo[k]]
        m[i] = hc[min(a for a, _ in span):max(b for _, b in span)] if span else None
        pos += len(w)
    return m


def wcer(a, b):
    d, _, _, _ = edit_counts(list(a), list(b))
    return d / max(1, len(a))


def oracle_hyp(d, st, h, only_standard=False):
    """방언 자리 어절을 정답으로 바꾼 출력. only_standard 면 「표준형으로 씀」 자리만 고친다. 다른 자리는 그대로."""
    dm = align(d, st)
    dialect = {i for i, w in enumerate(d) if dm.get(i) is None or dm[i] != w}
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=d, b=h, autojunk=False).get_opcodes():
        if tag in ("equal", "insert"):
            out += h[j1:j2]; continue
        hs = h[j1:j2]
        for k, i in enumerate(range(i1, i2)):
            hw = hs[k] if k < len(hs) else None
            if i in dialect and (not only_standard or (hw is not None and dm.get(i) == hw)):
                out.append(d[i])
            elif hw is not None:
                out.append(hw)
        if len(hs) > (i2 - i1):
            out += hs[(i2 - i1):]
    return " ".join(out)


def refs(r):
    d = (r.get("dialect_text_clean") or "").strip() or (r.get("dialect_text") or "")
    s = (r.get("standard_text_clean") or "").strip() or (r.get("standard_text") or "")
    return d, s


def main():
    A = collections.defaultdict(lambda: [0, 0])       # (cell, src, ratio_bucket) → [dist, ref_chars]
    B = collections.defaultdict(collections.Counter)  # cell → kind counts (+ "그외_어절", "그외_정답")
    C = collections.defaultdict(lambda: [0, 0])       # (cell, rate_bucket) → [dist, ref_chars]
    E = collections.defaultdict(lambda: [0, 0, 0, 0])  # cell → [base_dist, ref_chars, oracle_all_dist, oracle_std_dist]
    n_utt = collections.Counter()
    for cell in CELLS:
        s = f"v2_test_age_{cell}"
        attr = {r["utterance_id"]: r for r in csv.DictReader(open(SETS / s / "manifest.csv", encoding="utf-8"))}
        shard = IN / f"A__{s}.csv.gz"
        if not shard.exists():
            sys.exit(f"조각 없음: {shard}")
        for r in csv.DictReader(gzip.open(shard, "rt", encoding="utf-8")):
            sc = score_one(r["ref_dialect"], r["hypothesis"], status=r["status"], kspon=False)
            if "cer_dist" not in sc:
                continue
            n_utt[cell] += 1
            d, st, h = normalize(r["ref_dialect"]).split(), normalize(r["ref_standard"]).split(), normalize(r["hypothesis"]).split()
            if not d:
                continue
            m = attr[r["utterance_id"]]
            src = m["source_dataset"]
            dm = align(d, st)
            dialect_pos = [i for i, w in enumerate(d) if dm.get(i) is None or dm[i] != w]
            ratio = min(1.0, len(dialect_pos) / len(d))
            A[(cell, src, ratio_bucket(ratio))][0] += sc["cer_dist"]; A[(cell, src, ratio_bucket(ratio))][1] += sc["ref_chars"]
            dur = float(m["duration_label_sec"])
            rb = rate_bucket(len(d) / dur if dur > 0 else 0)
            C[(cell, rb)][0] += sc["cer_dist"]; C[(cell, rb)][1] += sc["ref_chars"]
            e = E[cell]; e[0] += sc["cer_dist"]; e[1] += sc["ref_chars"]
            e[2] += score_one(" ".join(d), oracle_hyp(d, st, h), status="PASS", kspon=False)["cer_dist"]
            e[3] += score_one(" ".join(d), oracle_hyp(d, st, h, only_standard=True), status="PASS", kspon=False)["cer_dist"]
            if d == st:
                continue                                   # 방언 자리 없는 발화는 B 밖
            hm = align_chars(d, h)                     # 출력 정렬은 공백 무시 (2026-09-09) · 두 층 정렬(dm)은 어절 단위 그대로
            b = B[cell]
            for i, w in enumerate(d):
                sw, hw = dm.get(i), hm.get(i)
                if sw is not None and sw == w:             # 그 외 자리
                    b["그외_어절"] += 1
                    if hw == w:
                        b["그외_정답"] += 1
                    continue
                b["방언_어절"] += 1
                if hw == w:
                    b["정답"] += 1
                elif hw is None:
                    b["누락"] += 1
                elif sw is not None and hw == sw:
                    b["표준형으로_씀"] += 1
                else:
                    cd, cs = wcer(w, hw), (wcer(sw, hw) if sw is not None else 1.0)
                    if cd <= NEAR and cd <= cs:
                        b["방언형에_가까움"] += 1
                    elif cs <= NEAR:
                        b["표준형에_가까움"] += 1
                    else:
                        b["둘_다_멂"] += 1

    # D. 신규 자유대화 연령대별 방언 어절 밀도 (분할표 전량)
    D = collections.defaultdict(lambda: [0, 0, 0.0, 0, 0])   # age → [rows, dialect_rows, sec, words, short2]
    with gzip.open(SPLIT, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_dataset"] != "신규_013-017" or r["speech_style"] != "free_dialogue":
                continue
            d, s = refs(r)
            a = D[r["age_at_recording"]]
            dur = float(r["duration_label_sec"])
            a[0] += 1; a[1] += (normalize(d) != normalize(s)); a[2] += dur; a[3] += len(normalize(d).split()); a[4] += dur < 2

    pct = lambda e, n: round(100 * e / n, 2) if n else None
    doc = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "스크립트": "src/stt/dialect_pos_track2.py",
           "입력": {"조각": str(IN.relative_to(ROOT)), "세트": str(SETS).replace(str(Path.home()), "~"), "분할표": str(SPLIT).replace(str(Path.home()), "~")},
           "정의": {"방언_어절": "방언층(_clean 우선)과 표준층이 다른 어절 — 대리 지표", "가까움": f"어절 CER ≤ {NEAR}",
                  "정렬": "방언층↔표준층은 어절 단위 difflib (align) · 방언층↔출력은 공백 무시 글자 정렬 (align_chars, 2026-09-09)",
                  "CER": "asr_metrics.score_one 의 cer_dist/ref_chars 합 (micro)"},
           "한계": ["기존 139 는 두 층이 같은 행 33.8% — 방언 발음을 표준 철자로 적은 어절은 0% 쪽",
                  "출력 정렬은 공백 무시 — 띄어쓰기 차이는 더 이상 누락·둘 다 멂이 아니다 (2026-09-09 이전 표 B 와 비교 불가)",
                  "2~3글자 어절은 한 글자 오류로도 둘 다 멂이 된다 (NEAR 0.34) — 미수정", "청취 검증 없음"],
           "발화_수": dict(n_utt),
           "A_방언어절비율별_CER": {f"{c}|{s}|{b}": {"cer": pct(*A[(c, s, b)]), "ref_chars": A[(c, s, b)][1]}
                              for c in CELLS for s in ("기존_139", "신규_013-017") for b in RATIO_BUCKETS if (c, s, b) in A},
           "B_방언자리_오류유형": {c: {**{k: pct(B[c][k], B[c]["방언_어절"]) for k in KINDS}, "방언_어절": B[c]["방언_어절"],
                              "그외_정답_pct": pct(B[c]["그외_정답"], B[c]["그외_어절"]), "그외_어절": B[c]["그외_어절"]} for c in CELLS},
           "C_초당어절별_CER": {f"{c}|{b}": {"cer": pct(*C[(c, b)]), "ref_chars": C[(c, b)][1]} for c in CELLS for b in RATE_BUCKETS if (c, b) in C},
           "D_신규자유대화_연령별_방언밀도": {k: {"발화": v[0], "방언어절_있는_발화_pct": pct(v[1], v[0]), "평균_초": round(v[2] / v[0], 2),
                                          "초당_어절": round(v[3] / v[2], 2), "2초미만_pct": pct(v[4], v[0])} for k, v in sorted(D.items())},
           "E_오라클_CER": {c: {"무튜닝": pct(E[c][0], E[c][1]), "방언자리_전부_정답": pct(E[c][2], E[c][1]),
                            "표준형으로_쓴_것만_정답": pct(E[c][3], E[c][1])} for c in CELLS},
           "판정": "판정은 사람이 한다. 표는 숫자만 낸다."}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("A. 연령칸 · 원천 · 방언 어절 비율 → CER")
    print("| 연령칸 | 원천 | " + " | ".join(RATIO_BUCKETS) + " |\n|---|---|" + "---:|" * len(RATIO_BUCKETS))
    for c in CELLS:
        for s in ("기존_139", "신규_013-017"):
            if any((c, s, b) in A for b in RATIO_BUCKETS):
                print(f"| {c} | {s[:2]} | " + " | ".join(f"{pct(*A[(c, s, b)])} ({A[(c, s, b)][1]:,})" if (c, s, b) in A else "—" for b in RATIO_BUCKETS) + " |")
    print("\nB. 연령칸 · 방언 자리 어절 결과 (%)  · 그 외 자리 정답 %")
    print("| 연령칸 | 방언 어절 n | " + " | ".join(KINDS) + " | 그 외 정답 |\n|---|---:|" + "---:|" * (len(KINDS) + 1))
    for c in CELLS:
        b = B[c]; print(f"| {c} | {b['방언_어절']:,} | " + " | ".join(str(pct(b[k], b["방언_어절"])) for k in KINDS) + f" | {pct(b['그외_정답'], b['그외_어절'])} |")
    print("\nC. 연령칸 · 초당 어절 → CER")
    print("| 연령칸 | " + " | ".join(RATE_BUCKETS) + " |\n|---|" + "---:|" * len(RATE_BUCKETS))
    for c in CELLS:
        print(f"| {c} | " + " | ".join(str(pct(*C[(c, b)])) for b in RATE_BUCKETS) + " |")
    print("\nD. 신규 자유대화 연령별 방언 어절 밀도")
    print("| 연령 | 발화 | 방언 어절 있는 발화 % | 평균 초 | 초당 어절 | 2초 미만 % |\n|---|---:|---:|---:|---:|---:|")
    for k, v in doc["D_신규자유대화_연령별_방언밀도"].items():
        print(f"| {k} | {v['발화']:,} | {v['방언어절_있는_발화_pct']} | {v['평균_초']} | {v['초당_어절']} | {v['2초미만_pct']} |")
    print("\nE. 오라클 CER (방언 자리 어절을 정답으로 바꿨을 때 — 개선 상한)")
    print("| 연령칸 | 무튜닝 | 방언 자리 전부 정답 | 표준형으로 쓴 것만 정답 |\n|---|---:|---:|---:|")
    for c in CELLS:
        v = doc["E_오라클_CER"][c]; print(f"| {c} | {v['무튜닝']} | {v['방언자리_전부_정답']} | {v['표준형으로_쓴_것만_정답']} |")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
