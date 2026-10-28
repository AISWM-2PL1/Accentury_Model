#!/usr/bin/env python3
"""val 8세트 위 방언 자리 정답률 — 무튜닝 vs 체크포인트 (`docs/TRACK2_BENCHMARK_V2_SPEC.md` §1 1차 지표 ①)

  왜  파인튜닝 1차에서 고령 3칸 CER 이 19.74 → 14.59 로 내렸다. 그것이 **방언 어절을 배운 것**인지
      간투사·발화 경계 같은 관례를 배운 것인지는 CER 만으로 못 가른다 (inference 명세 §14 8번 미완 ③).
      방언 자리(방언층 ≠ 표준층 어절)에서 정답을 맞춘 비율이 무튜닝보다 오르면 방언 학습이다.
  뭘  `dialect_pos_track2.py` 의 표 B(방언 자리 오류 유형)·E(오라클)를 **val 8세트 · 모델마다** 낸다.
      정렬·분류 규칙은 그 파일의 `align`·`wcer`·`oracle_hyp`·`KINDS`·`NEAR` 를 그대로 가져온다 — 복제하지 않는다.
      지표는 셋이다:
        · 방언 자리 정답률   = 정답 / 방언 어절            (1차 지표 ①)
        · 그 외 자리 정답률  = 그외 정답 / 그외 어절        (관례·일반 개선과 분리해 보기 위한 대조)
        · 오라클 CER         = 방언 자리를 전부 정답으로 바꿨을 때의 CER (개선 상한 — 모델마다 다시 계산)
      표는 val 8세트 각각과 판정 묶음(고령 3칸 · 경상 10~40대 · 서울 3칸) 합산으로 낸다.
  입력  --model <라벨>=<조각 폴더>  (run_track2_stt.py 산출 · `A__v2val_*.csv.gz`)  반복
        --sets-dir  기본 `$ACCENTURY_DATA/work/finetune/track2_bench_v2_val/sets`
  출력  registry/benchmark/track2_bench_v2_val_dialect_pos.json · 표는 stdout
  한계 (dialect_pos_track2.py 와 같다)
    - 「방언 어절」은 방언층≠표준층 어절이다. 방언 발음을 표준 철자로 적은 어절은 잡히지 않는다.
    - 출력 정렬은 **공백 무시 글자 정렬**(`align_chars`, 2026-09-09 박재영 결정). 그 전 어절 단위 정렬에서는
      「둘 다 멂」 표본 30개 중 15개가 띄어쓰기 차이였다. 방언층↔표준층 정렬은 어절 단위 그대로다.
    - 2~3글자 어절은 한 글자만 틀려도 어절 CER 이 NEAR(0.34)를 넘어 「둘 다 멂」이 된다 (같은 표본에서 30개 중 6개). 미수정.
    - 발화 단위 CI 는 내지 않는다 — 어절 비율이라 group 클러스터 재표집 규칙(§4.3)을 그대로 못 쓴다. 점추정만.
    - 청취 검증 없음 · GPU 없음 · 추론 없음.

  python3 src/stt/dialect_pos_val_track2.py --model baseline=<무튜닝 조각> --model w_ep14=<조각> ...
"""
import argparse
import collections
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.asr_metrics import normalize, score_one                     # 채점 규칙 복제 금지
from stt.dialect_pos_track2 import KINDS, align, align_chars, oracle_hyp   # 정렬·분류 규칙 복제 금지
from stt.dialect_pos_track2 import NEAR, wcer
from stt.paired_delta_track2 import sets_root
from stt.score_val_track2 import BUNDLES, EXPECTED, SAFE, kv

ROOT = Path(__file__).resolve().parents[2]
BENCH = "track2_bench_v2_val"
OUT = ROOT / f"registry/benchmark/{BENCH}_dialect_pos.json"


def classify(w, sw, hw):
    """dialect_pos_track2.main 의 방언 자리 분류 분기 그대로."""
    if hw == w:
        return "정답"
    if hw is None:
        return "누락"
    if sw is not None and hw == sw:
        return "표준형으로_씀"
    cd, cs = wcer(w, hw), (wcer(sw, hw) if sw is not None else 1.0)
    if cd <= NEAR and cd <= cs:
        return "방언형에_가까움"
    if cs <= NEAR:
        return "표준형에_가까움"
    return "둘_다_멂"


def tally(shard: Path, cond: str):
    """조각 하나 → Counter(어절 종류) + [dist, ref_chars, oracle_dist]."""
    b = collections.Counter()
    e = [0, 0, 0]
    for r in csv.DictReader(gzip.open(shard, "rt", encoding="utf-8")):
        sc = score_one(r["ref_dialect"], r["hypothesis"], status=r["status"], kspon=False)
        if "cer_dist" not in sc:
            continue
        d, st, h = normalize(r["ref_dialect"]).split(), normalize(r["ref_standard"]).split(), normalize(r["hypothesis"]).split()
        if not d:
            continue
        b["발화"] += 1
        e[0] += sc["cer_dist"]; e[1] += sc["ref_chars"]
        e[2] += score_one(" ".join(d), oracle_hyp(d, st, h), status="PASS", kspon=False)["cer_dist"]
        if d == st:
            continue
        dm, hm = align(d, st), align_chars(d, h)       # 출력 정렬은 공백 무시 (2026-09-09) · 두 층 정렬은 어절 단위 그대로
        for i, w in enumerate(d):
            sw, hw = dm.get(i), hm.get(i)
            if sw is not None and sw == w:
                b["그외_어절"] += 1
                b["그외_정답"] += (hw == w)
                continue
            b["방언_어절"] += 1
            b[classify(w, sw, hw)] += 1
    return b, e


def pct(a, n):
    return round(100 * a / n, 2) if n else None


def summarize(b, e):
    return {"발화": b["발화"], "방언_어절": b["방언_어절"], "그외_어절": b["그외_어절"],
            "방언자리_정답률": pct(b["정답"], b["방언_어절"]), "그외자리_정답률": pct(b["그외_정답"], b["그외_어절"]),
            "유형": {k: pct(b[k], b["방언_어절"]) for k in KINDS},
            "cer": pct(e[0], e[1]), "오라클_cer": pct(e[2], e[1])}


def main():
    ap = argparse.ArgumentParser(description="val 8세트 방언 자리 정답률 — 모델별")
    ap.add_argument("--model", dest="models", action="append", default=[], required=True, help="<라벨>=<조각 폴더> 반복")
    ap.add_argument("--sets-dir")
    ap.add_argument("--cond", default="A")
    ap.add_argument("--out", help=f"기본 {OUT.relative_to(ROOT)}")
    a = ap.parse_args()
    sets_dir = Path(a.sets_dir) if a.sets_dir else sets_root(BENCH)
    models = dict(kv(x, "model") for x in a.models)
    for l in models:
        if not SAFE.match(l):
            sys.exit(f"라벨은 [A-Za-z0-9._-] 만: {l}")
    missing = [s for s in EXPECTED if not (sets_dir / s / "manifest.csv").exists()]
    if missing:
        sys.exit(f"val 세트가 없다: {missing} ({sets_dir})")

    res = {}
    for l, d in models.items():
        d = Path(d) if Path(d).is_absolute() else ROOT / d
        per = {}
        for s in EXPECTED:
            shard = d / f"{a.cond}__{s}.csv.gz"
            if not shard.exists():
                sys.exit(f"조각 없음: {shard}")
            per[s] = tally(shard, a.cond)
        bundles = {}
        for n, spec in BUNDLES.items():
            b, e = collections.Counter(), [0, 0, 0]
            for s in spec["세트"]:
                b.update(per[s][0]); e = [x + y for x, y in zip(e, per[s][1])]
            bundles[n] = summarize(b, e)
        res[l] = {"in_dir": str(d), "세트": {s: summarize(*per[s]) for s in EXPECTED}, "묶음": bundles}

    base = next(iter(models))
    labels = list(models)
    # 표 1 — 판정 묶음 × 모델: 방언 자리 정답률 (그 외 자리 정답률)
    hdr = "| 묶음 | " + " | ".join(labels) + " |\n|---|" + "---:|" * len(labels)
    rows = []
    for n in BUNDLES:
        rows.append(f"| {n} 방언자리 | " + " | ".join(str(res[l]["묶음"][n]["방언자리_정답률"]) for l in labels) + " |")
        rows.append(f"| {n} 그외자리 | " + " | ".join(str(res[l]["묶음"][n]["그외자리_정답률"]) for l in labels) + " |")
    t1 = hdr + "\n" + "\n".join(rows)
    # 표 2 — 고령 3칸 오류 유형 × 모델
    hdr2 = "| 고령 3칸 유형 % | " + " | ".join(labels) + " |\n|---|" + "---:|" * len(labels)
    t2 = hdr2 + "\n" + "\n".join(f"| {k} | " + " | ".join(str(res[l]["묶음"]["고령_3칸"]["유형"][k]) for l in labels) + " |" for k in KINDS)
    # 표 3 — 고령 3칸 CER · 오라클
    hdr3 = "| 고령 3칸 | " + " | ".join(labels) + " |\n|---|" + "---:|" * len(labels)
    t3 = hdr3 + "\n" + "\n".join(f"| {k} | " + " | ".join(str(res[l]["묶음"]["고령_3칸"][k]) for l in labels) + " |" for k in ("cer", "오라클_cer"))
    # 표 4 — 세트별 방언 자리 정답률
    hdr4 = "| 세트 (방언 어절 n) | " + " | ".join(labels) + " |\n|---|" + "---:|" * len(labels)
    t4 = hdr4 + "\n" + "\n".join(f"| {s} ({res[base]['세트'][s]['방언_어절']:,}) | " + " | ".join(str(res[l]["세트"][s]["방언자리_정답률"]) for l in labels) + " |" for s in EXPECTED)

    doc = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "스크립트": "src/stt/dialect_pos_val_track2.py", "명세": "docs/TRACK2_BENCHMARK_V2_SPEC.md §1 1차 지표 ①",
           "벤치마크": BENCH, "세트_루트": str(sets_dir).replace(str(Path.home()), "~"),
           "정의": {"방언_어절": "방언층(ref_dialect)과 표준층(ref_standard)이 다른 어절 — 대리 지표",
                  "방언자리_정답률": "방언 자리에서 출력 어절 == 방언층 어절 인 비율 (%)",
                  "그외자리_정답률": "방언층 == 표준층인 어절에서 출력이 맞은 비율 (%) — 대조군",
                  "오라클_cer": "방언 자리 어절을 전부 정답으로 바꿨을 때의 micro CER — 모델마다 다시 계산",
                  "정렬": "방언층↔표준층은 어절 단위 (dialect_pos_track2.align) · 방언층↔출력은 공백 무시 글자 정렬 (align_chars, 2026-09-09)",
                  "가까움": f"어절 CER ≤ {NEAR}",
                  "규칙_출처": "src/stt/dialect_pos_track2.py 표 B·E 와 같은 분류·정렬 — 함수 import"},
           "용도": "val 세트 — 성능 보고가 아니라 「CER 하락이 방언 자리에서 났는가」의 진단 전용 (§4.3)",
           "한계": ["방언 발음을 표준 철자로 적은 어절은 방언 자리로 안 잡힌다 (기존 139 두 층 같은 행 33.8%)",
                  "출력 정렬은 공백 무시 — 띄어쓰기 차이는 더 이상 누락·둘 다 멂이 아니다 (2026-09-09 이전 값과 비교 불가)",
                  "2~3글자 어절은 한 글자 오류로도 둘 다 멂이 된다 (NEAR 0.34) — 미수정", "CI 없음 — 점추정만", "서울 3칸은 방언 어절이 거의 없다 (NIKL 표준 전사)", "청취 검증 없음"],
           "모델": res, "표": {"묶음_정답률": t1, "고령_유형": t2, "고령_cer_오라클": t3, "세트별_방언자리_정답률": t4},
           "판정": "판정은 사람이 한다. 표는 숫자만 낸다."}
    out = Path(a.out) if a.out else OUT
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("1. 판정 묶음 × 모델 — 방언 자리 정답률 % / 그 외 자리 정답률 %\n" + t1)
    print("\n2. 고령 3칸 — 방언 자리 오류 유형 %\n" + t2)
    print("\n3. 고령 3칸 — CER · 오라클 CER(방언 자리 전부 정답)\n" + t3)
    print("\n4. 세트별 방언 자리 정답률 %\n" + t4)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
