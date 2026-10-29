#!/usr/bin/env python3
"""방언5 채점 결과를 test WER 로 다시 집계한다 — 결과 문서 §5·§6·§8·§14.1 표의 WER 판 (2026-10-03, `docs/DIALECT5_RESULTS_2026-10-02.md` §15).

  왜   결과 문서와 노션은 CER 만 적었다. 채점기(`score_dialect5_val.py`)는 보고 JSON 에 `wer_micro` 를, 발화별 CSV 에 `wer` 를
       처음부터 함께 저장했으므로 GPU 재추론 없이 WER 표를 낼 수 있다. 연령별·부트스트랩 구간은 CSV 에 어절 수가 없어 다시 채점해야 한다.
  뭘   ① 전체 WER 은 보고 JSON 의 `채점.wer_micro` 를 그대로 읽는다(교차 20건 포함).
       ② 연령별·구간은 CSV 의 reference/hypothesis 를 채점기와 같은 `stt.asr_metrics.score_one` 으로 다시 채점해
          발화별 wer_dist·ref_words 를 얻고, micro WER = Σwer_dist / Σref_words 로 묶는다(`asr_metrics.aggregate` 의 정의).
          재채점한 전체값을 JSON 의 wer_micro·cer_micro·n_scored 와 대조해 일치를 확인한다(불일치가 있으면 표에 ❌).
       ③ 통합 − 방언별 구간은 §6 과 같은 설계 — 같은 발화끼리 짝지은 부트스트랩(기본 2,000회) 95% 백분위, seed 고정.
       ④ 라벨 결함 대화(`dialect5_exclusions.EXCLUDE_DIALOGUES`, 결과 문서 §17.1)는 표·구간에서 기본으로 뺀다(2026-10-03 추가).
          교차 행도 JSON 대신 CSV 재채점으로 내야 뺄 수 있어 20건을 모두 다시 채점한다. 대조(JSON 대 재채점)는 빼기 전 값으로 한다.
          `--no-exclude` 면 옛 표(전체)가 그대로 나온다.
  입력 `--rep-dir`: S3 `results_dialect5/` 사본(`<region>/<run>/report/test_cer_<step>.{json,csv}` + `seoul/seoul_pure/report/val_cer_none.*`).
       `--man-dir`: test manifest 5개(`<region>_test.csv`) + 서울 val manifest(`seoul_val.csv`) — `cell`(u60/o60) 열을 쓴다.
  출력 표준 출력에 markdown 표(대조 · §5 · §6 · §14.1 · §8 · 대체율). `--out-md` / `--out-json` 에 같은 내용을 쓴다.

  예)
    aws s3 sync s3://accentury-track2-325771561913/results_dialect5/ /tmp/rep --exclude "*" --include "*/report/test_cer_*" --include "seoul/seoul_pure/report/val_cer_none.*"
    for r in gyeongsang chungcheong jeolla gangwon jeju; do aws s3 cp s3://…/datasets/dialect5_bench_v2/$r/01bab6ab…/sets/${r}_test/manifest.csv /tmp/man/${r}_test.csv; done
    cp ~/AccenturyData/work/finetune/dialect5_bench_v2/seoul/sets/seoul_val/manifest.csv /tmp/man/seoul_val.csv
    uv run --no-project --with 'numpy~=2.3' python src/finetune/aggregate_dialect5_wer.py --rep-dir /tmp/rep --man-dir /tmp/man \
        --out-md registry/finetune/dialect5_wer_2026-10-03.md --out-json registry/finetune/dialect5_wer_2026-10-03.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.asr_metrics import score_one  # noqa: E402  채점 정본
from finetune.dialect5_exclusions import EXCLUDE_DIALOGUES  # noqa: E402

REGIONS = ["gyeongsang", "chungcheong", "jeolla", "gangwon", "jeju"]
KO = {"gyeongsang": "경상", "chungcheong": "충청", "jeolla": "전라", "gangwon": "강원", "jeju": "제주",
      "pooled": "통합", "none": "무튜닝", "self": "방언별"}
# 결과 문서 §5 최종 모델(run, step) · 무튜닝 run · 교차 run 이름의 모델 꼬리표
FINAL = {"gyeongsang": ("gyeongsang_GS-01", "step004686"), "chungcheong": ("chungcheong_GS-23", "step000921"),
         "jeolla": ("jeolla_GS-09", "step004761"), "gangwon": ("gangwon_GS-10", "step003092"), "jeju": ("jeju_GS-10", "step006192")}
BASE = {"gyeongsang": "gyeongsang_baseline", "chungcheong": "chungcheong_GS-01", "jeolla": "jeolla_GS-01",
        "gangwon": "gangwon_GS-01", "jeju": "jeju_GS-01"}
XTAG = {"gyeongsang": "GS01", "chungcheong": "GS23", "jeolla": "GS09", "gangwon": "GS10", "jeju": "GS10"}
POOLED_STEP = "step021244"   # pooled_GS-09


def report_path(rep, test_region, model, split="test"):
    """(json, csv). model ∈ {'none', 'self', 'pooled', <다른 지역>}"""
    if model == "none":
        run, step = BASE[test_region], "none"
    elif model == "self":
        run, step = FINAL[test_region]
    elif model == "pooled":
        run, step = f"{test_region}_pooledGS-09", POOLED_STEP
    else:
        run, step = f"{test_region}_x{model}{XTAG[model]}", FINAL[model][1]
    base = rep / test_region / run / "report" / f"{split}_cer_{step}"
    return base.with_suffix(".json"), base.with_suffix(".csv")


def json_summary(p):
    j = json.loads(p.read_text(encoding="utf-8"))["채점"]
    return j["wer_micro"], j["cer_micro"], j["n_scored"]


def per_utt(csv_path):
    """utterance_id → (wer_dist, ref_words, cer_dist, ref_chars). PASS 이고 정답이 빈 발화가 아닌 것만."""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = score_one(r["reference"], r["hypothesis"], status=r["status"], kspon=False)
            if s.get("include"):
                out[r["utterance_id"]] = (s["wer_dist"], s["ref_words"], s["cer_dist"], s["ref_chars"])
    return out


def micro(d, ids=None):
    ids = list(d) if ids is None else [i for i in ids if i in d]
    wd = sum(d[i][0] for i in ids); rw = sum(d[i][1] for i in ids)
    cd = sum(d[i][2] for i in ids); rc = sum(d[i][3] for i in ids)
    return round(100 * wd / max(1, rw), 2), round(100 * cd / max(1, rc), 2), len(ids)


def cells(man_path):
    with open(man_path, newline="", encoding="utf-8") as f:
        return {r["utterance_id"]: r["cell"] for r in csv.DictReader(f)}


def excluded(man_path, enabled=True):
    """manifest 에서 라벨 결함 대화에 속한 utterance_id 집합 (enabled=False 면 빈 집합)."""
    if not enabled:
        return set()
    with open(man_path, newline="", encoding="utf-8") as f:
        return {r["utterance_id"] for r in csv.DictReader(f) if r.get("dialogue_id") in EXCLUDE_DIALOGUES}


def drop(d, ex):
    return {k: v for k, v in d.items() if k not in ex} if ex else d


def paired_boot(a, b, n, seed):
    """통합(b) − 방언별(a) micro WER 차이(%p)의 점추정과 95% 백분위 구간. 같은 발화끼리 짝지음."""
    ids = sorted(set(a) & set(b))
    wa = np.array([a[i][0] for i in ids], float); ra = np.array([a[i][1] for i in ids], float)
    wb = np.array([b[i][0] for i in ids], float); rb = np.array([b[i][1] for i in ids], float)
    rng = np.random.default_rng(seed)
    N = len(ids)
    diffs = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, N, N)
        diffs[k] = 100 * (wb[idx].sum() / rb[idx].sum() - wa[idx].sum() / ra[idx].sum())
    point = 100 * (wb.sum() / rb.sum() - wa.sum() / ra.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(point), 2), round(float(lo), 2), round(float(hi), 2), N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-dir", required=True, help="S3 results_dialect5/ 사본")
    ap.add_argument("--man-dir", required=True, help="<region>_test.csv 5개 + seoul_val.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-exclude", action="store_true", help="라벨 결함 대화도 넣는다(옛 표)")
    a = ap.parse_args()
    rep, man = Path(a.rep_dir), Path(a.man_dir)

    checks, T1, T2, T21, T3, EXN = [], {}, {}, {}, {}, {}
    for tr in REGIONS:
        cell = cells(man / f"{tr}_test.csv")
        ex = excluded(man / f"{tr}_test.csv", not a.no_exclude)
        EXN[tr] = len(ex)
        u60 = [i for i, c in cell.items() if c == "u60"]; o60 = [i for i, c in cell.items() if c == "o60"]
        d = {}
        for model in ("none", "self", "pooled"):
            jp, cp = report_path(rep, tr, model)
            jw, jc, jn = json_summary(jp)
            d[model] = per_utt(cp)
            w_all, c_all, n_all = micro(d[model])
            checks.append((KO[tr], KO[model], jw, w_all, jc, c_all, jn, n_all))
            d[model] = drop(d[model], ex)
            w_all, c_all, n_all = micro(d[model])
            w_u, _, n_u = micro(d[model], u60); w_o, _, n_o = micro(d[model], o60)
            T1.setdefault(tr, {})[model] = (w_all, w_u, w_o, n_all, n_u, n_o)
        pt, lo, hi, N = paired_boot(d["self"], d["pooled"], a.boot, a.seed)
        T2[tr] = (T1[tr]["none"][0], T1[tr]["self"][0], T1[tr]["pooled"][0], pt, lo, hi, N)
        T21[tr] = (T1[tr]["none"][1], T1[tr]["self"][1], T1[tr]["pooled"][1], T1[tr]["none"][2], T1[tr]["self"][2], T1[tr]["pooled"][2])
        row = {"none": T1[tr]["none"][0], "pooled": T1[tr]["pooled"][0], tr: T1[tr]["self"][0]}
        for mr in REGIONS:
            if mr != tr:
                jp, cp = report_path(rep, tr, mr)
                jw, jc, jn = json_summary(jp)
                xd = per_utt(cp)
                w, c, n = micro(xd)
                checks.append((KO[tr], f"{KO[mr]} 모델", jw, w, jc, c, jn, n))
                row[mr] = micro(drop(xd, ex))[0]
        T3[tr] = row

    sj = rep / "seoul" / "seoul_pure" / "report" / "val_cer_none.json"
    sw, scer, sn = json_summary(sj)
    sd = per_utt(sj.with_suffix(".csv"))
    scell = cells(man / "seoul_val.csv")
    s_all = micro(sd); s_u = micro(sd, [i for i in sd if scell.get(i) == "u60"]); s_o = micro(sd, [i for i in sd if scell.get(i) == "o60"])
    checks.append(("서울(val)", "무튜닝", sw, s_all[0], scer, s_all[1], sn, s_all[2]))

    L = []
    P = L.append
    exd = [] if a.no_exclude else sorted(EXCLUDE_DIALOGUES)
    P(f"라벨 결함 대화 제외: {', '.join(exd) if exd else '없음(--no-exclude)'} — test 발화 " + " · ".join(f"{KO[tr]} {EXN[tr]}" for tr in REGIONS)
      + ". 대조 표는 빼기 전 값, 나머지 표는 뺀 뒤 값이다.\n")
    P("## 대조 — 보고 JSON 의 wer_micro vs CSV 재채점 (CER 도 함께)")
    P("| test | 모델 | JSON WER | 재계산 WER | JSON CER | 재계산 CER | JSON n | 재계산 n |")
    P("|---|---|---:|---:|---:|---:|---:|---:|")
    bad = 0
    for tr, model, jw, w, jc, c, jn, n in checks:
        ok = abs(jw - w) < 0.006 and abs(jc - c) < 0.006 and jn == n
        bad += not ok
        P(f"| {tr} | {model} | {jw:.2f} | {w:.2f} | {jc:.2f} | {c:.2f} | {jn} | {n} |{'' if ok else ' ❌'}")
    P(f"\n불일치 {bad}건\n")

    P("## 15.1 방언별 LoRA 튜닝 — test WER (%, micro) · §5 대응")
    P("| 지역 | 무튜닝 전체 | 튜닝 전체 | 무튜닝 60대 미만 | 튜닝 60대 미만 | 무튜닝 60대 이상 | 튜닝 60대 이상 |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for tr in REGIONS:
        b, s = T1[tr]["none"], T1[tr]["self"]
        P(f"| {KO[tr]} | {b[0]:.2f} | **{s[0]:.2f}** | {b[1]:.2f} | {s[1]:.2f} | {b[2]:.2f} | {s[2]:.2f} |")
    P(f"| 서울(표준어, 참고 · val) | {s_all[0]:.2f} | — | {s_u[0]:.2f} | — | {s_o[0]:.2f} | — |")
    P("")
    P("발화 수(채점 포함): " + " · ".join(f"{KO[tr]} {T1[tr]['self'][3]:,} (u60 {T1[tr]['self'][4]:,} · o60 {T1[tr]['self'][5]:,})" for tr in REGIONS)
      + f" · 서울 val {s_all[2]:,} (u60 {s_u[2]:,} · o60 {s_o[2]:,})")
    P("")
    P(f"## 15.2 통합 모델(64h) — 전체 test WER · §6 대응 (부트스트랩 {a.boot:,}회 · seed {a.seed})")
    P("| 지역 | 무튜닝 | 방언별 | 통합 | 통합 − 방언별 [95% 구간] |")
    P("|---|---:|---:|---:|---|")
    for tr in REGIONS:
        b, s, p, pt, lo, hi, N = T2[tr]
        cell = f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        P(f"| {KO[tr]} | {b:.2f} | {s:.2f} | {p:.2f} | {('**' + cell + '**') if (lo > 0 or hi < 0) else cell} |")
    P("")
    P("## 15.3 통합 모델 — 연령별 test WER · §14.1 대응")
    P("| 지역 | 60대 미만 무튜닝 | 60대 미만 방언별 | 60대 미만 통합 | 60대 이상 무튜닝 | 60대 이상 방언별 | 60대 이상 통합 |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for tr in REGIONS:
        P(f"| {KO[tr]} | " + " | ".join(f"{x:.2f}" for x in T21[tr]) + " |")
    P("")
    P("## 15.4 교차 채점 — 행은 test 지역, 열은 학습한 모델 (test WER) · §8 대응")
    P("| test ＼ 모델 | 무튜닝 | 경상 | 충청 | 전라 | 강원 | 제주 | 통합 |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tr in REGIONS:
        row = T3[tr]
        cs = [f"{row['none']:.2f}"] + [(f"**{row[mr]:.2f}**" if mr == tr else f"{row[mr]:.2f}") for mr in REGIONS] + [f"{row['pooled']:.2f}"]
        P(f"| {KO[tr]} | " + " | ".join(cs) + " |")
    P("")
    P("자기 방언 모델 개선폭 중 다른 방언 모델이 얻은 비율 (= (무튜닝 − 다른 모델) / (무튜닝 − 자기 모델), WER 기준):")
    P("| test | 경상 모델 | 충청 모델 | 전라 모델 | 강원 모델 | 제주 모델 |")
    P("|---|---:|---:|---:|---:|---:|")
    ratios = {}
    for tr in REGIONS:
        row = T3[tr]; own = row["none"] - row[tr]
        ratios[tr] = {mr: (None if mr == tr else round(100 * (row["none"] - row[mr]) / own)) for mr in REGIONS}
        P(f"| {KO[tr]} | " + " | ".join("—" if mr == tr else f"{ratios[tr][mr]}%" for mr in REGIONS) + " |")

    text = "\n".join(L) + "\n"
    sys.stdout.write(text)
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(text, encoding="utf-8")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps({
            "스크립트": "src/finetune/aggregate_dialect5_wer.py", "boot": a.boot, "seed": a.seed,
            "규칙": "asr_metrics.normalize 뒤 공백 분리 어절 편집거리 · micro = Σ거리/Σ정답어절 · PASS 이고 정답이 빈 발화 아닌 것만",
            "대조_불일치": bad, "대조": checks, "제외_대화": exd, "제외_발화수": {KO[tr]: EXN[tr] for tr in REGIONS},
            "방언별": {KO[tr]: {KO[m]: dict(zip(("all", "u60", "o60", "n_all", "n_u60", "n_o60"), v)) for m, v in T1[tr].items()} for tr in REGIONS},
            "통합_구간": {KO[tr]: dict(zip(("무튜닝", "방언별", "통합", "차", "lo", "hi", "n_pairs"), T2[tr])) for tr in REGIONS},
            "교차": {KO[tr]: {KO[m]: v for m, v in T3[tr].items()} for tr in REGIONS},
            "대체율": {KO[tr]: {KO[m]: v for m, v in ratios[tr].items()} for tr in REGIONS},
            "서울_val_무튜닝": {"all": s_all[0], "u60": s_u[0], "o60": s_o[0], "n": s_all[2]},
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
