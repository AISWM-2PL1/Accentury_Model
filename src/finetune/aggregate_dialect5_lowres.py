#!/usr/bin/env python3
"""방언 저자원 A·B 24개 실험의 test CER·WER 표 (2026-10-05, `docs/DIALECT5_LOW_RESOURCE_RESULTS_2026-10-05.md`).

  왜   러너(`run_dialect5_lowres.sh`)는 실험마다 test 채점 보고(`test_cer_<step>.{json,csv}`)와 `JOB_DONE.json` 을 S3 에 남긴다.
       채점 보고의 요약은 라벨 결함 대화를 빼기 전 값이고, 기준값(무튜닝·기존 통합 B 12.8h)은 다른 run 폴더에 있다.
       이 스크립트가 같은 규칙으로 한 표에 모은다.
  뭘   ① 실행 설정(`registry/finetune/dialect5_low_resource_runs_2026-10-04.json`)의 24개 run 과 지역별 기준 2개(무튜닝 · 통합 step021244)를
          `aggregate_dialect5_wer.py` 의 `per_utt`·`micro`·`excluded`·`drop` 으로 다시 채점한다(채점 규칙 복제 금지).
       ② 대조: 빼기 전 재채점 CER·WER·n 이 채점 보고 JSON 요약과 같은지 본다(다르면 ❌).
       ③ 라벨 결함 대화(`dialect5_exclusions`)를 뺀 표 — 지역 × 조건(무튜닝 · B 0/1.6/3.2/6.4/12.8h · A 3.2/12.8h).
       ④ run 별 기록: `JOB_DONE.json` 의 best step·val loss·global step·학습 벽시계와 실행 설정의 epoch당 업데이트로 종료 epoch.
       조건마다 학습은 1회다. 구간(부트스트랩)은 이 스크립트에서 내지 않는다.
  입력 `--rep-dir`: S3 `results_dialect5/` 사본 — `<region>/<run>/report/{JOB_DONE.json, test_cer_<step>.{json,csv}}`
       `--man-dir`: test manifest 5개(`<region>_test.csv`)
  출력 `--out-md` · `--out-json`

  예)
    aws s3 sync s3://accentury-track2-325771561913/results_dialect5/ /tmp/rep --exclude "*" \
        --include "*_[AB]*h_GS-09/report/JOB_DONE.json" --include "*_[AB]*h_GS-09/report/test_cer_*" \
        --include "*_pooledGS-09/report/test_cer_step021244.*" --include "*_GS-01/report/test_cer_none.*" --include "*/gyeongsang_baseline/report/test_cer_none.*"
    python3 src/finetune/aggregate_dialect5_lowres.py --rep-dir /tmp/rep --man-dir /tmp/man \
        --out-md registry/finetune/dialect5_low_resource_results_2026-10-05.md --out-json registry/finetune/dialect5_low_resource_results_2026-10-05.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.aggregate_dialect5_wer import KO, REGIONS, excluded, drop, json_summary, micro, per_utt, report_path  # noqa: E402
from finetune.dialect5_exclusions import EXCLUDE_DIALOGUES  # noqa: E402

RUNS_CFG = Path(__file__).resolve().parents[2] / "registry/finetune/dialect5_low_resource_runs_2026-10-04.json"
TAG = {0: "0h", 1.6: "1p6h", 3.2: "3p2h", 6.4: "6p4h", 12.8: "12p8h"}
COLS = [("none", "무튜닝"), ("B0", "B 0h"), ("B1.6", "B 1.6h"), ("B3.2", "B 3.2h"), ("B6.4", "B 6.4h"), ("B12.8", "B 12.8h (기존 통합)"),
        ("A3.2", "A 3.2h"), ("A12.8", "A 12.8h")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-dir", required=True)
    ap.add_argument("--man-dir", required=True)
    ap.add_argument("--runs-config", default=str(RUNS_CFG), help="실행 설정 (10/6 A 3.2h 확장은 …runs_2026-10-06.json)")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    rep, man = Path(a.rep_dir), Path(a.man_dir)
    config_path = Path(a.runs_config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    checks, table, runs, nex = [], {}, [], {}
    for region in REGIONS:
        ex = excluded(man / f"{region}_test.csv")
        nex[region] = len(ex)
        cells = {}
        sources = {"none": report_path(rep, region, "none"), "B12.8": report_path(rep, region, "pooled")}
        for r in cfg["runs"]:
            if r["region"] != region:
                continue
            done = json.loads((rep / region / r["run_id"] / "report" / "JOB_DONE.json").read_text(encoding="utf-8"))
            key = f"{r['mode']}{r['target_train_hours']:g}"
            base = rep / region / r["run_id"] / "report" / f"test_cer_{done['best_step']}"
            sources[key] = (base.with_suffix(".json"), base.with_suffix(".csv"))
            spe = r["train"]["steps_per_epoch"]
            runs.append({"run_id": r["run_id"], "region": region, "mode": r["mode"], "target_train_hours": r["target_train_hours"],
                         "train_rows": r["train"]["rows"], "steps_per_epoch": spe, "best_step": done["best_step"],
                         "best_epoch_pos": round(int(done["best_step"][4:]) / spe, 2), "best_val_loss": done["best_val_loss"],
                         "global_step": done["global_step"], "end_epoch_pos": round(done["global_step"] / spe, 2),
                         "train_wall_h": done["벽시계_h"], "stop_reason": done["종료_이유"], "test_manifest_sha256": done["test"]["manifest_sha256"],
                         "adapter_sha256": done["test"].get("어댑터_sha256"), "done": done["done"]})
        for key, (jp, cp) in sources.items():
            jw, jc, jn = json_summary(jp)
            d = per_utt(cp)
            w0, c0, n0 = micro(d)
            checks.append({"region": region, "cond": key, "json": [jc, jw, jn], "recomputed": [c0, w0, n0], "ok": [jc, jw, jn] == [c0, w0, n0]})
            w, c, n = micro(drop(d, ex))
            cells[key] = {"cer": c, "wer": w, "n": n, "source": str(cp.relative_to(rep))}
        table[region] = cells

    L = []
    P = L.append
    P(f"라벨 결함 대화 제외: {', '.join(sorted(EXCLUDE_DIALOGUES))} — test 발화 " + " · ".join(f"{KO[r]} {nex[r]}" for r in REGIONS)
      + ". 대조 표는 빼기 전 값, 나머지 표는 뺀 뒤 값이다. 단위 %(micro). 조건마다 학습 1회.\n")
    for metric, title in (("cer", "CER"), ("wer", "WER")):
        P(f"## test {title}")
        P("| 지역 | " + " | ".join(h for _, h in COLS) + " | n |")
        P("|---|" + "---:|" * (len(COLS) + 1))
        for region in REGIONS:
            c = table[region]
            P(f"| {KO[region]} | " + " | ".join(f"{c[k][metric]:.2f}" if k in c else "" for k, _ in COLS) + f" | {c['none']['n']} |")
        P("")
    P("## run 별 기록 (JOB_DONE.json · 실행 설정)")
    P("| run | train 발화 | epoch당 업데이트 | best step (epoch 위치) | best val loss | 종료 step (epoch 위치) | 학습 h |")
    P("|---|---:|---:|---|---:|---|---:|")
    for r in runs:
        P(f"| {r['run_id']} | {r['train_rows']} | {r['steps_per_epoch']} | {r['best_step']} ({r['best_epoch_pos']}) | {r['best_val_loss']:.4f} | "
          f"{r['global_step']} ({r['end_epoch_pos']}) | {r['train_wall_h']} |")
    P("")
    bad = [c for c in checks if not c["ok"]]
    P(f"## 대조 — 채점 보고 JSON 요약 vs CSV 재채점 (빼기 전) — {len(checks) - len(bad)}/{len(checks)} 일치")
    P("| test | 조건 | JSON CER | 재계산 CER | JSON WER | 재계산 WER | JSON n | 재계산 n | |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for c in checks:
        P(f"| {KO[c['region']]} | {c['cond']} | {c['json'][0]} | {c['recomputed'][0]} | {c['json'][1]} | {c['recomputed'][1]} | {c['json'][2]} | {c['recomputed'][2]} | {'✅' if c['ok'] else '❌'} |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.out_md:
        Path(a.out_md).write_text(md, encoding="utf-8")
    if a.out_json:
        config_source = str(config_path.relative_to(RUNS_CFG.parents[2])) if config_path.is_relative_to(RUNS_CFG.parents[2]) else str(config_path)
        Path(a.out_json).write_text(json.dumps({"script": "src/finetune/aggregate_dialect5_lowres.py", "runs_config": config_source,
                                                "excluded_dialogues": sorted(EXCLUDE_DIALOGUES), "excluded_utterances": nex,
                                                "unit": "% micro · asr_metrics.normalize", "table": table, "runs": runs, "checks": checks},
                                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if bad:
        sys.exit(f"대조 불일치 {len(bad)}건")


if __name__ == "__main__":
    main()
