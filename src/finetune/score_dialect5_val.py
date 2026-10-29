#!/usr/bin/env python3
"""방언 5지역 v2 val 세트 CER 채점 — LoRA 어댑터 하나를 얹어 val 전체를 받아쓰고 CER 을 낸다 (`docs/GYEONGSANG_LORA_SEARCH_PLAN_2026-09-27.md` §5 8번).

  왜   스모크·본학습은 val loss 로 고르고(키페이퍼 §3.2), CER 은 본학습 뒤 후보의 best 어댑터에만 잰다.
       기존 `--val-hook`(val_hook.py)은 track2 val 세트 전용이라 방언 5지역 manifest 에는 못 쓴다 (2026-09-28).
  뭘   manifest(`utterance_id,clip_file,duration_label_sec,label_text`)를 한 줄씩 읽어 `track2_asr.build(adapter)` 로 받아쓰고
       `asr_metrics.score_one/aggregate` 로 채점한다 — 추론·채점 규칙은 그 두 정본을 그대로 쓴다(채점 규칙 복제 금지).
       배치 1 · 어댑터 병합 없음 — 9/12 시험지 채점과 같은 조건이다.
  출력 `--out` JSON(요약: micro/macro CER·WER·SER, 어댑터 sha256, manifest sha256, RTFx) + 같은 이름 `.csv`(발화별).

  예)
    python3 src/finetune/score_dialect5_val.py --manifest $DATASET_ROOT/sets/gyeongsang_val/manifest.csv \
        --adapter $ACCENTURY_DATA/work/finetune/dialect5_lora/gyeongsang_GS-07/step001176 \
        --out registry/finetune/dialect5_lora/gyeongsang_GS-07/val_cer_step001176.json
    # 배선 확인 (맥 · whisper-tiny · 토이 어댑터)
    python3 src/finetune/score_dialect5_val.py --repo openai/whisper-tiny --manifest .../lora_toy_subset/manifest.csv --limit 6 --adapter <tiny 어댑터> --out /tmp/x.json
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt import track2_asr as A                     # 추론 정본 (adapter 얹기 · batch 1 · forced korean/transcribe)
from stt.asr_metrics import score_one, aggregate    # 채점 정본

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--adapter", default=None, help="train_lora_whisper.py 가 저장한 어댑터 폴더 (step*/ 또는 epochNN/). 비우면 무튜닝 기준선 (원본 large-v3)")
    ap.add_argument("--out", required=True, help="요약 JSON 경로. 발화별 CSV 는 같은 이름 .csv")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N 발화만 (0 = 전부)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", default=None, help="배선 확인용 (whisper-tiny 등). 실제 채점은 기본값(large-v3)")
    ap.add_argument("--label", default="", help="보고에 남길 이름 (기본: 어댑터 상위 폴더/폴더명)")
    ap.add_argument("--prompt-file", default=None, help="Whisper 힌트(앞 문장 칸) 글 파일 — 방언 철자 힌트 실험용 (2026-10-07). 비우면 힌트 없음(옛 동작)")
    a = ap.parse_args()
    prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip() if a.prompt_file else None

    man = Path(a.manifest)
    base = man.parent
    rows = [r for r in csv.DictReader(open(man, encoding="utf-8", newline=""))]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        sys.exit(f"manifest 가 비었다: {man}")
    if a.repo:
        A.REPO = a.repo
    device = A.pick_device(a.device)
    t0 = time.time()
    run, repo, dtype, ad = A.build(device, adapter=a.adapter, prompt=prompt)
    load_s = round(time.time() - t0, 1)
    label = a.label or (f"{Path(a.adapter).resolve().parent.name}/{Path(a.adapter).name}" if a.adapter else "baseline(무튜닝)")
    print(f"모델 {repo} {dtype} {device} · 어댑터 {ad['sha256'][:12] if ad else '없음(무튜닝 기준선)'} · 로드 {load_s}s · {len(rows)} 발화 · {label}", flush=True)

    scored, fails, audio_sec = [], 0, 0.0
    t1 = time.time()
    out_csv = Path(a.out).with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "duration_label_sec", "status", "reference", "hypothesis", "cer", "wer", "ser", "ref_chars", "error"])
        for i, r in enumerate(rows, start=1):
            ref = r.get("label_text", "")
            hyp, status, err = "", "PASS", ""
            try:
                x, sr = sf.read(str(base / r["clip_file"]), dtype="float32", always_2d=False)
                if x.ndim != 1 or sr != SR:
                    raise ValueError(f"16 kHz mono 가 아니다 ({sr} Hz · ndim {x.ndim})")
                audio_sec += len(x) / SR
                hyp = run(x)
            except Exception as e:                        # noqa: BLE001 — 실패는 FAIL 로 기록하고 계속 (§4.4)
                status, err, fails = "FAIL", f"{type(e).__name__}: {e}"[:200], fails + 1
            s = score_one(ref, hyp, status=status, kspon=False)
            scored.append(s)
            w.writerow([r.get("utterance_id", ""), r.get("duration_label_sec", ""), s["status"], ref, hyp,
                        s.get("cer", ""), s.get("wer", ""), s.get("ser", ""), s["ref_chars"], err])
            if i % 100 == 0 or i == len(rows):
                el = time.time() - t1
                print(f"  {i}/{len(rows)}  {el/60:.1f}m  RTFx {audio_sec/max(el,1e-6):.2f}  실패 {fails}", flush=True)
    wall = time.time() - t1
    agg = aggregate(scored)
    rep = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "스크립트": "src/finetune/score_dialect5_val.py", "label": label,
           "모델": repo, "dtype": dtype, "장치": device, "어댑터": ad,
           "manifest": str(man), "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
           "발화": len(rows), "오디오_초": round(audio_sec, 1), "벽시계_초": round(wall, 1), "RTFx": round(audio_sec / max(wall, 1e-6), 2),
           "로드_초": load_s, "실패": fails,
           **({"힌트": {"file": a.prompt_file, "sha256": hashlib.sha256(Path(a.prompt_file).read_bytes()).hexdigest(), "text": prompt}} if prompt else {}),
           "채점": agg,
           "규칙": "asr_metrics.normalize 뒤 글자 단위 편집거리 · micro = Σ거리/Σ정답글자 · 배치 1 · 어댑터 병합 없음 · 정답이 빈 발화 제외",
           "발화별_csv": str(out_csv)}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"끝 — CER micro {agg.get('cer_micro')} · macro {agg.get('cer_macro')} · WER micro {agg.get('wer_micro')} · n {agg.get('n_scored')}/{len(rows)} · 실패 {fails} · {wall/60:.1f}m · RTFx {rep['RTFx']} → {a.out}", flush=True)


if __name__ == "__main__":
    main()
