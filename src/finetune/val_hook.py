#!/usr/bin/env python3
"""학습 도중 체크포인트 하나를 val 로 채점한다 — `train_lora_whisper.py` 가 부르는 훅.

왜   epoch 는 논문에서 빌릴 수 없어(스위스독일어 약 1.1 대 LoRA-Whisper 10) val 로 고른다
     (`docs/TRACK2_BENCHMARK_V2_SPEC.md` §4.3). 트레이너는 채점 규칙을 갖지 않기로 했고
     (그 파일 「왜 in-loop 채점이 아니라 훅인가」), 이 스크립트가 그 사이를 잇는다.
계약 트레이너가 `val_hook.py <체크포인트_디렉터리> <epoch>` 로 부른다.
     이 스크립트는 `<체크포인트_디렉터리>/val_score.json` 에 `{"고령_micro_cer": <float>, ...}` 를 쓴다.
     트레이너가 읽는 키는 `고령_micro_cer` 하나뿐이고 나머지는 해석 없이 보고 JSON 에 옮겨 적힌다.
뭘   ① 그 어댑터를 얹어 val 8세트를 추론하고(`run_track2_stt.py --adapter`),
     ② `score_val_track2.py` 로 채점한 뒤,
     ③ 판정 묶음(고령 3칸 합산 micro · 경상 10~40대 · 서울 3칸)을 뽑아 위 파일에 쓴다.
주의 - **탈락·선택 판정을 여기서 하지 않는다.** §4.3 의 3단 규칙은 학습이 끝난 뒤
       `score_val_track2.py --select` 가 체크포인트 전부를 놓고 돌린다. 훅은 숫자만 낸다.
     - 조각은 epoch 마다 다른 폴더에 쓴다. 어댑터가 다른 조각이 한 폴더에 섞이면 그 폴더의 CER 은
       두 모델의 평균이 된다 (`run_track2_stt.py` 가 adapter sha256 으로 막지만 폴더부터 나눈다).
     - 추론이 val 8.35h 라 GPU 로 약 50분 걸린다. 2 epoch 마다 부르는 비용이 여기서 나온다.
     - 실패해도 0 이 아닌 코드로 끝내기만 한다. 학습을 죽일지는 트레이너가 `--val-hook-fatal` 로 정한다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))
BENCH = "track2_bench_v2_val"
PICK = "고령_3칸"                  # §4.3 4·5번의 지표. 트레이너가 이 값으로 patience 를 센다.


def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="어댑터 폴더 (train_lora_whisper.py 가 저장한 epochNN/)")
    ap.add_argument("epoch", type=int)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--cond", default="A")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shard-root", default=None,
                    help="조각을 쓸 자리 (기본 $ACCENTURY_DATA/work/finetune/val_shards/<run>)")
    ap.add_argument("--run-label", default=None, help="조각·registry 파일 이름의 앞머리 (기본: 체크포인트 부모 폴더 이름)")
    a = ap.parse_args()

    ckpt = Path(a.ckpt).resolve()
    if not (ckpt / "adapter_config.json").exists():
        sys.exit(f"어댑터 폴더가 아니다: {ckpt}")
    runlabel = a.run_label or ckpt.parent.name
    label = f"{runlabel}_ep{a.epoch:02d}"
    shard_root = Path(a.shard_root) if a.shard_root else V / "work/finetune/val_shards" / runlabel
    shard_dir = shard_root / f"ep{a.epoch:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    logp = shard_dir / "hook.log"
    t0 = time.time()

    with open(logp, "a", encoding="utf-8") as log:
        log.write(f"=== {label} · {time.strftime('%F %T')} · ckpt {ckpt}\n")
        rc = run([sys.executable, "src/stt/run_track2_stt.py", "--bench", a.bench, "--all",
                  "--cond", a.cond, "--adapter", str(ckpt), "--out-dir", str(shard_dir),
                  "--device", a.device], log)
        if rc:
            sys.exit(f"추론 실패 (exit {rc}) — {logp}")
        rc = run([sys.executable, "src/stt/score_val_track2.py", "--model", f"{label}={shard_dir}",
                  "--cond", a.cond], log)
        if rc:
            sys.exit(f"채점 실패 (exit {rc}) — {logp}")

    res = ROOT / "registry/benchmark" / f"{a.bench}_stt__{label}.json"
    if not res.exists():
        sys.exit(f"채점 결과가 없다: {res}")
    j = json.loads(res.read_text(encoding="utf-8"))
    b = j.get("묶음", {})
    pick = (b.get(PICK) or {}).get("cer")
    if pick is None:
        sys.exit(f"{PICK} CER 이 비었다 — {res}")

    out = {
        "고령_micro_cer": pick,                       # 트레이너가 읽는 유일한 키
        "epoch": a.epoch, "라벨": label,
        "채점_결과": str(res.relative_to(ROOT)),
        "조각": str(shard_dir).replace(str(Path.home()), "~"),
        "걸린_초": round(time.time() - t0, 1),
        # 아래는 트레이너가 해석 없이 보고 JSON 에 옮기기만 한다. 판정은 학습이 끝난 뒤
        # `score_val_track2.py --select` 가 체크포인트 전부를 놓고 한다 (§4.3 3단 규칙).
        "묶음": {k: {"cer": v.get("cer"), "ci_group": v.get("ci_group"), "역할": v.get("역할")}
               for k, v in b.items()},
        "경고": j.get("경고", []),
    }
    (ckpt / "val_score.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"{label}: {PICK} micro CER {pick:.2f} · {out['걸린_초']:.0f}s → {ckpt / 'val_score.json'}")


if __name__ == "__main__":
    main()
