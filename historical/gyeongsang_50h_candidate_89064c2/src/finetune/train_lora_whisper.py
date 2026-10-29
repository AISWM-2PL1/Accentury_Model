#!/usr/bin/env python3
"""whisper-large-v3 + LoRA 학습 — 본 학습(`docs/TRACK2_BENCHMARK_V2_SPEC.md` §5 6번) · 토이 측정(§5 4번) 겸용

  왜   epoch 는 논문에서 빌릴 수 없다(스위스독일어 약 1.1 대 LoRA-Whisper 10). 상한 15 epoch 까지 돌리며
       2 epoch 마다 어댑터를 남기고, 그중 하나를 val 로 고른다(명세 §4.3 「체크포인트 선택 규칙」).
       이 스크립트는 **남기는 일까지** 한다 — 고르는 일은 하지 않는다(아래 「채점」 참조).
  뭘   교재 manifest(`label_text` 열)로 LoRA 어댑터를 epoch 단위로 학습한다.
       하이퍼파라미터 정본은 `docs/FINETUNE_METHOD_SELECTION.md` §5.4.1 과 그 아래 정정 블록이고,
       이 파일의 기본값은 그 표를 그대로 옮긴 것이다(각 인자 help 에 근거를 적었다).
       `--max-steps` 를 주면 **step 모드**로 떨어진다 — 토이 실측(§5 4번)과 CPU 스모크가 쓰는 길이고,
       체크포인트·조기 종료·LR 스케줄이 전부 꺼진다.
  입력 manifest CSV (`utterance_id,clip_file,duration_label_sec,label_text` 열을 쓴다) ·
       그 옆의 `clips/` (16 kHz mono WAV)
  출력 `--out` 아래 `epoch{NN}/` 어댑터(peft `save_pretrained` — 어댑터만, 원본 가중치 없음) +
       각 체크포인트의 `run_kind.json` · `--report` JSON(epoch 별 loss·처리량·피크 메모리·체크포인트 경로·
       실제로 쓴 하이퍼파라미터 전부). 보고 JSON 은 **epoch 마다 덮어쓴다** — 중간에 죽어도 근거가 남는다.

  주의
  - 채점: 이 스크립트는 **val 을 채점하지 않는다.** 채점기는 별도로 만들고 있고(2026-09-03 기준 미완성),
    여기서는 `--val-hook` 인터페이스만 정의한다. 아래 「val 훅 계약」 참조. 훅이 없으면 조기 종료도 없고
    상한 epoch 까지 돌린 뒤 체크포인트를 전부 남긴다 — 고르기는 나중에 오프라인으로 한다.
  - 결정론: seed 는 torch·python random·numpy 셋 다 고정한다(SpecAugment 마스크가 numpy 전역 RNG 를 쓴다).
    cudnn deterministic 은 설정하지 않는다(코덱스 2차 13번 기록 유지).
  - 재시작 (2026-09-09 · 코덱스 4차 8번): epoch 모드는 체크포인트를 저장할 때마다 `<--out>/train_state.pt` 에
    optimizer·scheduler·RNG(torch·cuda·python·numpy)·에폭 기록·설정 검증값을 덮어쓴다. `--resume` 을 주면 그 파일과
    `epochNN/` 어댑터를 읽어 다음 epoch 부터 잇는다 (설정이 다르면 거부). **`--out` 이 비어 있지 않으면 `--resume` 없이는 시작하지 않는다**
    — 같은 폴더에 다시 돌려 이전 epoch·val_score·shard 가 섞이는 사고를 막는다. 데이터 순서는 `Batches.epoch(seed, ep)` 가
    epoch 마다 새로 섞으므로 재개 epoch 의 순서는 처음 돌렸을 때와 같다.

  val 훅 계약 (`--val-hook`)
    체크포인트를 저장한 **직후** 트레이너가 `<--val-hook 로 준 명령> <체크포인트_디렉터리> <epoch>` 을
    subprocess 로 부른다(cwd = repo 루트). 훅은 채점을 마치고 `<체크포인트_디렉터리>/val_score.json` 을
    남겨야 한다. 트레이너가 **읽는 키는 하나뿐**이다:
        {"고령_micro_cer": <float>}     # 명세 §4.3 4번 — 경상 60·70·80대+ 를 글자 가중 합산한 micro CER
    나머지 키(`탈락`·`경고`·`세트별`·`무튜닝_고령_micro_cer` …)는 **그대로 보고 JSON 에 옮겨 적기만** 하고
    해석하지 않는다. hard veto(§4.3 2번)와 선택(4번)은 학습 밖의 판단이라 여기서 하지 않는다.
    훅이 실패하거나 파일이 없으면 그 epoch 은 「채점 안 됨」으로 기록하고 **학습은 계속한다**
    (채점기 사고로 18h 짜리 학습을 죽이지 않는다 — 체크포인트는 디스크에 있으니 나중에 채점하면 된다).
    그 판단을 뒤집으려면 `--val-hook-fatal`.
    ~~훅으로 쓸 스크립트는 아직 없다~~ **`src/finetune/val_hook.py` 가 그 글루다 (2026-09-03 신설).**
    「어댑터로 val 8세트 추론(`run_track2_stt.py --adapter`) → `score_val_track2.py` 채점 →
    `val_score.json` 쓰기」를 잇는다. `--val-hook "python3 src/finetune/val_hook.py"` 로 준다.
    같이 뚫은 것 둘 — `track2_data.set_bench` 가 `work/finetune` 도 찾고(val 세트가 거기 있다),
    `track2_asr.build(device, adapter)` 가 어댑터를 얹는다(병합하지 않고, sha256 을 조각 메타에 남긴다).

  왜 in-loop 채점이 아니라 훅인가 (2026-09-03 판단 · 근거를 남긴다)
    1. 채점 규칙을 복제하게 된다. 추론은 `src/stt/run_track2_stt.py`, 채점은 `src/stt/asr_metrics.py`·
       `score_track2.py` 가 정본이고 이 repo 규칙은 「채점 규칙 복제 금지」다. in-loop 로 하려면 8 세트
       루프·정규화·클러스터 부트스트랩을 학습 스크립트가 다시 갖게 된다.
    2. 채점기가 아직 없다. in-loop 는 없는 함수를 import 해야 하고, 그러면 지금 이 파일을 돌릴 수 없다.
    3. 메모리. 토이 실측 피크가 L40S 44.39 GiB 중 36.74 GiB 다(`registry/finetune/lora_toy/lora_toy_b8.json`).
       같은 프로세스에서 `generate`(use_cache=True, KV cache)를 돌리면 학습 피크 위에 얹힌다 —
       batch 16 이 470 MiB 부족으로 OOM 났던 여유다. 프로세스를 나누면 학습 피크가 채점에 종속되지 않는다.
    4. 대신 잃는 것: 훅은 프로세스 기동·모델 재로드 비용을 매번 낸다(토이 실측 로드 294.9s / 캐시 후 6.0s).
       2 epoch 에 한 번이라 8 회 이하이므로 감수한다.

  예)
    # 본 학습 — EC2 에서는 `src/finetune/run_lora_train.sh v2_train_weighted` 가 이 명령을 감싼다
    # (§5.4.1 확정값이 전부 기본값이라 하이퍼파라미터를 따로 주지 않는다)
    python3 src/finetune/train_lora_whisper.py --run-kind main --limit 0 \
        --manifest $ACCENTURY_DATA/work/finetune/track2_bench_v2/sets/v2_train_weighted/manifest.csv \
        --out $ACCENTURY_DATA/work/finetune/lora_main_v2_train_weighted \
        --report registry/finetune/lora_main/v2_train_weighted/lora_main_v2_train_weighted.json

    # 토이 실측 재현 (step 모드 · 2026-09-02 실측과 같은 설정 — 바뀐 기본값을 전부 되돌린다)
    python3 src/finetune/train_lora_whisper.py --manifest $SUB/manifest.csv --limit 0 --max-steps 20 --batch 8 \
        --grad-accum 1 --lr 1e-4 --alpha 32 --weight-decay 0 --label-smoothing 0 --no-spec-augment \
        --lr-schedule constant --warmup-steps 0 --out work/finetune/lora_toy --report registry/finetune/lora_toy_b8.json
"""
import argparse
import csv
import hashlib
import json
import math
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.asr_metrics import score_one, aggregate          # 채점 규칙 복제 금지

ROOT = Path(__file__).resolve().parents[2]
REPO = "openai/whisper-large-v3"                          # inference 명세 §3 — 새로 고를 항목이 아니다
SR = 16000
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]   # 전 linear (FINETUNE_METHOD_SELECTION §5.4.1)


def load_rows(manifest: Path, limit: int, seed: int, max_sec: float):
    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8")) if r.get("label_text", "").strip()]
    rows = [r for r in rows if float(r["duration_label_sec"]) < max_sec]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def load_wav(path: Path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim != 1 or sr != SR:
        raise ValueError(f"16 kHz mono 가 아니다: {path} ({sr} Hz · ndim {x.ndim})")
    return x


def dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


class Batches:
    """manifest 행 → (input_features, labels). 오디오는 매 배치 디스크에서 읽는다 (캐시 없음).

    두 가지 순회가 있다.
      `__iter__`   무한 반복 — step 모드 전용. **토이 실측을 재현해야 하므로 손대지 않는다**
                   (전역 `random.shuffle` 을 쓰는 것까지 그대로다).
      `epoch`      한 epoch 분만 — 본 학습용. 전역 RNG 를 건드리지 않으려고 epoch 마다 별도
                   `random.Random(seed*1000+epoch)` 을 쓴다. 배치에 못 채우는 꼬리는 버린다.
    """

    def __init__(self, rows, base, proc, batch, device):
        self.rows, self.base, self.proc, self.batch, self.device = rows, base, proc, batch, device
        self.tok = proc.tokenizer
        self.tok.set_prefix_tokens(language="korean", task="transcribe")
        self.start_id = None

    def __iter__(self):
        while True:                                      # max-steps 가 끝을 정한다
            order = list(range(len(self.rows)))
            random.shuffle(order)
            for i in range(0, len(order) - self.batch + 1, self.batch):
                yield self.make([self.rows[j] for j in order[i:i + self.batch]])

    def n_micro(self) -> int:
        """1 epoch 의 마이크로 배치 수 (꼬리 버림)."""
        return len(self.rows) // self.batch

    def epoch(self, seed: int, epoch: int):
        order = list(range(len(self.rows)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        for i in range(0, self.n_micro() * self.batch, self.batch):
            yield self.make([self.rows[j] for j in order[i:i + self.batch]])

    def make(self, rs):
        audio = [load_wav(self.base / r["clip_file"]) for r in rs]
        feats = self.proc.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
        ids = [self.tok(r["label_text"]).input_ids for r in rs]
        # HF 관례: 라벨 첫 토큰이 decoder_start(<|startoftranscript|>) 면 뗀다 — 모델이 shift 하며 다시 붙인다
        if self.start_id is not None:
            ids = [x[1:] if x and x[0] == self.start_id else x for x in ids]
        L = max(len(x) for x in ids)
        labels = torch.full((len(ids), L), -100, dtype=torch.long)
        for k, x in enumerate(ids):
            labels[k, :len(x)] = torch.tensor(x)
        sec = sum(len(a) for a in audio) / SR
        return feats.to(self.device), labels.to(self.device), sec, sum(len(x) for x in ids)


def forward_loss(model, feats, labels, pad_id, start_id, smoothing, device, shift):
    """마이크로 배치 1회 forward. 반환: 평균 CE loss(라벨 토큰 기준).

    label smoothing 을 걸려고 **모델 내부 loss 를 쓰지 않는다.** transformers 4.57.6 의
    `WhisperForConditionalGeneration.forward` 는 `CrossEntropyLoss()` 를 인자 없이 쓴다
    (`models/whisper/modeling_whisper.py:1311`) — smoothing 을 넣을 자리가 없다.
    그래서 `labels=` 대신 `decoder_input_ids=`(같은 파일의 `shift_tokens_right`)를 직접 주고
    `F.cross_entropy(..., label_smoothing=)` 로 바깥에서 잰다. smoothing 0 이면 모델 내부와 같은 식이다
    (둘 다 ignore_index=-100 · reduction='mean'), 즉 토이 실측의 loss 정의가 유지된다.
    """
    dec_in = shift(labels, pad_id, start_id)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        out = model(input_features=feats, decoder_input_ids=dec_in)
        loss = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, out.logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, label_smoothing=smoothing)
    return loss


def save_ckpt(model, out: Path, epoch: int, tag: str, meta: dict, why: str) -> dict:
    """어댑터만 저장한다. peft `PeftModel.save_pretrained` 는 LoRA 가중치와 adapter_config 만 쓴다 (원본 1.6B 는 안 쓴다).

    크기: 학습 파라미터 57,671,680개인데 peft 는 `get_peft_model(autocast_adapter_dtype=True)` 기본값으로
    bf16 어댑터를 **fp32 로 올린다**(`peft/tuners/tuners_utils.py:590 _cast_adapter_dtype` — 「only upcasts
    float16 and bfloat16 to float32」). 그러면 한 장에 약 230 MiB 이고 15 epoch 8 장이면 약 1.8 GiB 다.
    (~~bf16 115 MiB~~ 는 이 upcast 를 빼먹은 계산이다.) **GPU 실측은 아직 없다** — 실제 크기는 반환값 `바이트` 가 정본.
    """
    d = out / f"epoch{epoch:02d}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(d))
    # 어댑터 표식 — 토이 어댑터가 본 실험에 섞이지 않게 (코덱스 2차 14번)
    (d / "run_kind.json").write_text(json.dumps({"run_kind": tag, "epoch": epoch, **meta},
                                                ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # 완료 마커 — 마지막에 쓴다. 러너가 학습 중 주기적으로 S3 에 올리고 채점기가 이 파일을 보고 「다 썼다」고 판단한다 (2차 명세 §6 8번).
    (d / "_complete").write_text(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    return {"경로": str(d), "바이트": dir_bytes(d), "이유": why}


RESUME_KEYS = ["repo", "manifest", "limit", "max_sec", "seed", "epochs", "batch", "grad_accum", "lr", "r", "alpha", "dropout",
               "weight_decay", "label_smoothing", "lr_schedule", "warmup_ratio", "warmup_steps", "ckpt_every", "run_kind",
               # 코덱스 5차 7번 — 학습 동작에 영향을 주는 나머지 옵션도 검증한다
               "spec_augment", "mask_time_prob", "mask_time_length", "mask_feature_prob", "mask_feature_length",
               "max_grad_norm", "grad_ckpt", "beta1", "beta2", "eps"]


def save_state(out: Path, epoch: int, opt, sched, rep: dict, a, wall_h: float) -> dict:
    """재시작용 상태 — 어댑터(epochNN/)와 별도로 `<out>/train_state.pt` 하나에 덮어쓴다 (코덱스 4차 8번).
    AdamW 상태는 학습 파라미터의 2배(fp32)라 약 460 MiB · 어댑터 230 MiB 는 epochNN/ 에 이미 있다."""
    st = {"epoch": epoch, "opt": opt.state_dict(), "sched": sched.state_dict() if sched is not None else None,
          "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  "python": random.getstate(), "numpy": np.random.get_state()},
          "에폭": rep["에폭"], "체크포인트": rep["체크포인트"], "벽시계_h_누적": round(wall_h, 3),
          "설정_검증": {k: getattr(a, k) for k in RESUME_KEYS} | {"manifest_sha256": rep["manifest_sha256"]},
          "저장": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
    p = out / "train_state.pt"
    tmp = out / "train_state.pt.tmp"
    torch.save(st, str(tmp))
    tmp.replace(p)                                    # 쓰다 죽어도 이전 상태가 남는다
    return {"경로": str(p), "바이트": p.stat().st_size, "epoch": epoch}


def load_state(out: Path, a, manifest_sha: str) -> dict:
    """`--resume`: 상태 파일을 읽고 설정이 같은지 확인한다. 다르면 멈춘다 — 다른 설정을 이어 붙이면 run 이 아니다."""
    p = out / "train_state.pt"
    if not p.exists():
        sys.exit(f"--resume 인데 상태 파일이 없다: {p}")
    st = torch.load(str(p), map_location="cpu", weights_only=False)
    want = {k: getattr(a, k) for k in RESUME_KEYS} | {"manifest_sha256": manifest_sha}
    diff = {k: (st["설정_검증"].get(k), want[k]) for k in want if st["설정_검증"].get(k) != want[k]}
    if diff:
        sys.exit(f"--resume 설정 불일치 (저장값, 지금값): {diff}")
    ck = out / f"epoch{st['epoch']:02d}"
    if not (ck / "adapter_config.json").exists():
        sys.exit(f"--resume: 상태는 epoch {st['epoch']} 인데 어댑터 폴더가 없다: {ck}")
    return st


def call_val_hook(argv, ckpt_dir: Path, epoch: int) -> dict:
    """val 훅 호출. 반환: {"고령_micro_cer": float, ...} 또는 {"실패": "..."}.

    계약은 모듈 docstring 「val 훅 계약」에 있다. 여기서는 훅이 남긴 JSON 을 읽어 옮길 뿐,
    CER 을 계산하지도 탈락 규칙을 적용하지도 않는다.
    """
    score_path = ckpt_dir / "val_score.json"
    if score_path.exists():
        score_path.unlink()                              # 이전 실행의 잔재를 성공으로 읽지 않는다
    cmd = argv + [str(ckpt_dir), str(epoch)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT))
    except OSError as err:
        return {"실패": f"훅 실행 불가: {err}", "명령": cmd}
    took = round(time.time() - t0, 1)
    if r.returncode != 0:
        return {"실패": f"훅 exit {r.returncode}", "명령": cmd, "초": took}
    if not score_path.exists():
        return {"실패": f"훅이 {score_path.name} 을 남기지 않았다", "명령": cmd, "초": took}
    try:
        got = json.loads(score_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        return {"실패": f"{score_path.name} 파싱 실패: {err}", "명령": cmd, "초": took}
    if not isinstance(got, dict) or not isinstance(got.get("고령_micro_cer"), (int, float)):
        return {"실패": "고령_micro_cer 이 없거나 수가 아니다", "받은_키": list(got) if isinstance(got, dict) else None,
                "명령": cmd, "초": took}
    got["초"] = took
    return got


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0, help="쓰는 발화 수 (0 = 전부 — 본 학습 기본값)")
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default="work/finetune/lora_main")
    ap.add_argument("--report", required=True)
    ap.add_argument("--repo", default=REPO, help="스모크 전용 (whisper-tiny 등). 실제 실험은 기본값만")
    ap.add_argument("--run-kind", default="toy", choices=["toy", "main", "smoke"],
                    help="어댑터에 남기는 표식 — 본 학습만 main")
    ap.add_argument("--resume", action="store_true",
                    help="<--out>/train_state.pt 와 epochNN/ 어댑터에서 이어 한다 (epoch 모드 · 설정이 같아야 한다 · 코덱스 4차 8번)")
    ap.add_argument("--stop-after-epoch", type=int, default=0,
                    help="스모크 전용 — 이 epoch 의 상태를 저장한 직후 exit 4 로 죽은 척한다 (--resume 검증용)")

    g = ap.add_argument_group("§5.4.1 확정값 (FINETUNE_METHOD_SELECTION.md · 바꾸면 그 문서와 어긋난다)")
    g.add_argument("--epochs", type=int, default=15, help="상한 15 (§5.4.1 — 누적 노출로 스위스독일어와 맞춘 값). val 로 그 전에 멈춘다")
    g.add_argument("--batch", type=int, default=8, help="물리 배치. L40S 44.39 GiB 실측 상한 (16 은 OOM — lora_toy_b16.json)")
    g.add_argument("--grad-accum", type=int, default=32, help="8 × 32 = effective 256 (§5.4.1 정정 블록 C 안)")
    g.add_argument("--lr", type=float, default=1e-5, help="스위스독일어 Run 5~12 (large-v3)")
    g.add_argument("--r", type=int, default=32, help="LoRA-Whisper 최적값 (우리 50h 와 같은 자릿수)")
    g.add_argument("--alpha", type=int, default=8, help="α/r = 8/32 = 0.25 (「0.2」는 오기 — 코덱스 5차 19번). 스위스독일어 §5.2.1 의 0.2(32/160)를 옮긴 값이며 r=32 와의 조합은 미검증 전이")
    g.add_argument("--dropout", type=float, default=0.05, help="스위스독일어 Table 5")
    g.add_argument("--weight-decay", type=float, default=0.03, help="스위스독일어 Table 5 (Run 8~12)")
    g.add_argument("--label-smoothing", type=float, default=0.1, help="스위스독일어 Table 5 (Run 8~12)")
    g.add_argument("--beta1", type=float, default=0.9, help="AdamW β1 — §5.4.1")
    g.add_argument("--beta2", type=float, default=0.999, help="AdamW β2 — §5.4.1")
    g.add_argument("--eps", type=float, default=1e-8, help="AdamW ε — §5.4.1")
    g.add_argument("--grad-ckpt", action="store_true", help="§5.4.1 은 끔 — batch 8 이 안 켜고 들어간다(피크 36.74 GiB)")
    g.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="토이가 쓰던 값을 유지한다. **논문에 clip 값 기재 없음** — 우리 임시값이다")

    g = ap.add_argument_group("SpecAugment (§5.4.1 「켬」. 세부 값은 논문 미기재 — HF 기본값을 쓴다)")
    g.add_argument("--spec-augment", dest="spec_augment", action="store_true", default=True)
    g.add_argument("--no-spec-augment", dest="spec_augment", action="store_false")
    g.add_argument("--mask-time-prob", type=float, default=0.05, help="transformers WhisperConfig 기본값 (논문 미기재 — 미확인)")
    g.add_argument("--mask-time-length", type=int, default=10, help="같음")
    g.add_argument("--mask-feature-prob", type=float, default=0.0, help="같음 (0 = 주파수축 마스킹 안 함)")
    g.add_argument("--mask-feature-length", type=int, default=10, help="같음")

    g = ap.add_argument_group("LR 스케줄 (미결 — §5.4.1 이 근거를 「우리 판단」으로 적어 둔 자리)")
    g.add_argument("--lr-schedule", default="cosine", choices=["cosine", "linear", "constant", "constant_with_warmup"],
                   help="스위스독일어 §4.4 는 Run 구분 없이 `linear or cosine` 만 적는다 — cosine 은 우리 선택. **미결**")
    g.add_argument("--warmup-ratio", type=float, default=0.06,
                   help="총 업데이트의 6%% (weighted 108 · random 124). 논문의 250~500 step 은 우리 총 업데이트"
                        "(weighted 1,800 · random 2,070)에서 12~28%% 가 되어 LoRA 원문 RoBERTa 의 0.06 보다 크다. **미결 — 1차 후보값**")
    g.add_argument("--warmup-steps", type=int, default=-1, help="0 이상이면 --warmup-ratio 대신 이 값을 쓴다 (논문 값 250~500 을 그대로 시험할 때)")

    g = ap.add_argument_group("체크포인트·조기 종료 (TRACK2_BENCHMARK_V2_SPEC.md §4.3)")
    g.add_argument("--ckpt-every", type=int, default=2, help="§4.3 1번 — 2 epoch 마다. 마지막 epoch 은 항상 저장한다")
    g.add_argument("--val-hook", default="", help="체크포인트 직후 부를 명령 (docstring 「val 훅 계약」). 비우면 채점·조기 종료 없음")
    g.add_argument("--val-tau", type=float, default=None,
                   help="마진 τ. **§4.3 5번 조기 종료는 2026-09-09 삭제됐다 — 주지 않는다** (없으면 조기 종료를 안 한다). 코드는 남겨 둔다")
    g.add_argument("--early-stop-patience", type=int, default=2, help="옛 §4.3 5번 — --val-tau 가 없으면 무효 (2026-09-09 규칙 삭제)")
    g.add_argument("--val-hook-fatal", action="store_true", help="훅이 실패하면 학습을 멈춘다 (기본은 기록하고 계속)")

    g = ap.add_argument_group("step 모드 (토이 실측 §5 4번 · CPU 스모크)")
    g.add_argument("--max-steps", type=int, default=0, help="0 보다 크면 step 모드. epoch·체크포인트·스케줄·조기 종료가 전부 꺼진다")
    g.add_argument("--eval-n", type=int, default=16, help="학습 뒤 생성해 볼 **학습** 발화 수 (배선 확인용, 성능 아님)")
    return ap.parse_args()


def main():
    a = build_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed % 2**32)
    manifest = Path(a.manifest)
    base = manifest.parent
    rows = load_rows(manifest, a.limit, a.seed, a.max_sec)
    if len(rows) < a.batch:
        sys.exit(f"발화 {len(rows)}개 < 배치 {a.batch}")
    step_mode = a.max_steps > 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hook_argv = shlex.split(a.val_hook) if a.val_hook else []
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if a.val_tau is not None:
        sys.exit("--val-tau 는 받지 않는다 — §4.3 5번 조기 종료는 2026-09-09 삭제됐다 (코덱스 5차 18번: 레거시 분기를 명시적으로 막는다)")

    # 고유 폴더 강제 (코덱스 4차 8번) — epoch 모드에서 --out 이 비어 있지 않으면 --resume 없이는 시작하지 않는다
    out_dir = Path(a.out)
    state = None
    if not step_mode:
        if a.resume:
            state = load_state(out_dir, a, manifest_sha)
            print(f"재개 — epoch {state['epoch']} 상태 ({state['저장']}) 에서 {state['epoch'] + 1} 부터", flush=True)
        elif out_dir.exists() and any(out_dir.iterdir()):
            sys.exit(f"--out 이 비어 있지 않다: {out_dir} — 새 run 은 새 폴더에, 이어 하려면 --resume (코덱스 4차 8번)")

    micro_per_epoch = len(rows) // a.batch                        # 꼬리(배치 미달) 버림
    steps_per_epoch = math.ceil(micro_per_epoch / a.grad_accum)   # 마지막 업데이트는 부분 누적으로 친다
    total_steps = steps_per_epoch * a.epochs
    warmup = a.warmup_steps if a.warmup_steps >= 0 else round(a.warmup_ratio * total_steps)

    rep = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "스크립트": "src/finetune/train_lora_whisper.py",
           "모드": "step" if step_mode else "epoch",
           "모델": a.repo, "방식": "LoRA (사람 결정 2026-09-03) · 전 linear " + ",".join(TARGETS),
           "설정": vars(a) | {"warmup_해석": warmup, "총_업데이트_계획": total_steps,
                            "epoch당_업데이트": steps_per_epoch, "epoch당_마이크로배치": micro_per_epoch,
                            "effective_batch": a.batch * a.grad_accum,
                            "파생값_주의": "step 모드에서는 warmup_해석·총_업데이트_계획·epoch당_* 을 쓰지 않는다 (스케줄·epoch 루프가 꺼진다)"
                                       if step_mode else "epoch당_업데이트 = ceil(발화//batch / grad_accum) — 꼬리 마이크로배치는 버리고 마지막 업데이트는 부분 누적"},
           "설정_정본": "docs/FINETUNE_METHOD_SELECTION.md §5.4.1 + 그 아래 정정 블록",
           "미결": {"warmup": "총 step 의 6% 는 1차 후보다. 논문 값 250~500 step 은 우리 총 업데이트의 12~28% 라 그대로 못 쓴다",
                   "lr_schedule": "스위스독일어 §4.4 는 `linear or cosine` 만 적는다 — cosine 은 우리 선택",
                   "val_tau": "삭제 (2026-09-09 §4.3 개정 — 조기 종료 없음 · 이 스크립트는 --val-tau 를 거부한다)",
                   "SpecAugment_세부": "논문은 켬/끔만 적는다. mask_* 는 transformers 기본값 — 미확인",
                   "grad_clip": "논문에 clip 값 기재 없음. 1.0 은 토이가 쓰던 임시값"},
           "manifest": str(manifest).replace(str(Path.home()), "~"), "발화": len(rows),
           "manifest_sha256": manifest_sha, "run_kind": a.run_kind,
           "결정론": "seed 를 torch·random·numpy 에 건다 (SpecAugment 마스크가 numpy 전역 RNG) — cudnn deterministic 미설정 (코덱스 2차 13번)",
           "장치": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "gpu_mem_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if device == "cuda" else None,
           "torch": torch.__version__}
    report = Path(a.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        report.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    import transformers, peft
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_scheduler
    from transformers.models.whisper.modeling_whisper import shift_tokens_right
    from peft import LoraConfig, get_peft_model
    rep["transformers"], rep["peft"] = transformers.__version__, peft.__version__

    t0 = time.time()
    proc = WhisperProcessor.from_pretrained(a.repo)
    model = WhisperForConditionalGeneration.from_pretrained(a.repo, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False                     # teacher forcing 에 KV cache 불필요 — 메모리 실측이 cache 에 종속되지 않게 (코덱스 2차 9번)
    # SpecAugment — transformers 는 `WhisperConfig.apply_spec_augment` 로 켜고, 마스킹은
    # `WhisperModel._mask_input_features`(modeling_whisper.py:995)가 `self.training` 일 때만 건다.
    # 즉 model.eval() 로 바꾸면 자동으로 꺼진다. 별도 콜백이 필요 없다.
    model.config.apply_spec_augment = a.spec_augment
    model.config.mask_time_prob = a.mask_time_prob
    model.config.mask_time_length = a.mask_time_length
    model.config.mask_feature_prob = a.mask_feature_prob
    model.config.mask_feature_length = a.mask_feature_length
    pad_id, start_id = model.config.pad_token_id, model.config.decoder_start_token_id
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if state is not None:
        from peft import PeftModel
        # 저장된 어댑터를 학습 가능 상태로 얹는다. adapter_config 에 r·α·dropout·target_modules 가 들어 있다.
        model = PeftModel.from_pretrained(model, str(out_dir / f"epoch{state['epoch']:02d}"), is_trainable=True)
    else:
        cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", target_modules=TARGETS)
        model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    rep["파라미터"] = {"학습": n_train, "전체": n_all, "비율_pct": round(100 * n_train / n_all, 3)}
    rep["로드_초"] = round(time.time() - t0, 1)
    print(f"모델 로드 {rep['로드_초']}s · 학습 파라미터 {n_train/1e6:.1f}M / {n_all/1e6:.0f}M ({rep['파라미터']['비율_pct']}%)", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    # weight decay 는 학습 대상 전체에 건다 — bias="none" 이라 LoRA A·B 행렬뿐이고 LayerNorm·bias 는 얼어 있다.
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps, weight_decay=a.weight_decay)
    sched = None
    if not step_mode:
        # transformers.get_scheduler — warmup 뒤 cosine/linear decay. 총 step 은 **상한 15 epoch 기준**이다.
        # 조기 종료로 일찍 끝나도 스케줄은 계획대로 간다(끝까지 갈 것을 전제로 감쇠한다).
        sched = get_scheduler(a.lr_schedule, optimizer=opt, num_warmup_steps=warmup, num_training_steps=total_steps)
    if state is not None:
        opt.load_state_dict(state["opt"])
        if sched is not None and state["sched"] is not None:
            sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"]["torch"])
        if device == "cuda" and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        rep["재개"] = {"from_epoch": state["epoch"], "상태_저장_시각": state["저장"], "이전_벽시계_h": state["벽시계_h_누적"],
                     "상태_파일": str(out_dir / "train_state.pt")}
    data = Batches(rows, base, proc, a.batch, device)
    data.start_id = start_id
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def peak():
        return round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None

    # ---------------------------------------------------------------- step 모드 (토이 실측 · 스모크)
    if step_mode:
        it = iter(data)
        losses, step_sec, audio_sec, tok_n = [], [], 0.0, 0
        t_start = time.time()
        try:
            for step in range(1, a.max_steps + 1):
                ts = time.time()
                acc_loss = 0.0
                for _ in range(a.grad_accum):
                    feats, labels, sec, nt = next(it)
                    audio_sec += sec; tok_n += nt
                    loss = forward_loss(model, feats, labels, pad_id, start_id, a.label_smoothing, device, shift_tokens_right) / a.grad_accum
                    loss.backward()
                    acc_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - ts
                losses.append(round(acc_loss, 4)); step_sec.append(round(dt, 3))
                print(f"  step {step:3d}  loss {acc_loss:.4f}  {dt:.2f}s", flush=True)
        except torch.OutOfMemoryError as err:            # noqa: PERF203 — 토이의 목적이 이것을 재는 것
            rep["OOM"] = {"step": len(losses) + 1, "msg": str(err)[:200], "peak_GiB": peak()}
            flush()
            print(f"OOM at step {rep['OOM']['step']} — {a.report}", file=sys.stderr)
            sys.exit(3)
        wall = time.time() - t_start
        steady = step_sec[2:] if len(step_sec) > 4 else step_sec       # 앞 2 step 은 워밍업
        rep["학습"] = {"step": len(losses), "loss": losses, "step_초": step_sec,
                     "정상_step_초_중앙": round(float(np.median(steady)), 3),
                     "발화_per_s": round(len(losses) * a.batch * a.grad_accum / wall, 2),
                     "오디오_초_per_s": round(audio_sec / wall, 1),
                     "벽시계_초": round(wall, 1),
                     "피크_메모리_GiB": peak(),
                     "예약_메모리_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device == "cuda" else None}
        # 1 epoch 외삽 — Whisper 는 모든 입력을 30초로 패딩하므로 비용은 오디오 시간이 아니라 **배치 수**다 (코덱스 2차 8번).
        # 워밍업(앞 2 step) 뺀 정상 step 시간 × 실제 교재의 배치 수. 데이터 로딩 동기 포함.
        st_med = float(np.median(steady)) if steady else float("nan")
        ext = {"가정": "정상 step 시간(워밍업 2 step 제외) × ceil(교재 발화 수 / 배치) · 데이터 로딩 동기 포함 · 실측 아님",
               "정상_step_초": round(st_med, 3), "배치": a.batch * a.grad_accum}
        sets_json = ROOT / "registry/benchmark/track2_bench_v2_train_sets.json"
        if sets_json.exists():
            for name, v in json.loads(sets_json.read_text(encoding="utf-8"))["세트"].items():
                steps = math.ceil(v["발화"] / (a.batch * a.grad_accum))
                ext[name] = {"발화": v["발화"], "시간_h": v["시간_h"], "step_per_epoch": steps, "1epoch_h": round(steps * st_med / 3600, 2)}
        rep["외삽"] = ext
        print(f"학습 {len(losses)} step · 정상 step {rep['학습']['정상_step_초_중앙']}s · {rep['학습']['오디오_초_per_s']} 오디오초/s · "
              f"피크 {rep['학습']['피크_메모리_GiB']} GiB · 1 epoch: " +
              " · ".join(f"{k} {v['1epoch_h']}h" for k, v in ext.items() if isinstance(v, dict) and "1epoch_h" in v), flush=True)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        (out / "run_kind.json").write_text(json.dumps({"run_kind": a.run_kind, "step": len(losses), "manifest": rep["manifest"],
                                                       "manifest_sha256": rep["manifest_sha256"], "설정": rep["설정"]},
                                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        rep["어댑터"] = str(out)

    # ---------------------------------------------------------------- epoch 모드 (본 학습)
    else:
        ck_meta = {"manifest": rep["manifest"], "manifest_sha256": rep["manifest_sha256"], "설정": rep["설정"]}
        rep["에폭"] = list(state["에폭"]) if state is not None else []
        rep["체크포인트"] = list(state["체크포인트"]) if state is not None else []
        # `기준` = patience 를 세는 잣대(τ 이상 좋아졌을 때만 갱신). 실제 최저값은 마지막에 에폭 기록에서 다시 뽑는다.
        기준 = {"epoch": None, "고령_micro_cer": None}
        patience, done_epochs, stop_reason = 0, (state["epoch"] if state is not None else 0), f"상한 {a.epochs} epoch 도달"
        start_ep = done_epochs + 1
        print(f"epoch 모드 — {len(rows)} 발화 · 마이크로 {micro_per_epoch}/epoch · "
              f"업데이트 {steps_per_epoch}/epoch · 계획 {total_steps} · warmup {warmup} ({a.lr_schedule})"
              + (f" · 재개 {start_ep} epoch 부터" if state is not None else ""), flush=True)
        t_all = time.time() - (state["벽시계_h_누적"] * 3600 if state is not None else 0)
        try:
            for ep in range(start_ep, a.epochs + 1):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                model.train()
                model.config.use_cache = False
                t_ep = time.time()
                ep_losses, audio_sec, tok_n, upd = [], 0.0, 0, 0
                acc_loss, acc_n = 0.0, 0
                lr_first = opt.param_groups[0]["lr"]
                for mi, (feats, labels, sec, nt) in enumerate(data.epoch(a.seed, ep), start=1):
                    audio_sec += sec; tok_n += nt
                    # 마지막 업데이트가 부분 누적이면 그만큼으로 나눈다 (남은 마이크로 수).
                    denom = min(a.grad_accum, micro_per_epoch - (mi - 1 - (mi - 1) % a.grad_accum))
                    loss = forward_loss(model, feats, labels, pad_id, start_id, a.label_smoothing, device, shift_tokens_right) / denom
                    loss.backward()
                    acc_loss += loss.item(); acc_n += 1
                    if acc_n == denom:
                        torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
                        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                        upd += 1
                        ep_losses.append(round(acc_loss, 4))
                        acc_loss, acc_n = 0.0, 0
                        if upd % 20 == 0 or upd == steps_per_epoch:
                            print(f"  ep {ep:2d}  upd {upd:4d}/{steps_per_epoch}  loss {ep_losses[-1]:.4f}  "
                                  f"lr {opt.param_groups[0]['lr']:.3e}  {(time.time()-t_ep)/60:.1f}m", flush=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                wall = time.time() - t_ep
                done_epochs = ep
                rec = {"epoch": ep, "업데이트": upd, "마이크로배치": micro_per_epoch,
                       "loss_평균": round(float(np.mean(ep_losses)), 4) if ep_losses else None,
                       "loss_처음": ep_losses[0] if ep_losses else None, "loss_마지막": ep_losses[-1] if ep_losses else None,
                       "loss_업데이트별": ep_losses,
                       "lr_시작": lr_first, "lr_끝": opt.param_groups[0]["lr"],
                       "벽시계_초": round(wall, 1), "발화": micro_per_epoch * a.batch,
                       "발화_per_s": round(micro_per_epoch * a.batch / wall, 2),
                       "오디오_초_per_s": round(audio_sec / wall, 1), "라벨_토큰": tok_n,
                       "피크_메모리_GiB": peak(),
                       "예약_메모리_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device == "cuda" else None,
                       "체크포인트": None, "val": None}
                # §4.3 1번 — 2 epoch 마다 + 마지막 epoch 은 항상
                if ep % a.ckpt_every == 0 or ep == a.epochs:
                    ck = save_ckpt(model, Path(a.out), ep, a.run_kind, ck_meta,
                                   f"{a.ckpt_every}epoch 주기" if ep % a.ckpt_every == 0 else "마지막(상한)")
                    rec["체크포인트"] = ck
                    rep["체크포인트"].append(ck)
                    print(f"  ep {ep:2d}  체크포인트 {ck['경로']} ({ck['바이트']/2**20:.0f} MiB)", flush=True)
                    if hook_argv:
                        v = call_val_hook(hook_argv, Path(ck["경로"]), ep)
                        rec["val"] = v
                        if "실패" in v:
                            print(f"  ep {ep:2d}  val 훅 실패: {v['실패']}", file=sys.stderr, flush=True)
                            if a.val_hook_fatal:
                                stop_reason = f"val 훅 실패 (--val-hook-fatal): {v['실패']}"
                                rep["에폭"].append(rec); flush()
                                break
                        else:
                            cur = float(v["고령_micro_cer"])
                            print(f"  ep {ep:2d}  val 고령 micro CER {cur}", flush=True)
                            # §4.3 5번 — 「τ 이상 좋아지지 않으면」이 patience 의 잣대다. τ 가 없으면 세지 않는다.
                            if 기준["고령_micro_cer"] is None or (a.val_tau is not None and 기준["고령_micro_cer"] - cur >= a.val_tau):
                                기준 = {"epoch": ep, "고령_micro_cer": cur}
                                patience = 0
                            elif a.val_tau is not None:
                                patience += 1
                            rec["patience"] = patience
                rep["에폭"].append(rec)
                flush()
                if rec["체크포인트"] is not None:                 # 어댑터를 저장한 epoch 에만 재시작 상태를 덮어쓴다
                    rec["재시작_상태"] = save_state(Path(a.out), ep, opt, sched, rep, a, (time.time() - t_all) / 3600)
                    flush()
                    if a.stop_after_epoch and ep == a.stop_after_epoch:
                        print(f"--stop-after-epoch {ep}: 상태 저장 뒤 죽은 척 (exit 4)", flush=True)
                        sys.exit(4)
                left = a.epochs - ep
                print(f"epoch {ep}/{a.epochs} · loss {rec['loss_평균']} · {wall/3600:.2f}h · 피크 {rec['피크_메모리_GiB']} GiB · "
                      f"남은 {left} epoch 예상 {left*wall/3600:.1f}h", flush=True)
                if a.val_tau is not None and hook_argv and patience >= a.early_stop_patience:
                    stop_reason = (f"조기 종료 — val 채점 {patience} 회 연속 개선 없음 (τ {a.val_tau}, §4.3 5번). "
                                   f"기준 epoch {기준['epoch']} CER {기준['고령_micro_cer']}")
                    print(stop_reason, flush=True)
                    break
        except torch.OutOfMemoryError as err:
            rep["OOM"] = {"epoch": done_epochs + 1, "msg": str(err)[:200], "peak_GiB": peak()}
            flush()
            print(f"OOM at epoch {rep['OOM']['epoch']} — {a.report}", file=sys.stderr)
            sys.exit(3)
        # 마지막으로 실제로 돈 epoch 이 저장 대상이 아니었으면 여기서 저장한다 (§4.3 1번 「마지막 epoch 은 항상」)
        if done_epochs and rep["에폭"] and rep["에폭"][-1]["체크포인트"] is None:
            ck = save_ckpt(model, Path(a.out), done_epochs, a.run_kind, ck_meta, "마지막(주기 밖)")
            rep["에폭"][-1]["체크포인트"] = ck
            rep["체크포인트"].append(ck)
        scored = [(r["epoch"], r["val"]["고령_micro_cer"]) for r in rep["에폭"]
                  if isinstance(r.get("val"), dict) and "고령_micro_cer" in r["val"]]
        rep["학습"] = {"epoch_돈": done_epochs, "총_업데이트": sum(r["업데이트"] for r in rep["에폭"]),
                     "벽시계_h": round((time.time() - t_all) / 3600, 2),
                     "피크_메모리_GiB": peak(), "종료_이유": stop_reason,
                     "val_채점된_epoch": len(scored),
                     "val_최저": (lambda e, c: {"epoch": e, "고령_micro_cer": c})(*min(scored, key=lambda x: x[1])) if scored else None,
                     "val_patience_기준": 기준 if 기준["epoch"] else None,
                     "체크포인트_고르기": "이 스크립트는 고르지 않는다 — val_최저는 참고값이다. "
                                    "TRACK2_BENCHMARK_V2_SPEC.md §4.3 의 3단 규칙(탈락 A · 경고 B · 고령 micro 최저)으로 사람이 고른다"}
        rep["어댑터"] = [c["경로"] for c in rep["체크포인트"]]
        print(f"끝 — {done_epochs} epoch · {rep['학습']['벽시계_h']}h · 체크포인트 {len(rep['체크포인트'])}개 · {stop_reason}", flush=True)

    # 배선 확인: 학습에 쓴 발화 몇 개를 생성해 본다 (성능 측정 아님 — 학습 데이터다)
    if a.eval_n:
        model.eval()                                   # SpecAugment 도 여기서 자동으로 꺼진다
        model.config.use_cache = True
        forced = proc.get_decoder_prompt_ids(language="korean", task="transcribe")
        ev = []
        with torch.no_grad():
            for r in rows[:a.eval_n]:
                x = load_wav(base / r["clip_file"])
                feats = proc(x, sampling_rate=SR, return_tensors="pt").input_features.to(device, torch.bfloat16 if device == "cuda" else torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    ids = model.generate(input_features=feats, forced_decoder_ids=forced, max_new_tokens=200)
                hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                ev.append({"utterance_id": r["utterance_id"], "ref": r["label_text"], "hyp": hyp})
        agg = aggregate([score_one(e["ref"], e["hyp"], status="PASS", kspon=False) for e in ev])
        rep["배선_확인"] = {"n": len(ev), "cer_학습발화(성능아님)": agg.get("cer_micro"), "예시": ev[:5]}
        print(f"생성 확인 {len(ev)}발화 · CER(학습 발화, 성능 아님) {agg.get('cer_micro')}", flush=True)

    flush()
    print(f"→ {a.report}")


if __name__ == "__main__":
    main()
