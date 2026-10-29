# 학습과 평가 실행

저장소 루트에서 명령을 실행합니다. 기존 상대경로와 계산을 보존했습니다. 아래 학습·논문 채점·최종 선택 예시는 출력을 `outputs/`에 두며, 기존 `val_hook.py`가 남기는 채점 요약은 `registry/benchmark/`에 저장됩니다. 파일·설정 공유를 위한 안내이며 새 학습 실행 결과를 뜻하지 않습니다.

## 환경

경상도 50시간 및 기존 통합 모델 보고에 기록된 버전은 `torch 2.9.1+cu130`, `transformers 4.57.6`, `peft 0.20.0`입니다. 먼저 실행할 GPU에 맞는 CUDA PyTorch 환경을 준비한 뒤 나머지 의존성을 설치합니다. 기존 Deep Learning AMI의 CUDA PyTorch를 맥용 requirements로 덮어쓰지 않습니다.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r config/requirements-train.txt
.venv/bin/python -c 'import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)'
```

`--system-site-packages`는 이미 준비된 PyTorch를 사용하는 경우입니다. 출력에서 CUDA 사용 가능 여부와 버전을 확인합니다. `numpy`, `soundfile`, `tensorboard`, Python 자체의 당시 EC2 버전은 이번 자료만으로 고정하지 않았으므로 완전한 환경 잠금 파일은 아닙니다. macOS의 CPU 명령 확인과 실제 GPU 학습은 구분합니다.

## 데이터

학습기 입력은 `manifest.csv`와 그 위치를 기준으로 찾는 WAV입니다.

```text
dataset/
  manifest.csv
  clips/example.wav
```

```csv
utterance_id,clip_file,duration_label_sec,label_text
example,clips/example.wav,1.5,예시 문장
```

WAV는 16 kHz mono입니다. 빈 정답과 `duration_label_sec >= 30`은 제외하며, 학습은 마지막 불완전 배치를 버립니다. 검증 loss는 유효 라벨 토큰 기준으로 집계합니다. 예시 CSV는 형식 설명용이며 실제 학습 데이터가 아닙니다.

논문 run별 데이터 위치·manifest SHA-256은 `registry/finetune/dialect5_low_resource_runs_2026-10-09.json`에 있습니다. 음성·전사 데이터와 pretrained/LoRA 가중치는 저장소 밖에 준비합니다. 경상도 50시간 평가는 `track2_data.py`와 선택기가 읽는 정답·화자 묶음·대화·세트 정보 등 원래 평가 manifest의 필드도 유지해야 합니다. 위 4열 형식은 학습기 입력 형식입니다.

## 논문 저자원 A/B 학습

최종 30개 run은 GS-09를 사용합니다. 15 epoch 상한, 배치 8, 누적 1, lr `1e-4`, linear schedule, warmup 비율 `0.0833`입니다. epoch의 약 20%마다 및 epoch 끝에 평가하고, 5회 연속 미개선이면 멈추되 global step 500 전에는 종료를 보류합니다. 최종 선택은 `best.json`의 val loss 최저 step입니다.

일반 환경의 직접 실행 예시입니다. `TRAIN_MANIFEST`, `VAL_MANIFEST`는 해당 run의 실제 목록으로 지정합니다.

```bash
export TRAIN_MANIFEST=/absolute/path/to/train/manifest.csv
export VAL_MANIFEST=/absolute/path/to/val/manifest.csv
.venv/bin/python src/finetune/train_lora_whisper.py \
  --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --out outputs/paper_run --report outputs/paper_run.json \
  --run-kind main --limit 0 --eval-n 0 \
  --epochs 15 --batch 8 --grad-accum 1 --lr 1e-4 \
  --lr-schedule linear --warmup-ratio 0.0833 --beta1 0.9 --beta2 0.99 \
  --r 32 --alpha 8 --dropout 0.05 --weight-decay 0.03 --label-smoothing 0.1 \
  --eval-every-frac 0.2 --val-loss-patience 5 --val-loss-min-steps 500 \
  --keep-last 3 --tensorboard outputs/tb/paper_run
```

완료된 모델 평가는 `best.json`의 `경로`를 `--adapter`에 넣고, 해당 지역의 **test** manifest를 지정합니다.

```bash
.venv/bin/python src/finetune/score_dialect5_val.py \
  --manifest /absolute/path/to/test/manifest.csv \
  --adapter /absolute/path/to/stepNNNNNN \
  --out outputs/paper_test.json
```

파일명은 `val`이지만 입력으로 준 세트를 채점합니다. 최종 보고 전에는 JSON의 `실패`와 `채점.n_scored`, CSV의 상태를 확인합니다. 기존 통합 B 12.8h와 충청 제외 규칙을 포함한 논문 표 집계 코드는 `aggregate_dialect5_lowres.py`, `aggregate_dialect5_wer.py`, `dialect5_exclusions.py`에 있습니다.

### 원래 EC2 runner

코드가 `$HOME/accentury`에 있고 데이터는 `$ACCENTURY_DATA`에 있는 팀 환경을 전제합니다. AWS CLI, 팀 S3 권한, `flock` 등도 필요합니다. runner는 S3 다운로드·업로드와 서버 정지 기능을 포함합니다. 아래는 한 run만 지정하며, `NO_SHUTDOWN=1`로 자동 정지를 끕니다.

```bash
LOWRES_CONFIG=registry/finetune/dialect5_low_resource_runs_2026-10-09.json \
  QUEUE=0 NO_SHUTDOWN=1 \
  bash src/finetune/run_dialect5_lowres.sh gyeongsang_A3p2h_GS-09
```

`LOWRES_CONFIG`를 생략하면 10월 4일 설정을 읽습니다. `run_dialect5_lowres.sh` → `run_dialect5_smoke.sh` → 학습기 → `run_dialect5_score.sh` → 채점기 순으로 실행됩니다. 이후 설정·S3 파일 대조를 거쳐 `JOB_DONE.json`을 남깁니다. 직접 Python 명령에는 이 클라우드 작업이 없습니다.

## 기존 통합 모델 B 12.8h

`pooled_GS-09/step021244`는 30개 저자원 run을 새로 학습해 얻은 모델이 아니라, 이전 통합 학습에서 재사용한 모델입니다. 통합 train과 통합 val을 사용했고, 같은 15 epoch 스케줄로 먼저 2 epoch를 돌린 뒤 상태를 이어받았습니다. 초기 단계는 patience 0, 재개 단계는 patience 5이며, 이 실험에는 500 step 종료 하한 옵션이 없었습니다. 설정은 두 `pooled_GS-09_*_settings.json`에서 확인합니다.

당시 EC2 흐름을 호출하는 명령은 다음과 같습니다. 현재 보존 학습기에서는 `--val-loss-min-steps 0`이 기본입니다. **당시 통합 학습 소스의 바이트 동일성을 별도로 검증한 재실행은 아닙니다.**

```bash
NO_SHUTDOWN=1 PHASE=smoke bash src/finetune/run_dialect5_smoke.sh pooled GS-09
NO_SHUTDOWN=1 PHASE=full FULL_PATIENCE=5 bash src/finetune/run_dialect5_smoke.sh pooled GS-09
```

두 번째 명령은 첫 단계가 완료되고 상태가 저장된 뒤 실행합니다. 어댑터·optimizer·scheduler·RNG 상태를 복원하며 스케줄을 새로 시작하지 않습니다.

## 경상도 약 50시간

최종 run `r2_lr5e-5`는 `v2_train_weighted`로 14 epoch를 학습했습니다. 배치 8 × gradient accumulation 32, lr `5e-5`, cosine schedule, warmup 0.06입니다. 이 실행은 학습 중 외부 val hook을 쓰지 않았고, 별도 평가 후 epoch14가 선택됐습니다.

학습 시점 Git **후보**의 직접 실행 예시입니다. 데이터 경로를 실제 위치로 바꿉니다. 학습기의 나머지 기본값은 `gyeongsang_50h_final_settings.json`과 대조합니다.

```bash
.venv/bin/python historical/gyeongsang_50h_candidate_89064c2/src/finetune/train_lora_whisper.py \
  --manifest /absolute/path/to/v2_train_weighted/manifest.csv \
  --out outputs/gyeongsang_50h --report outputs/gyeongsang_50h.json \
  --run-kind main --limit 0 --epochs 14 --lr 5e-5
```

`--eval-n 16`은 당시 기본값이며 학습 데이터 생성 확인용입니다. 이 값으로 논문 성능을 보고하지 않습니다. 최종 모델 선택은 8·10·12·14 epoch와 무튜닝 결과를 별도로 비교했습니다.

평가 데이터는 `$ACCENTURY_DATA/work/finetune/track2_bench_v2_val/sets`에 준비합니다. 예를 들어 epoch14의 추론·채점 연결은 다음과 같습니다.

```bash
export ACCENTURY_DATA=/absolute/path/to/AccenturyData
.venv/bin/python src/finetune/val_hook.py outputs/gyeongsang_50h/epoch14 14 \
  --run-label r2_lr5e-5-review --shard-root outputs/val_shards
```

위 hook은 `registry/benchmark/track2_bench_v2_val_stt__r2_lr5e-5-review_ep14.json`에 채점 요약을 남깁니다. 이어서 최종 선택을 다시 계산하려면 무튜닝과 8·10·12·14 epoch 각각의 추론 조각이 필요합니다. `--model`은 어댑터 경로가 아니라 `A__v2val_*.csv.gz`와 `_run_meta.json`이 있는 폴더입니다. 아래 `--out-dir`은 보존된 최종 선택 기록을 덮어쓰지 않도록 지정합니다.

```bash
.venv/bin/python src/stt/score_val_track2.py \
  --select --run-label r2_lr5e-5 --baseline baseline \
  --out-dir outputs/gyeongsang_50h_selection \
  --model baseline=/absolute/path/to/baseline_shards \
  --model r2_lr5e-5_ep08=/absolute/path/to/ep08_shards \
  --model r2_lr5e-5_ep10=/absolute/path/to/ep10_shards \
  --model r2_lr5e-5_ep12=/absolute/path/to/ep12_shards \
  --model r2_lr5e-5_ep14=/absolute/path/to/ep14_shards
```

이 평가 코드 묶음은 공유 시점의 보존본입니다. 역사적 소스 동일성 범위는 [SOURCE.md](../SOURCE.md)를 따릅니다. `run_lora_train.sh`와 `run_scorer.sh`는 당시 분리된 학습/평가 서버의 실행 방식을 보여 주는 운영 보조 코드로 함께 보존했습니다.
