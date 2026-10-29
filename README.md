# Accentury_Model

Accentury의 Whisper large-v3 LoRA **학습 코드와 실제 실행 설정**입니다. 멘토가 학습 구조·옵션·모델 선택 과정을 확인할 수 있도록 연구 저장소에서 필요한 파일을 모았습니다. 서비스 이미지 안의 추론 코드는 [Accentury_Inference](https://github.com/AISWM-2PL1/Accentury_Inference)에서 별도로 관리합니다.

## 먼저 볼 곳

| 대상 | 학습 코드 | 실제 설정·최종 모델 |
| --- | --- | --- |
| 논문 저자원 A/B 실험 | [현재 학습기](src/finetune/train_lora_whisper.py) | [10월 9일 최종 30개 실험 설정](registry/finetune/dialect5_low_resource_runs_2026-10-09.json), [GS-09](src/finetune/gs_grid.json). 각 run의 val loss 최저 step을 선택 |
| 논문 B 12.8h 기준인 기존 통합 모델 | 같은 학습 계열, 스모크 후 재개 | [스모크 설정](registry/finetune/pooled_GS-09_smoke_settings.json), [본학습 설정](registry/finetune/pooled_GS-09_full_settings.json). `pooled_GS-09/step021244` |
| 경상도 약 50시간 최종 학습 | [9월 Git 버전 후보](historical/gyeongsang_50h_candidate_89064c2/src/finetune/train_lora_whisper.py) | [실행 설정](registry/finetune/gyeongsang_50h_final_settings.json), [선택 결과](registry/benchmark/track2_bench_v2_val_select__r2_lr5e-5.json). `r2_lr5e-5/epoch14` |

논문 저자원 학습기의 소스는 **10월 9일 보존 배포본**을 기준으로 합니다. 경상도 50시간은 실행 설정·선택 결과가 확인됐지만, 보존한 Git 후보와 당시 서버 소스의 바이트 동일성은 확인되지 않았습니다. 이 구분은 [출처 기록](SOURCE.md)에 남겼습니다.

## 읽는 순서

1. [학습기](src/finetune/train_lora_whisper.py): 데이터 로딩 → Whisper/LoRA 구성 → loss·gradient accumulation → checkpoint·재개 → val loss 선택.
2. [실행 안내](docs/TRAINING.md): 데이터 형식, 환경 준비, 두 학습의 명령, 평가 방법.
3. [실험 설정](registry/finetune/dialect5_low_resource_runs_2026-10-09.json): 각 run의 train/val/test, 실제 하이퍼파라미터, 저장 위치.
4. [평가 코드](src/finetune/score_dialect5_val.py), [50시간 모델 선택 코드](src/stt/score_val_track2.py): 학습 뒤 성능 확인과 모델 선택.

## 구성

```text
src/finetune/       학습기, 실제 실행 스크립트, 평가·집계
src/stt/            공통 지표, 평가용 Whisper 로딩, 50시간 모델 선택 의존 코드
registry/finetune/  최종 실행 설정과 실행 보고에서 발췌한 설정
registry/benchmark/ 50시간 모델 선택에 필요한 집계 정보와 선택 결과
historical/        50시간 학습 시점의 Git 후보(원래 내부 경로 유지)
config/            학습용 Python 의존성
docs/              실행 방법과 확인 범위
SOURCE_MANIFEST.json  파일별 출처·SHA-256·발췌 여부
```

import와 스크립트 호출이 사용하는 `src/`·`registry/` 상대경로를 유지했습니다. `src/stt/track2_asr.py`는 **학습 결과 평가용**이며 서비스 API 서버가 아닙니다.

## 실행 전 확인

- [학습 의존성](config/requirements-train.txt)을 설치하고, 실제 기록의 CUDA PyTorch 환경은 [환경 설명](docs/TRAINING.md#환경)을 확인합니다.
- 데이터는 권한이 있는 외부 경로에서 준비합니다. 이 저장소에는 음성·전사 원자료, 가중치, optimizer 상태를 포함하지 않습니다.
- 원래 EC2 실행 스크립트는 `$HOME/accentury`, 팀 S3 접근 권한, AWS CLI, `flock`을 전제하며 종료 시 서버를 정지시킬 수 있습니다. 일반 환경에서는 문서의 Python 직접 실행 예시를 사용합니다.
- [검증 기록](docs/VERIFICATION.md)은 코드 공유 준비의 확인 범위를 적습니다. 실제 GPU 재학습 결과와 구분합니다.
