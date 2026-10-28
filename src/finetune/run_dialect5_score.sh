#!/usr/bin/env bash
# 방언 5지역 v2 — val CER 채점 EC2 러너: 어댑터 하나로 val 전체를 받아쓰고 CER 을 S3 에 올린 뒤 끈다.
#
# 쓰기 (EC2 안 · 보통 src/infra/launch_track2_gpu.sh start-dialect5-score 가 tmux 로 부른다)
#   src/finetune/run_dialect5_score.sh gyeongsang GS-07 step001176
#   src/finetune/run_dialect5_score.sh gyeongsang GS-07 none                      # 무튜닝 기준선 (어댑터 없이 원본 large-v3) — 결과는 val_cer_none.json
#   src/finetune/run_dialect5_score.sh seoul baseline none                        # 서울(국립국어원) val — 데이터 prefix 가 방언과 다르다 (2026-09-29)
#   ADAPTER_S3=s3://…/results_lora_main/r2_lr5e-5/adapters/epoch14 RUN_ID=gyeongsang_old50h src/finetune/run_dialect5_score.sh gyeongsang old50h epoch14   # 다른 prefix 의 어댑터
#   NO_SHUTDOWN=1 …    # 끝나도 끄지 않음
#   SPLIT=test …       # val 대신 test 를 채점 — 결과는 test_cer_<step>.json (2026-10-01, 기본 val)
#   PROMPT_FILE=<힌트 글 파일> …   # Whisper 힌트(앞 문장 칸)를 넣고 채점 — 방언 철자 힌트 실험 (2026-10-07). RUN_ID 를 따로 줘서 결과가 섞이지 않게 한다
# 산출물  registry/finetune/dialect5_lora/<RUN_ID>/val_cer_<step>.json · .csv → S3 results_dialect5/<region>/<RUN_ID>/report/
# 어댑터가 디스크에 없으면 S3 adapters/<step>/ 에서 받는다. 데이터셋(val)도 없으면 완성본 prefix 에서 받는다.
# 실패해도 끈다(결과가 작아 다시 돌리면 된다) — 단 summary.txt 에 exit 코드를 남기고 올린다.
set -euo pipefail
REGION="${1:?region}"; COMBO="${2:?combo GS-NN}"; STEP="${3:?어댑터 폴더 이름 (step001176 · epoch04)}"
case "$REGION" in gyeongsang|gangwon|chungcheong|jeolla|jeju|pooled|seoul) ;; *) echo "모르는 region: $REGION" >&2; exit 2 ;; esac
cd "$HOME/accentury"
export ACCENTURY_DATA="${ACCENTURY_DATA:-$HOME/AccenturyData}"
BUCKET=s3://accentury-track2-325771561913
MANIFEST_SHA=01bab6ab0e761418788678fc7227ebe40110a56a35854469f3f9d4d13aded9af   # 5지역이 같은 v2 manifest(같은 해시) (2026-09-28)
[ "$REGION" = seoul ] && MANIFEST_SHA=841921cacae8017d0303d214c64aa7c9c462556015cb930b039c29250d28ef3c   # 서울은 국립국어원 순수 서울 목록 (docs/SEOUL_NIKL_DIALECT5_SET_HANDOFF_2026-09-29.md)
DATASET_S3="${DATASET_S3:-$BUCKET/datasets/dialect5_bench_v2/$REGION/$MANIFEST_SHA}"
DATASET_ROOT="$ACCENTURY_DATA/work/finetune/dialect5_bench_v2/$REGION"
RUN_ID="${RUN_ID:-${REGION}_${COMBO}}"
ADAPTER="$ACCENTURY_DATA/work/finetune/dialect5_lora/$RUN_ID/$STEP"
OUT="registry/finetune/dialect5_lora/$RUN_ID"
REMOTE="$BUCKET/results_dialect5/$REGION/$RUN_ID"
SPLIT="${SPLIT:-val}"
case "$SPLIT" in val|test) ;; *) echo "SPLIT 은 val|test: $SPLIT" >&2; exit 2 ;; esac
VAL_MAN="$DATASET_ROOT/sets/${REGION}_${SPLIT}/manifest.csv"
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
cleanup() { rc=$?; set +e; log "=== score cleanup (exit $rc · $STEP) ==="; aws s3 sync "$OUT/" "$REMOTE/report/" --exclude "DONE.json" --exclude "*.tmp" >/dev/null 2>&1 && log "채점 결과 업로드 OK" || log "UPLOAD_FAILED 채점 결과"; [ "${NO_SHUTDOWN:-0}" = "1" ] && exit "$rc"; sudo shutdown -h now; exit "$rc"; }
trap cleanup EXIT
log "$SPLIT CER 채점 시작 RUN_ID=$RUN_ID STEP=$STEP"
.venv/bin/pip install -q "peft==0.20.0" 2>&1 | tail -1
# 디스크에 옛 판(같은 지역 다른 목록)이 남아 있어도 이번 prefix 로 맞추도록 매번 sync 한다 — 같은 판이면 받을 것이 없다 (2026-09-29)
aws s3 ls "$DATASET_S3/_SUCCESS.json" >/dev/null || { log "완성본 마커가 없다: $DATASET_S3/_SUCCESS.json"; exit 2; }
log "데이터셋 sync: $DATASET_S3"; timeout 3h aws s3 sync "$DATASET_S3/" "$DATASET_ROOT/" >/dev/null
ADAPTER_ARGS=()
if [ -n "${PROMPT_FILE:-}" ]; then
  [ -s "$PROMPT_FILE" ] || { log "힌트 파일이 없다: $PROMPT_FILE"; exit 2; }
  ADAPTER_ARGS+=(--prompt-file "$PROMPT_FILE"); log "힌트: $PROMPT_FILE ($(sha256sum "$PROMPT_FILE" | cut -c1-12))"
fi
if [ "$STEP" != none ]; then
  [ -f "$ADAPTER/adapter_config.json" ] || { log "어댑터 로컬에 없음 — S3 에서 받기 (${ADAPTER_S3:-$REMOTE/adapters/$STEP})"; aws s3 sync "${ADAPTER_S3:-$REMOTE/adapters/$STEP}/" "$ADAPTER/" >/dev/null; }
  [ -f "$ADAPTER/adapter_config.json" ] || { log "어댑터가 없다: $ADAPTER"; exit 2; }
  ADAPTER_ARGS+=(--adapter "$ADAPTER")
else
  log "무튜닝 기준선 — 어댑터 없이 원본 large-v3"
fi
.venv/bin/python src/finetune/score_dialect5_val.py --manifest "$VAL_MAN" ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} --out "$OUT/${SPLIT}_cer_$STEP.json" --label "$RUN_ID/$STEP/$SPLIT" 2>&1 | tee -a "$OUT/${SPLIT}_cer_$STEP.log"
rc=${PIPESTATUS[0]}
log "scorer exit $rc"
exit "$rc"
