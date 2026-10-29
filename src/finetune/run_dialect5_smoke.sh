#!/usr/bin/env bash
# 방언 5지역 v2 — Whisper-large-v3 LoRA 스모크·본학습 EC2 러너 (`docs/GYEONGSANG_LORA_SEARCH_PLAN_2026-09-27.md` §5)
#
# 쓰기 (EC2 안에서 · 보통은 src/infra/launch_track2_gpu.sh start-dialect5 가 tmux 로 부른다)
#   src/finetune/run_dialect5_smoke.sh gyeongsang GS-01                 # 스모크: 15 epoch 스케줄로 2 epoch 만 돌고 상태 저장 → S3 → 끔
#   PHASE=full src/finetune/run_dialect5_smoke.sh gyeongsang GS-01      # 본학습: S3 의 스모크 상태를 받아 --resume, 조기 종료/상한 15 까지 → S3 → 끔
#   PHASE=direct FULL_PATIENCE=5 src/finetune/run_dialect5_smoke.sh gyeongsang GS-13   # 확인용: 스모크 없이 처음부터 조기 종료까지 (GS-13~20 · 2026-09-28)
#   NO_SHUTDOWN=1 …                                                     # 사람이 붙어 볼 때 (텐서보드: ssh -L 6006:localhost:6006 뒤 tensorboard --logdir $TB_ROOT)
#   SMOKE_PATIENCE=0 (기본) · FULL_PATIENCE=3 (기본, 키페이퍼) — 조기 종료 patience. 스모크는 끄고 본학습에서만 켠다 (2026-09-28)
#
# 인자
#   <region>  gyeongsang | gangwon | chungcheong | jeolla | jeju | pooled  — 완성본은 datasets/dialect5_bench_v2/<region>/<manifest sha>/ (2026-09-28 기준 gyeongsang·jeolla)
#             seoul — 받아쓰기 방식 대조 학습 (2026-10-07). 기본 prefix 가 없어 DATASET_S3 · VAL_DATASET_S3 · TRAIN_SET 을 반드시 준다
#   <combo>   src/finetune/gs_grid.json 의 키 (GS-01 … GS-12)
#   세 번째 인자부터는 train_lora_whisper.py 로 그대로 넘어간다 (예: --limit 64 로 배선 확인)
#
# 환경변수
#   DATASET_S3   완성본 prefix (기본: <region> 의 handoff 문서 해시 prefix). `_SUCCESS.json` 이 없으면 시작하지 않는다
#   DATASET_ROOT 기본 지역 cache 대신 사용할 로컬 데이터 루트
#   TRAIN_SET    기본 <region>_train. 중첩 세트는 <region>_train_1p6h / 3p2h / 6p4h
#   VAL_DATASET_S3 · VAL_DATASET_ROOT   val manifest 를 train 과 다른 완성본에서 읽을 때 (2026-10-04 저자원 A·B: train 은 중첩·B 묶음, val 은 목표 지역 원본 완성본).
#                비우면 train 과 같은 루트의 sets/<region>_val 을 쓴다. 둘은 같이 주고, 루트는 DATASET_ROOT 와 달라야 한다
#   RUN_ID       기본 <region>_<combo>. 어댑터·보고·텐서보드·S3 prefix 가 전부 이 아래다
#   UPLOAD_EVERY_SEC  주기 업로드(텐서보드·보고 JSON) 간격 (기본 600)
#
# 산출물
#   로컬  $ACCENTURY_DATA/work/finetune/dialect5_lora/$RUN_ID/{step*/, epochNN/, train_state.pt, best.json}
#         $ACCENTURY_DATA/work/finetune/dialect5_tb/$RUN_ID/            텐서보드 이벤트
#         registry/finetune/dialect5_lora/$RUN_ID/{lora_*.json, *.log, summary.txt}
#   S3    s3://accentury-track2-325771561913/results_dialect5/<region>/$RUN_ID/{adapters/, tb/, report/}  — report/DONE.json 이 마지막
#
# 규율 (run_lora_train.sh 와 같다): set -euo pipefail · EXIT trap 이 산출물을 올리고 **반드시 끈다** · 업로드 실패면 끄지 않고 exit 1
set -euo pipefail

REGION="${1:?region (gyeongsang|gangwon|chungcheong|jeolla|jeju|pooled)}"
COMBO="${2:?combo (GS-01 … GS-12)}"
shift 2
case "$REGION" in gyeongsang|gangwon|chungcheong|jeolla|jeju|pooled|seoul) ;; *) echo "모르는 region: $REGION" >&2; exit 2 ;; esac
# 서울(국립국어원)은 train 과 val 이 다른 완성본이고 기본 prefix 가 없다 — 대조 학습용 (2026-10-07, docs/SEOUL_NIKL_TRAIN_3P2H_2026-10-06.md)
if [ "$REGION" = seoul ] && { [ -z "${DATASET_S3:-}" ] || [ -z "${VAL_DATASET_S3:-}" ] || [ -z "${TRAIN_SET:-}" ]; }; then echo "seoul 은 DATASET_S3 · VAL_DATASET_S3 · TRAIN_SET 을 모두 준다" >&2; exit 2; fi
case "$COMBO" in GS-[0-9][0-9]) ;; *) echo "combo 는 GS-NN: $COMBO" >&2; exit 2 ;; esac
PHASE="${PHASE:-smoke}"
case "$PHASE" in smoke|full|direct) ;; *) echo "PHASE 는 smoke|full|direct: $PHASE" >&2; exit 2 ;; esac

cd "$HOME/accentury"
export ACCENTURY_DATA="${ACCENTURY_DATA:-$HOME/AccenturyData}"
BUCKET=s3://accentury-track2-325771561913
DATASET_S3="${DATASET_S3:-$BUCKET/datasets/dialect5_bench_v2/$REGION/01bab6ab0e761418788678fc7227ebe40110a56a35854469f3f9d4d13aded9af}"   # 지역별 완성본 — 5지역이 같은 v2 manifest(같은 해시)에서 나온다 (2026-09-28)
DATASET_ROOT="${DATASET_ROOT:-$ACCENTURY_DATA/work/finetune/dialect5_bench_v2/$REGION}"
TRAIN_SET="${TRAIN_SET:-${REGION}_train}"
case "$TRAIN_SET" in *[!A-Za-z0-9._-]*|.|..) echo "TRAIN_SET 는 안전한 세트 이름이어야 한다: $TRAIN_SET" >&2; exit 2 ;; esac
RUN_ID="${RUN_ID:-${REGION}_${COMBO}}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*) echo "RUN_ID 는 [A-Za-z0-9._-] 만: $RUN_ID" >&2; exit 2 ;; esac
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"
ADAPTER="$ACCENTURY_DATA/work/finetune/dialect5_lora/$RUN_ID"
TB_ROOT="$ACCENTURY_DATA/work/finetune/dialect5_tb"
TB="$TB_ROOT/$RUN_ID"
OUT="registry/finetune/dialect5_lora/$RUN_ID"
REMOTE="$BUCKET/results_dialect5/$REGION/$RUN_ID"
LOCK="$ACCENTURY_DATA/work/finetune/.upload_dialect5_$RUN_ID.lock"
UPLOAD_EVERY_SEC="${UPLOAD_EVERY_SEC:-600}"
GRID=src/finetune/gs_grid.json
TRAIN_MAN="$DATASET_ROOT/sets/$TRAIN_SET/manifest.csv"
VAL_DATASET_S3="${VAL_DATASET_S3:-}"
VAL_DATASET_ROOT="${VAL_DATASET_ROOT:-$DATASET_ROOT}"
if [ -n "$VAL_DATASET_S3" ] && [ "$VAL_DATASET_ROOT" = "$DATASET_ROOT" ]; then echo "VAL_DATASET_S3 를 주면 VAL_DATASET_ROOT 는 DATASET_ROOT 와 달라야 한다 (두 완성본이 한 폴더에 섞인다)" >&2; exit 2; fi
if [ -z "$VAL_DATASET_S3" ] && [ "$VAL_DATASET_ROOT" != "$DATASET_ROOT" ]; then echo "VAL_DATASET_ROOT 만 주면 무엇을 받을지 모른다 — VAL_DATASET_S3 를 같이 준다" >&2; exit 2; fi
VAL_MAN="$VAL_DATASET_ROOT/sets/${REGION}_val/manifest.csv"

mkdir -p "$OUT" "$ADAPTER" "$TB" "$(dirname "$LOCK")"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
log "RUN_ID=$RUN_ID REGION=$REGION COMBO=$COMBO PHASE=$PHASE ATTEMPT=$ATTEMPT extra: $*"
log "train $DATASET_S3 [$TRAIN_SET] · val ${VAL_DATASET_S3:-train 과 같은 완성본}"

# S3 sync 3회 재시도 + dryrun 검증 (run_lora_train.sh 와 같은 규율). *.tmp 는 항상 뺀다.
s3_up() {
  local src="$1" dst="$2"; shift 2
  local waits=(30 120 300) i rc out
  [ -d "$src" ] || { log "s3_up: 없음 $src (건너뜀)"; return 0; }
  for i in 0 1 2 3; do
    aws s3 sync "$src" "$dst" --exclude "*.tmp" "$@" >/dev/null 2>>"$OUT/upload.err"; rc=$?
    if [ "$rc" -eq 0 ]; then
      out=$(aws s3 sync "$src" "$dst" --exclude "*.tmp" "$@" --dryrun 2>&1); rc=$?
      if [ "$rc" -eq 0 ] && [ -z "$out" ]; then log "업로드 OK $src → $dst"; return 0; fi
      log "업로드 검증 실패(rc $rc) $src: ${out:0:160}"
    else
      log "업로드 실패(rc $rc) $src"
    fi
    if [ "$i" -lt 3 ]; then log "  ${waits[$i]}초 뒤 재시도 $((i+1))/3"; sleep "${waits[$i]}"; fi
  done
  log "UPLOAD_FAILED $src → $dst"; return 1
}
# 어댑터 공개: `_complete` 가 있는 step*/ · epochNN/ 만 (반쯤 쓴 폴더는 안 올린다) + train_state.pt 스냅샷 + best.json
publish_adapters() {
  local d
  for d in "$ADAPTER"/step[0-9][0-9][0-9][0-9][0-9][0-9] "$ADAPTER"/epoch[0-9][0-9]; do
    [ -d "$d" ] && [ -f "$d/_complete" ] || continue
    s3_up "$d/" "$REMOTE/adapters/$(basename "$d")/" || return 1
  done
  # 트레이너가 지운 step 폴더(keep-last 밖)는 원격에서도 지운다 — best·최근만 남기는 규칙을 S3 에도 적용
  aws s3 sync "$ADAPTER/" "$REMOTE/adapters/" --exclude "*" --include "step*/*" --delete --dryrun 2>/dev/null | grep -q "delete:" && \
    aws s3 sync "$ADAPTER/" "$REMOTE/adapters/" --exclude "*" --include "step*/*" --delete >/dev/null 2>>"$OUT/upload.err" || true
  [ -f "$ADAPTER/best.json" ] && { aws s3 cp "$ADAPTER/best.json" "$REMOTE/adapters/best.json" >/dev/null 2>>"$OUT/upload.err" || return 1; }
  if [ -f "$ADAPTER/train_state.pt" ]; then
    cp -f "$ADAPTER/train_state.pt" "$ADAPTER/.train_state.snapshot.tmp" && mv -f "$ADAPTER/.train_state.snapshot.tmp" "$ADAPTER/.train_state.snapshot"
    aws s3 cp "$ADAPTER/.train_state.snapshot" "$REMOTE/adapters/train_state.pt" >/dev/null 2>>"$OUT/upload.err" || return 1
  fi
  return 0
}
heartbeat() { date -u +%FT%TZ | aws s3 cp - "$REMOTE/report/heartbeat" >/dev/null 2>>"$OUT/upload.err" || true; }
upload_cycle() { ( trap - EXIT; flock -x 9 || exit 1; s3_up "$TB/" "$REMOTE/tb/"; s3_up "$OUT/" "$REMOTE/report/" --exclude "DONE.json"; heartbeat ) 9>"$LOCK"; }
command -v flock >/dev/null || { echo "flock 이 없다 — util-linux 를 설치한다" >&2; exit 2; }

# ---------------------------------------------------------------- 준비
.venv/bin/pip install -q "peft==0.20.0" tensorboard 2>&1 | tail -1
.venv/bin/python -c "import peft, torch, tensorboard; print('peft', peft.__version__, 'tensorboard', tensorboard.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/gpu.txt"

aws s3 ls "$DATASET_S3/_SUCCESS.json" >/dev/null || { echo "완성본 마커가 없다: $DATASET_S3/_SUCCESS.json (handoff 문서 참조)" >&2; exit 2; }
timeout 3h aws s3 sync "$DATASET_S3/" "$DATASET_ROOT/" >/dev/null       # sets/ 만 받으면 clip_file(../../clips/…)을 못 찾는다 — prefix 전체
if [ -n "$VAL_DATASET_S3" ]; then
  aws s3 ls "$VAL_DATASET_S3/_SUCCESS.json" >/dev/null || { echo "val 완성본 마커가 없다: $VAL_DATASET_S3/_SUCCESS.json" >&2; exit 2; }
  timeout 3h aws s3 sync "$VAL_DATASET_S3/" "$VAL_DATASET_ROOT/" >/dev/null
fi
[ -f "$TRAIN_MAN" ] && [ -f "$VAL_MAN" ] || { echo "manifest 가 없다: $TRAIN_MAN / $VAL_MAN" >&2; exit 2; }

# 조합 → 인자 (gs_grid.json). 공통 인자 뒤에 조합 인자, 그 뒤에 명령줄 추가 인자 (뒤가 이긴다)
mapfile -t COMBO_ARGS < <(python3 - "$GRID" "$COMBO" <<'PY'
import json, sys
g = json.load(open(sys.argv[1], encoding="utf-8")); c = g["조합"][sys.argv[2]]
for x in g["공통"]: print(x)
for x in ["--lr", str(c["lr"]), "--grad-accum", str(c["grad_accum"]), "--dropout", str(c["dropout"])] + [str(x) for x in c.get("추가", [])]: print(x)   # `추가` 는 조합별 덧붙임 (확인용 GS-13~20 · 2026-09-28)
PY
)
[ "${#COMBO_ARGS[@]}" -gt 0 ] || { echo "조합 인자를 못 만들었다: $COMBO" >&2; exit 2; }

PHASE_ARGS=(); RESUME_ARG=()
if [ "$PHASE" = direct ]; then
  # 확인용: 스모크 없이 처음부터 조기 종료(patience)·상한 15 epoch 까지 한 번에 (2026-09-28, GS-13~20). 안전 검사는 스모크와 같다
  if [ -n "$(ls -A "$ADAPTER" 2>/dev/null)" ]; then echo "어댑터 폴더가 비어 있지 않다: $ADAPTER — 다른 RUN_ID" >&2; exit 2; fi
  if aws s3 ls "$REMOTE/" >/dev/null 2>&1; then echo "S3 에 같은 RUN_ID 가 있다: $REMOTE" >&2; exit 2; fi
  PHASE_ARGS=(--run-kind main --val-loss-patience "${FULL_PATIENCE:-5}")
elif [ "$PHASE" = smoke ]; then
  if [ -n "$(ls -A "$ADAPTER" 2>/dev/null)" ]; then echo "어댑터 폴더가 비어 있지 않다: $ADAPTER — 다른 RUN_ID" >&2; exit 2; fi
  if aws s3 ls "$REMOTE/" >/dev/null 2>&1; then echo "S3 에 같은 RUN_ID 가 있다: $REMOTE" >&2; exit 2; fi
  PHASE_ARGS=(--run-epochs 2 --run-kind main --val-loss-patience "${SMOKE_PATIENCE:-0}")   # 스모크는 조기 종료 없이 2 epoch 을 다 돌려 같은 지점에서 비교한다 (멘토 9/15·9/24). run_kind 는 본학습과 같아야 --resume 이 된다
else
  # 본학습: 스모크가 남긴 상태(epoch02 + train_state.pt + step*/best.json)를 받아 잇는다. 옛 DONE 은 먼저 무효화.
  # 인스턴스 역할에 s3:DeleteObject 가 없어(2026-09-28 실측 — rm 실패로 4대가 멈춤) 지우지 않고 「진행 중」 표식으로 덮어쓴다. 감시는 phase 가 full 인 DONE 만 완료로 본다.
  if aws s3 ls "$REMOTE/report/DONE.json" >/dev/null 2>&1; then
    printf '{"run": "%s", "phase": "%s-in-progress", "attempt": "%s", "note": "옛 DONE 무효화 — 본학습 시작"}\n' "$RUN_ID" "$PHASE" "$ATTEMPT" \
      | aws s3 cp - "$REMOTE/report/DONE.json" >/dev/null 2>&1 || { echo "옛 DONE.json 을 덮어쓰지 못했다 — 재개하지 않는다" >&2; exit 2; }
  fi
  aws s3 sync "$REMOTE/adapters/" "$ADAPTER/" >/dev/null
  aws s3 sync "$REMOTE/report/" "$OUT/" --exclude "DONE.json" --exclude "heartbeat" >/dev/null
  aws s3 sync "$REMOTE/tb/" "$TB/" >/dev/null || true
  rm -f "$OUT/DONE.json"
  [ -f "$ADAPTER/train_state.pt" ] || { echo "PHASE=full 인데 $REMOTE/adapters/train_state.pt 가 없다 (스모크 먼저)" >&2; exit 2; }
  PHASE_ARGS=(--run-kind main --val-loss-patience "${FULL_PATIENCE:-3}")   # run_kind 는 스모크와 같은 main 이어야 --resume 이 받는다 (기본값 toy — 2026-09-28 실측 거부). patience: 키페이퍼 3 · 스모크 흔들림 기준 5
  RESUME_ARG=(--resume)
fi

UPLOAD_PID=""
cleanup() {
  rc=$?
  set +e
  [ -n "$UPLOAD_PID" ] && kill "$UPLOAD_PID" 2>/dev/null
  log "=== cleanup (exit $rc · attempt $ATTEMPT · phase $PHASE) ==="
  fail=0
  ( trap - EXIT; flock -x 9 || exit 1
    publish_adapters || exit 1
    s3_up "$TB/" "$REMOTE/tb/" || exit 1
    s3_up "$OUT/" "$REMOTE/report/" --exclude "DONE.json" || exit 1
    exit 0 ) 9>"$LOCK" || fail=1
  if [ "$fail" = 1 ]; then
    log "UPLOAD_FAILED — 인스턴스를 끄지 않는다. 사람이 $REMOTE 로 올린 뒤 끈다"
    aws s3 cp "$OUT/summary.txt" "$REMOTE/report/summary.txt" >/dev/null 2>&1
    exit 1
  fi
  python3 - "$OUT/DONE.json.tmp" "$RUN_ID" "$ATTEMPT" "$rc" "$ADAPTER" "$PHASE" <<'PY' || { log "UPLOAD_FAILED DONE.json 생성 실패"; exit 1; }
import json, sys, pathlib, datetime
out, run, attempt, rc, adapter, phase = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), pathlib.Path(sys.argv[5]), sys.argv[6]
best = json.loads((adapter / "best.json").read_text()) if (adapter / "best.json").exists() else None
steps = sorted(p.name for p in adapter.glob("step[0-9]*") if (p / "_complete").exists())
epochs = sorted(p.name for p in adapter.glob("epoch[0-9][0-9]") if (p / "_complete").exists())
doc = {"run": run, "phase": phase, "attempt": attempt, "exit": rc, "best": best, "step_checkpoints": steps, "epoch_checkpoints": epochs,
       "done": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n"); json.loads(pathlib.Path(out).read_text())
PY
  mv -f "$OUT/DONE.json.tmp" "$OUT/DONE.json"
  aws s3 cp "$OUT/summary.txt" "$REMOTE/report/summary.txt" >/dev/null 2>&1
  aws s3 cp "$OUT/DONE.json" "$REMOTE/report/DONE.json" >/dev/null 2>>"$OUT/upload.err" && log "DONE.json 공개" || { log "UPLOAD_FAILED DONE.json"; exit 1; }
  if [ "${NO_SHUTDOWN:-0}" = "1" ]; then log "NO_SHUTDOWN=1 — 끄지 않는다"; exit "$rc"; fi
  sudo shutdown -h now
  exit "$rc"
}
trap cleanup EXIT

heartbeat
( trap - EXIT; while :; do sleep "$UPLOAD_EVERY_SEC"; upload_cycle >> "$OUT/upload_loop.log" 2>&1 || true; done ) &
UPLOAD_PID=$!

log "trainer args: ${COMBO_ARGS[*]} ${PHASE_ARGS[*]} ${RESUME_ARG[*]:-} $*"
set +e
.venv/bin/python src/finetune/train_lora_whisper.py \
    --manifest "$TRAIN_MAN" --val-manifest "$VAL_MAN" --limit 0 --eval-n 0 \
    --out "$ADAPTER" --report "$OUT/lora_${RUN_ID}_${PHASE}.json" --tensorboard "$TB" \
    "${COMBO_ARGS[@]}" "${PHASE_ARGS[@]}" ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} "$@" 2>&1 | tee -a "$OUT/lora_${RUN_ID}_${PHASE}.log"
rc=${PIPESTATUS[0]}
set -e
log "trainer exit $rc"
exit "$rc"
