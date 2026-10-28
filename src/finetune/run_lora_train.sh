#!/usr/bin/env bash
# LoRA 본 학습 — EC2 래퍼 (`docs/TRACK2_BENCHMARK_V2_SPEC.md` §5 6번 · 하이퍼파라미터는 `FINETUNE_METHOD_SELECTION.md` §5.4.1 · 2차는 `TRACK2_FINETUNE_V2_SPEC.md`)
#
# 쓰기
#   src/finetune/run_lora_train.sh v2_train_weighted                       # 새 run — RUN_ID 는 자동 (UTC 시각_세트)
#   RUN_ID=r2_lr2e-5 src/finetune/run_lora_train.sh v2_train_weighted --lr 2e-5   # 이름을 정해 주는 새 run (2차)
#   RUN_ID=r2_lr2e-5 RESUME=1 src/finetune/run_lora_train.sh v2_train_weighted --lr 2e-5   # 죽은 run 이어 하기 (S3 에서 상태를 내려받는다)
#   NO_SHUTDOWN=1 src/finetune/run_lora_train.sh v2_train_weighted --epochs 1 --limit 512   # 사람이 붙어 볼 때
#
# 세 번째 인자부터는 train_lora_whisper.py 로 그대로 넘어간다 (--epochs · --lr · --alpha · --r …).
# **--val-tau 는 주지 않는다** — §4.3 5번 조기 종료는 2026-09-09 삭제됐다 (트레이너도 거부한다).
#
# 규율 (코덱스 2차 10번 — 토이 러너에 set -e 가 없어 실패가 묻혔다)
#   - set -euo pipefail 로 시작한다.
#   - EXIT trap 을 건다: 성공이든 실패든 있는 산출물을 S3 에 올리고 **반드시 끈다**. 끈 채로 두고 싶으면 NO_SHUTDOWN=1.
#
# 고유 run 폴더 (코덱스 4차 8번 · 2026-09-09)
#   - 어댑터·보고·S3 prefix 가 전부 RUN_ID 아래다. 같은 폴더를 두 run 이 쓰지 못한다 — 트레이너도 `--out` 이 비어 있으면서 `--resume` 이 없을 때만 시작한다.
#   - RESUME=1 이면 S3 `results_lora_main/<RUN_ID>/` 를 내려받아 `--resume` 으로 잇는다. 설정이 다르면 트레이너가 거부한다. 원격 DONE.json 은 지운다.
#
# S3 공개 규약 (코덱스 4차 9·10번 · 5차 2·3·4·5번 · 2026-09-09) — 채점기(`run_scorer.sh`)가 다른 인스턴스에서 이것만 보고 움직인다
#   adapters/epochNN/<본체>           트레이너가 폴더를 다 쓰고 로컬 `_complete` 를 남긴 것만 올린다 (반쯤 쓴 폴더는 안 올린다)
#   adapters/epochNN/_complete.json   본체 sync + dryrun 검증이 끝난 **뒤** 따로 올리는 공개 마커 — 파일 이름·크기 목록. 채점기는 이것이 있는 폴더만 받고 크기를 대조한다
#                                     (sync 는 병렬·이름순이라 마커를 같이 올리면 가중치보다 먼저 보일 수 있다 — 코덱스 5차 2번)
#   adapters/train_state.pt           재개용 (cp 로 원자적 교체)
#   report/heartbeat                  주기 업로드마다 갱신하는 시각 — 채점기의 「학습기가 살아 있나」 판정 (코덱스 5차 5번)
#   report/DONE.json                  cleanup 이 **모든 업로드를 검증한 뒤 마지막에** 올리는 종료 manifest(시도 ID·rc·최종 체크포인트 목록) — 채점기의 완료 판정 (5차 4번)
#   report/ · val_shards/ · report/val_scores/   보고 JSON·로그·summary · (훅을 썼을 때) val 조각·채점 JSON
#   업로드는 전부 `s3_up`(3회 재시도 + dryrun 검증 · *.tmp 제외)이고, 주기 업로드와 cleanup 은 **같은 flock** 으로 직렬화한다 — 진행 중 전송이 끝난 뒤에야 cleanup 이 올린다 (5차 3번).
#   하나라도 실패하면 summary 에 `UPLOAD_FAILED` 를 적고 **인스턴스를 끄지 않고** exit 1 로 남긴다 — 사람이 들어가 올린다 (종료 동작 stop 이라 디스크는 남는다).
set -euo pipefail

SET="${1:?교재 세트 이름이 필요하다 — v2_train_weighted 또는 v2_train_random}"
shift
case "$SET" in v2_train_weighted|v2_train_random) ;; *) echo "모르는 교재 세트: $SET" >&2; exit 2 ;; esac
for x in "$@"; do case "$x" in --val-tau|--val-tau=*) echo "--val-tau 는 주지 않는다 — §4.3 5번 조기 종료는 삭제됐다 (2026-09-09)" >&2; exit 2 ;; esac; done

cd "$HOME/accentury"
export ACCENTURY_DATA="${ACCENTURY_DATA:-$HOME/AccenturyData}"
BUCKET=s3://accentury-track2-325771561913
BENCH=track2_bench_v2_val
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)_$SET}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*) echo "RUN_ID 는 [A-Za-z0-9._-] 만: $RUN_ID" >&2; exit 2 ;; esac
RESUME="${RESUME:-0}"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"                            # 시도 ID — 재개할 때마다 다르다 (DONE.json 에 적힌다)
TRAIN_ROOT="$ACCENTURY_DATA/work/finetune/track2_bench_v2/sets"
VAL_ROOT="$ACCENTURY_DATA/work/finetune/track2_bench_v2_val"
ADAPTER="$ACCENTURY_DATA/work/finetune/lora_main/$RUN_ID"      # 어댑터 + train_state.pt (데이터 디스크)
SHARDS="$ACCENTURY_DATA/work/finetune/val_shards/$RUN_ID"       # val 훅이 쓰는 발화별 조각 (epNN/)
OUT="registry/finetune/lora_main/$RUN_ID"                       # 보고 JSON·로그·summary
SCORES="registry/benchmark"                                     # val 훅이 쓰는 채점 JSON: ${BENCH}_stt__${RUN_ID}_epNN.json
REMOTE="$BUCKET/results_lora_main/$RUN_ID"
LOCK="$ACCENTURY_DATA/work/finetune/.upload_$RUN_ID.lock"
VAL_HOOK="${VAL_HOOK-.venv/bin/python src/finetune/val_hook.py}"   # 글루 (`val_hook.py`). 2차는 채점기가 따로 있으므로 VAL_HOOK= (빈 값)으로 준다.
UPLOAD_EVERY_SEC="${UPLOAD_EVERY_SEC:-600}"
echo "RUN_ID=$RUN_ID SET=$SET RESUME=$RESUME ATTEMPT=$ATTEMPT"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }

# S3 sync 3회 재시도 + dryrun 검증 (코덱스 4차 10번 · 5차 8번: sync 와 dryrun 의 종료 코드·출력을 각각 바로 잡는다). *.tmp 는 항상 뺀다.
s3_up() {                                                        # s3_up <로컬/> <원격/> [aws s3 sync 추가 인자…]
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

# 체크포인트 공개 — 로컬 `_complete` 가 있고 원격 `_complete.json` 이 없는 epochNN 을 본체 → 검증 → 마커 순으로 올린다 (5차 2번)
publish_epochs() {
  local d nn
  for d in "$ADAPTER"/epoch[0-9][0-9]; do
    [ -d "$d" ] && [ -f "$d/_complete" ] || continue
    nn=${d##*epoch}
    if aws s3 ls "$REMOTE/adapters/epoch$nn/_complete.json" >/dev/null 2>&1; then continue; fi
    s3_up "$d/" "$REMOTE/adapters/epoch$nn/" --exclude "_complete.json" || return 1
    python3 - "$d" "$nn" "$RUN_ID" "$ATTEMPT" > "$d/_complete.json.tmp" <<'PY'
import json, sys, pathlib, datetime
d = pathlib.Path(sys.argv[1])
files = {p.name: p.stat().st_size for p in sorted(d.iterdir()) if p.is_file() and p.name != "_complete.json" and not p.name.endswith(".tmp")}
print(json.dumps({"epoch": int(sys.argv[2]), "run": sys.argv[3], "attempt": sys.argv[4], "files": files,
                  "uploaded": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
PY
    mv -f "$d/_complete.json.tmp" "$d/_complete.json"
    aws s3 cp "$d/_complete.json" "$REMOTE/adapters/epoch$nn/_complete.json" >/dev/null 2>>"$OUT/upload.err" || { log "UPLOAD_FAILED 마커 epoch$nn"; return 1; }
    log "체크포인트 공개 epoch$nn ($(python3 -c "import json;print(len(json.load(open('$d/_complete.json'))['files']))")개 파일)"
  done
  if [ -f "$ADAPTER/train_state.pt" ]; then                      # 재개용 — 트레이너가 도중에 교체해도 전송 원본이 섞이지 않게 스냅샷을 떠서 올린다 (5차b 5번)
    cp -f "$ADAPTER/train_state.pt" "$ADAPTER/.train_state.snapshot.tmp" && mv -f "$ADAPTER/.train_state.snapshot.tmp" "$ADAPTER/.train_state.snapshot"
    aws s3 cp "$ADAPTER/.train_state.snapshot" "$REMOTE/adapters/train_state.pt" >/dev/null 2>>"$OUT/upload.err" || { log "train_state.pt 업로드 실패 (다음 주기)"; return 1; }
  fi
  return 0
}
heartbeat() { date -u +%FT%TZ | aws s3 cp - "$REMOTE/report/heartbeat" >/dev/null 2>>"$OUT/upload.err" || true; }
upload_cycle() { ( trap - EXIT; flock -x 9 || { echo "flock 실패 — 잠금 없이 공개하지 않는다" >&2; exit 1; }; publish_epochs; heartbeat ) 9>"$LOCK"; }   # 5차 3번 · 5차b 2번
command -v flock >/dev/null || { echo "flock 이 없다 — 업로드 직렬화 불가. util-linux 를 설치한다" >&2; exit 2; }

mkdir -p "$OUT" "$ADAPTER" "$SHARDS" "$(dirname "$LOCK")"
if [ "$RESUME" = "1" ]; then
  # 옛 종료 manifest 를 **먼저** 무효화한다 — 지우지 못하면 재개하지 않는다 (5차 4번 · 5차b 3번). 그 뒤에야 내려받는다.
  if aws s3 ls "$REMOTE/report/DONE.json" >/dev/null 2>&1; then
    aws s3 rm "$REMOTE/report/DONE.json" >/dev/null 2>&1 || { echo "RESUME: 옛 DONE.json 을 지우지 못했다 — 채점기가 완료로 오판할 수 있어 재개하지 않는다" >&2; exit 2; }
    aws s3 ls "$REMOTE/report/DONE.json" >/dev/null 2>&1 && { echo "RESUME: DONE.json 이 아직 있다" >&2; exit 2; }
  fi
  aws s3 sync "$REMOTE/adapters/" "$ADAPTER/"                   # 상태·어댑터를 내려받아 잇는다
  aws s3 sync "$REMOTE/report/" "$OUT/" --exclude "DONE.json" --exclude "heartbeat"
  aws s3 sync "$REMOTE/val_shards/" "$SHARDS/" || true
  rm -f "$OUT/DONE.json"
  # 재개로 다시 계산할 epoch(상태 epoch 보다 큰 것)의 옛 공개 마커는 로컬에서 지워 재공개되게 한다 (5차b 5번)
  st_ep=$(python3 -c "import torch,sys; print(torch.load('$ADAPTER/train_state.pt', map_location='cpu', weights_only=False)['epoch'])" 2>/dev/null || echo "")
  for d in "$ADAPTER"/epoch[0-9][0-9]; do [ -d "$d" ] || continue; n=$((10#${d##*epoch})); if [ -n "$st_ep" ] && [ "$n" -gt "$st_ep" ]; then rm -rf "$d"; aws s3 rm "$REMOTE/adapters/epoch$(printf %02d $n)/" --recursive >/dev/null 2>&1 || true; fi; done
  [ -f "$ADAPTER/train_state.pt" ] || { echo "RESUME=1 인데 $REMOTE/adapters/train_state.pt 가 없다" >&2; exit 2; }
  RESUME_ARG=(--resume)
else
  if [ -n "$(ls -A "$ADAPTER" 2>/dev/null)" ]; then
    echo "어댑터 폴더가 비어 있지 않다: $ADAPTER — 다른 RUN_ID 를 쓰거나 RESUME=1 (코덱스 4차 8번)" >&2; exit 2
  fi
  if aws s3 ls "$REMOTE/" >/dev/null 2>&1; then
    echo "S3 에 같은 RUN_ID 가 있다: $REMOTE — 다른 RUN_ID 를 쓰거나 RESUME=1" >&2; exit 2
  fi
  RESUME_ARG=()
fi
log "RUN_ID=$RUN_ID SET=$SET RESUME=$RESUME ATTEMPT=$ATTEMPT args: $*"

UPLOAD_PID=""
cleanup() {                                           # 실패해도 인스턴스를 켜 둔 채 남기지 않는다 — 업로드가 안 됐을 때만 예외
  rc=$?
  set +e
  [ -n "$UPLOAD_PID" ] && kill "$UPLOAD_PID" 2>/dev/null    # 루프는 멈추고, 진행 중 전송은 아래 flock 이 기다린다
  log "=== cleanup (exit $rc · attempt $ATTEMPT) ==="
  log "exit $rc"
  fail=0
  (
    trap - EXIT
    flock -x 9 || exit 1                              # 주기 업로드가 진행 중이면 끝날 때까지 기다린다 (5차 3번) · 잠금 실패면 공개하지 않는다 (5차b 2번)
    publish_epochs || exit 1
    s3_up "$OUT/" "$REMOTE/report/" --exclude "DONE.json" || exit 1                # 로그·보고 JSON·summary (DONE 은 맨 마지막)
    s3_up "$SHARDS/" "$REMOTE/val_shards/" || exit 1                               # 9번: (훅을 썼을 때) 발화별 val 조각
    mkdir -p "$OUT/val_scores"; cp -f "$SCORES/${BENCH}_stt__${RUN_ID}_ep"*.json "$OUT/val_scores/" 2>/dev/null
    s3_up "$OUT/val_scores/" "$REMOTE/report/val_scores/" || exit 1
    exit 0
  ) 9>"$LOCK" || fail=1
  if [ "$fail" = "1" ]; then
    log "UPLOAD_FAILED — 인스턴스를 끄지 않는다. 사람이 들어가 $REMOTE 로 올린 뒤 끈다 (코덱스 4차 10번)"
    aws s3 cp "$OUT/summary.txt" "$REMOTE/report/summary.txt" >/dev/null 2>&1
    exit 1
  fi
  # 종료 manifest — 모든 업로드가 검증된 뒤 맨 마지막에 공개한다 (5차 4번). 채점기는 이 목록으로 「전부 채점했나」를 판정한다.
  if ! python3 - "$OUT/DONE.json.tmp" "$RUN_ID" "$ATTEMPT" "$rc" "$ADAPTER" <<'PY'
import json, sys, pathlib, datetime
out, run, attempt, rc, adapter = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), pathlib.Path(sys.argv[5])
cks = sorted(int(p.name[5:]) for p in adapter.glob("epoch[0-9][0-9]") if (p / "_complete.json").exists())
doc = {"run": run, "attempt": attempt, "exit": rc, "checkpoints": cks, "done": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
json.loads(pathlib.Path(out).read_text())                       # 쓴 것을 다시 읽어 검증 (5차b 4번)
PY
  then log "UPLOAD_FAILED DONE.json 생성 실패 — 인스턴스를 끄지 않는다"; exit 1; fi
  mv -f "$OUT/DONE.json.tmp" "$OUT/DONE.json"
  aws s3 cp "$OUT/summary.txt" "$REMOTE/report/summary.txt" >/dev/null 2>&1
  if aws s3 cp "$OUT/DONE.json" "$REMOTE/report/DONE.json" >/dev/null 2>>"$OUT/upload.err"; then log "DONE.json 공개 ($(tr -d '\n ' < "$OUT/DONE.json" | cut -c1-120))"
  else log "UPLOAD_FAILED DONE.json — 인스턴스를 끄지 않는다"; exit 1; fi
  if [ "${NO_SHUTDOWN:-0}" = "1" ]; then log "NO_SHUTDOWN=1 — 인스턴스를 끄지 않는다"; exit "$rc"; fi
  sudo shutdown -h now
  exit "$rc"
}
trap cleanup EXIT

.venv/bin/pip install -q "peft==0.20.0" 2>&1 | tail -1        # 토이 실측과 같은 버전으로 못 박는다 (코덱스 2차 13번)
.venv/bin/python -c "import peft, torch; print('peft', peft.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/gpu.txt"

aws s3 sync "$BUCKET/bench_v2/train_sets/$SET/" "$TRAIN_ROOT/$SET/"       # 0. 교재 (시험지는 받지 않는다)
HOOK=()
if [ -n "$VAL_HOOK" ]; then
  aws s3 sync "$BUCKET/bench_v2_val/sets/" "$VAL_ROOT/sets/"              # 1. val (훅을 줄 때만)
  HOOK=(--val-hook "$VAL_HOOK --shard-root $SHARDS")
fi

heartbeat
( trap - EXIT; while :; do sleep "$UPLOAD_EVERY_SEC"; upload_cycle >> "$OUT/upload_loop.log" 2>&1 || true; done ) &
UPLOAD_PID=$!

set +e                                                                     # 학습의 실패(OOM=3 포함)는 잡아서 기록하고, 업로드는 trap 이 한다
.venv/bin/python src/finetune/train_lora_whisper.py \
    --manifest "$TRAIN_ROOT/$SET/manifest.csv" \
    --run-kind main --limit 0 \
    --out "$ADAPTER" --report "$OUT/lora_main_$RUN_ID.json" \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} ${HOOK[@]+"${HOOK[@]}"} "$@" 2>&1 | tee -a "$OUT/lora_main_$RUN_ID.log"   # 빈 배열도 set -u 에서 안전한 형태
rc=${PIPESTATUS[0]}                                                        # tee 가 아니라 학습의 종료 코드를 본다
set -e
log "train exit $rc"
if [ "$rc" -eq 3 ]; then log "OOM — batch·grad_ckpt 를 다시 본다 (§5.4.1 물리 상한 실측)"; fi
exit "$rc"                                                                 # 업로드·종료는 EXIT trap 이 한다
