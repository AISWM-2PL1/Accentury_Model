#!/usr/bin/env bash
# LoRA 체크포인트 채점기 — EC2 래퍼 (학습기와 다른 GPU 에서 체크포인트가 나오는 대로 val 8세트 추론·채점)
#
# 왜  1차에서는 이 역할을 인스턴스 위에서 손으로 짠 `watch_ckpt.sh` 가 했고, 그 파일은 repo 에 없다. 종료 조건이 없어
#     `while true` 로 남았고 채점기 2대가 작업 종료 뒤 37~38시간 유휴로 돌아 약 76 GPU 시간을 버렸다
#     (inference 명세 §14 18번 · 2026-09-09 이 파일이 그 정본이다). 어댑터를 얹은 val 추론은 체크포인트당 127분(1차 `_run_meta.json`)이라
#     학습기에 붙이면 학습이 그만큼 늦어진다 — 그래서 채점기를 따로 둔다 (2차 명세 §5).
#
# 쓰기
#   EPOCH_MIN=8 src/finetune/run_scorer.sh <RUN_ID> [<RUN_ID> …]   # 학습기 RUN_ID 하나 이상. epoch 8 미만은 채점하지 않는다 (2차 명세 §5)
#   MAKE_BASELINE=1 … run_scorer.sh …                             # 이 채점기가 무튜닝 조각을 만든다 (하나에만 준다 — 코덱스 5차 10번)
#   IDLE_MAX_MIN=240 NO_SHUTDOWN=1 src/finetune/run_scorer.sh r2_lr2e-5
#
# 학습기(`run_lora_train.sh`)의 S3 공개 규약을 그대로 읽는다 (그 파일 머리말):
#   adapters/epochNN/_complete.json 이 있는 폴더만 받고 파일 크기를 대조한다 · report/heartbeat 로 학습기 생존을 본다 · report/DONE.json 으로 완료를 판정한다.
#
# 종료 조건 (§14 18번 — 반드시 셋 중 하나로 끝난다 · 코덱스 5차 1·4·5번 반영)
#   ① 완료: 모든 RUN_ID 에 DONE.json 이 있고, 그 목록의 체크포인트(EPOCH_MIN 이상)를 전부 채점했다.
#   ② 학습기 사망: DONE.json 이 없는데 heartbeat 가 IDLE_MAX_MIN(기본 180분) 넘게 안 바뀌었다 (heartbeat 자체가 START_MAX_MIN(기본 60분) 안에 안 생겨도 같다).
#      채점할 게 없어도 heartbeat 가 살아 있으면 기다린다 — EPOCH_MIN=8 이면 첫 대상은 학습 시작 약 9시간 뒤다.
#   ③ 업로드 실패: `UPLOAD_FAILED` 면 끄지 않고 exit 1 (사람이 올린다 — 학습기와 같은 규칙).
#
# 산출물 (코덱스 4차 9번 · 5차 11번 — 채점기를 바꿔 켜도 다시 추론하지 않는다)
#   results_lora_main/<RUN_ID>/val_shards/epNN/A__v2val_*.csv.gz + _run_meta.json + _scored.json   발화별 조각 + 채점 완료 마커(어댑터 마커·조건)
#   results_lora_main/<RUN_ID>/report/val_scores/<BENCH>_stt__<RUN_ID>_epNN.json · epNN_val_score.json
#   bench_v2_val/baseline_shards/ (+ _complete.json)   무튜닝 조각 — MAKE_BASELINE=1 인 채점기 하나가 만든다 (run 무관 · --select 의 baseline)
#   선택은 오프라인: score_val_track2.py --select --baseline baseline --model baseline=<baseline_shards> --model <RUN_ID>_epNN=<val_shards/epNN> …
set -euo pipefail

[ $# -eq 1 ] || { echo "학습기 RUN_ID 는 정확히 하나 (채점기 1대 = 학습기 1대 · 여러 개는 완료·유휴 타이머가 run 별로 분리돼 있지 않다 — 코덱스 5차b 6번)" >&2; exit 2; }
RUNS=("$@")
for r in "${RUNS[@]}"; do case "$r" in *[!A-Za-z0-9._-]*) echo "RUN_ID 는 [A-Za-z0-9._-] 만: $r" >&2; exit 2 ;; esac; done

cd "$HOME/accentury"
export ACCENTURY_DATA="${ACCENTURY_DATA:-$HOME/AccenturyData}"
BUCKET=s3://accentury-track2-325771561913
BUCKET_NAME=accentury-track2-325771561913
BENCH=track2_bench_v2_val
COND="${COND:-A}"
EPOCH_MIN="${EPOCH_MIN:-0}"
IDLE_MAX_MIN="${IDLE_MAX_MIN:-180}"
START_MAX_MIN="${START_MAX_MIN:-60}"
POLL_SEC="${POLL_SEC:-300}"
VAL_ROOT="$ACCENTURY_DATA/work/finetune/$BENCH"
BASE_SHARDS="$ACCENTURY_DATA/work/finetune/val_shards/baseline"
SCORER_ID="scorer_$(date -u +%Y%m%dT%H%MZ)"
OUT="registry/finetune/scorer/$SCORER_ID"                      # 채점기 자체 로그·summary
mkdir -p "$OUT" "$VAL_ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
log "SCORER $SCORER_ID · RUNS ${RUNS[*]} · EPOCH_MIN $EPOCH_MIN · IDLE_MAX_MIN $IDLE_MAX_MIN · MAKE_BASELINE ${MAKE_BASELINE:-0}"

s3_up() {                                                        # run_lora_train.sh 와 같은 규칙 (3회 재시도 + dryrun 검증 · *.tmp 제외)
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
s3_age_min() {                                                   # 객체의 LastModified 로부터 지난 분. 없으면 빈 문자열
  local key="$1" lm
  lm=$(aws s3api head-object --bucket "$BUCKET_NAME" --key "$key" --query LastModified --output text 2>/dev/null) || { echo ""; return 0; }
  python3 -c "import sys,datetime; t=datetime.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')); print(int((datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()//60))" "$lm"
}

FAIL=0
STOP_REASON=""
cleanup() {
  rc=$?
  set +e
  log "=== cleanup (exit $rc · 이유: ${STOP_REASON:-없음}) ==="
  log "exit $rc"
  s3_up "$OUT/" "$BUCKET/results_scorer/$SCORER_ID/" || FAIL=1
  if [ "$FAIL" = "1" ]; then log "UPLOAD_FAILED — 인스턴스를 끄지 않는다. 사람이 올린 뒤 끈다"; exit 1; fi
  if [ "${NO_SHUTDOWN:-0}" = "1" ]; then log "NO_SHUTDOWN=1 — 끄지 않는다"; exit "$rc"; fi
  sudo shutdown -h now
  exit "$rc"
}
trap cleanup EXIT

.venv/bin/pip install -q "peft==0.20.0" 2>&1 | tail -1
.venv/bin/python -c "import peft, torch; print('peft', peft.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/gpu.txt"
aws s3 sync "$BUCKET/bench_v2_val/sets/" "$VAL_ROOT/sets/"                 # val 8세트 (8.353h)

# 무튜닝 조각 — MAKE_BASELINE=1 인 채점기 하나만 만든다. 이미 _complete.json 이 있으면 건너뛴다 (5차 10번: 동시 생성·불완전 상태 방지)
if [ "${MAKE_BASELINE:-0}" = "1" ] && ! aws s3 ls "$BUCKET/bench_v2_val/baseline_shards/_complete.json" >/dev/null 2>&1; then
  log "무튜닝 조각을 만든다 (MAKE_BASELINE=1)"
  mkdir -p "$BASE_SHARDS"
  .venv/bin/python src/stt/run_track2_stt.py --bench "$BENCH" --all --cond "$COND" --out-dir "$BASE_SHARDS" --device auto >> "$OUT/baseline.log" 2>&1
  .venv/bin/python src/stt/score_val_track2.py --model "baseline=$BASE_SHARDS" --cond "$COND" >> "$OUT/baseline.log" 2>&1
  s3_up "$BASE_SHARDS/" "$BUCKET/bench_v2_val/baseline_shards/" --exclude "_complete.json" || { FAIL=1; STOP_REASON="무튜닝 조각 업로드 실패"; exit 1; }
  python3 -c "
import json,pathlib,datetime; d=pathlib.Path('$BASE_SHARDS')
print(json.dumps({'files':{p.name:p.stat().st_size for p in sorted(d.iterdir()) if p.is_file() and p.name!='_complete.json'},'cond':'$COND','made':datetime.datetime.now().astimezone().isoformat(timespec='seconds')},ensure_ascii=False))" > "$BASE_SHARDS/_complete.json"
  aws s3 cp "$BASE_SHARDS/_complete.json" "$BUCKET/bench_v2_val/baseline_shards/_complete.json" >/dev/null || { FAIL=1; STOP_REASON="무튜닝 마커 업로드 실패"; exit 1; }
  log "무튜닝 조각 공개"
fi

# ── 체크포인트 하나 채점 ────────────────────────────────────────
scored_marker() { echo "$ACCENTURY_DATA/work/finetune/val_shards/$1/ep$2/_scored.json"; }

score_one() {                                                    # score_one <RUN_ID> <NN>  · 0 채점함 · 2 아직 못 받음 · 1 실패
  local run="$1" nn="$2"
  local remote="$BUCKET/results_lora_main/$run"
  local ck="$ACCENTURY_DATA/work/finetune/lora_main/$run/epoch$nn"
  local shards="$ACCENTURY_DATA/work/finetune/val_shards/$run"
  mkdir -p "$ck" "$shards/ep$nn"
  # 5차 11번 — 원격에 채점 완료 마커가 있으면(다른 채점기가 했으면) 추론하지 않고 로컬 마커만 만든다
  if aws s3 cp "$remote/val_shards/ep$nn/_scored.json" "$(scored_marker "$run" "$nn")" >/dev/null 2>&1; then
    log "epoch$nn 은 이미 채점돼 있다 (원격 _scored.json) — 건너뜀"; return 0
  fi
  # 5차 2번 — 공개 마커가 있는 폴더만 받고, 마커의 파일 크기와 대조한다
  aws s3 cp "$remote/adapters/epoch$nn/_complete.json" "$ck/_complete.json" >/dev/null 2>&1 || { log "epoch$nn 아직 공개 전 (_complete.json 없음)"; return 2; }
  aws s3 sync "$remote/adapters/epoch$nn/" "$ck/" --exclude "val_score.json" >/dev/null 2>>"$OUT/upload.err" || { log "epoch$nn 내려받기 실패"; return 2; }
  if ! python3 - "$ck" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); want = json.loads((d / "_complete.json").read_text())["files"]
bad = {n: (s, (d / n).stat().st_size if (d / n).exists() else None) for n, s in want.items() if not (d / n).exists() or (d / n).stat().st_size != s}
if bad: print("크기 불일치:", bad); sys.exit(1)
PY
  then log "epoch$nn 받은 파일이 마커와 다르다 — 다음 폴링"; return 2; fi
  log "채점 시작 $run epoch$nn"
  local t0=$SECONDS
  if ! .venv/bin/python src/finetune/val_hook.py "$ck" "$((10#$nn))" --bench "$BENCH" --cond "$COND" --shard-root "$shards" >> "$OUT/score_${run}.log" 2>&1; then
    log "채점 실패 $run epoch$nn — $OUT/score_${run}.log (다음 폴링에 다시 시도)"; return 1
  fi
  mkdir -p "$shards/val_scores"
  cp -f "registry/benchmark/${BENCH}_stt__${run}_ep${nn}.json" "$shards/val_scores/" 2>/dev/null || true
  cp -f "$ck/val_score.json" "$shards/val_scores/ep${nn}_val_score.json" 2>/dev/null || true
  s3_up "$shards/ep$nn/" "$remote/val_shards/ep$nn/" --exclude "_scored.json" || { FAIL=1; return 1; }
  s3_up "$shards/val_scores/" "$remote/report/val_scores/" || { FAIL=1; return 1; }
  python3 -c "
import json,datetime; m=json.load(open('$ck/_complete.json'))
print(json.dumps({'run':'$run','epoch':$((10#$nn)),'adapter_attempt':m.get('attempt'),'adapter_files':m.get('files'),'cond':'$COND','scorer':'$SCORER_ID',
  'scored':datetime.datetime.now().astimezone().isoformat(timespec='seconds')},ensure_ascii=False))" > "$(scored_marker "$run" "$nn")"
  aws s3 cp "$(scored_marker "$run" "$nn")" "$remote/val_shards/ep$nn/_scored.json" >/dev/null 2>>"$OUT/upload.err" || { FAIL=1; log "UPLOAD_FAILED _scored.json epoch$nn"; return 1; }
  log "채점 끝 $run epoch$nn · $(( (SECONDS - t0) / 60 ))분 · 고령 micro $(python3 -c "import json;print(json.load(open('$ck/val_score.json'))['고령_micro_cer'])" 2>/dev/null)"
  return 0
}

list_published() {                                               # <RUN_ID> → 공개된(_complete.json 있는) epoch NN 목록 (stdout 은 목록만 · 진단은 stderr/log — 5차b 1번)
  local run="$1" out rc
  out=$(aws s3 ls "$BUCKET/results_lora_main/$run/adapters/" --recursive 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then
    [ -n "$out" ] && log "s3 ls 오류 ($run) rc $rc — 이번 폴링은 빈 목록으로 본다" >&2   # 빈 prefix 도 rc 1 이지만 출력이 없다
    return 0
  fi
  printf '%s\n' "$out" | awk '$4 ~ /adapters\/epoch[0-9][0-9]\/_complete\.json$/{n=$4; sub(/.*epoch/,"",n); sub(/\/.*/,"",n); print n}' | sort -u
}

t_start=$SECONDS
last_progress=$SECONDS                                           # 마지막으로 체크포인트 하나를 채점한 시각
while :; do
  all_done=1
  for run in "${RUNS[@]}"; do
    remote="$BUCKET/results_lora_main/$run"
    for nn in $(list_published "$run"); do
      [ "$((10#$nn))" -lt "$EPOCH_MIN" ] && continue
      [ -f "$(scored_marker "$run" "$nn")" ] && continue
      if score_one "$run" "$nn"; then last_progress=$SECONDS; fi
      [ "$FAIL" = "1" ] && { STOP_REASON="업로드 실패"; exit 1; }
    done
    # 완료 판정 — DONE.json 의 체크포인트 목록(EPOCH_MIN 이상)이 전부 채점됐나 (5차 4번)
    if aws s3 cp "$remote/report/DONE.json" "$OUT/DONE_$run.json" >/dev/null 2>&1; then
      # DONE 파싱 실패는 「남은 것 없음」이 아니다 (5차b 4번) — run 이름이 맞고 checkpoints 가 정수 목록일 때만 믿는다
      if ! want=$(python3 -c "
import json,sys; d=json.load(open('$OUT/DONE_$run.json'))
assert d.get('run')=='$run' and isinstance(d.get('checkpoints'),list) and all(isinstance(e,int) for e in d['checkpoints'])
print(' '.join(f'{e:02d}' for e in d['checkpoints'] if e >= $EPOCH_MIN))" 2>/dev/null); then
        log "DONE.json 파싱 실패 ($run) — 완료로 치지 않는다"; all_done=0
      else
        for nn in $want; do [ -f "$(scored_marker "$run" "$nn")" ] || { all_done=0; break; }; done
      fi
      # DONE 이 있는데 남은 체크포인트가 IDLE_MAX_MIN 동안 채점되지 않으면(마커 불일치·채점 실패 반복) 무한 대기하지 않는다
      if [ "$all_done" = "0" ] && [ $(( (SECONDS - last_progress) / 60 )) -ge "$IDLE_MAX_MIN" ]; then
        STOP_REASON="$run DONE 뒤 ${IDLE_MAX_MIN}분 진전 없음 — 남은 체크포인트를 채점하지 못했다 (사람이 본다)"; log "$STOP_REASON"; exit 1
      fi
    else
      all_done=0
      # 학습기 생존 — heartbeat 나이 (5차 5번). DONE 이 없는데 heartbeat 가 오래됐으면 학습기가 죽은 것
      age=$(s3_age_min "results_lora_main/$run/report/heartbeat")
      if [ -z "$age" ]; then
        if [ $(( (SECONDS - t_start) / 60 )) -ge "$START_MAX_MIN" ]; then STOP_REASON="$run heartbeat 가 ${START_MAX_MIN}분 안에 안 생김 — 학습기 없음"; log "$STOP_REASON"; exit 0; fi
      elif [ "$age" -ge "$IDLE_MAX_MIN" ]; then
        STOP_REASON="$run heartbeat ${age}분 전 — 학습기 사망으로 보고 끝낸다 (§14 18번)"; log "$STOP_REASON"; exit 0
      fi
    fi
  done
  if [ "$all_done" = "1" ]; then STOP_REASON="완료 — DONE.json 목록 전부 채점"; log "$STOP_REASON"; exit 0; fi
  log "대기 — ${POLL_SEC}초 뒤 다시 (heartbeat ${age:-?}분 전)"
  sleep "$POLL_SEC"
done
