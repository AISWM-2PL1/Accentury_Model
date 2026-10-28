#!/usr/bin/env bash
# 방언 5지역 저자원 A·B — 실험 하나를 「학습 → val loss 최저 어댑터 → 목표 지역 test CER·WER → S3 저장 확인」까지 돌리고
# 다음 실험으로 넘어가는 EC2 러너 (`docs/DIALECT5_LOW_RESOURCE_PARALLEL_TRAIN_PLAN_2026-10-04.md` §3~4)
#
# 쓰기 (EC2 안 · 보통 src/infra/launch_track2_gpu.sh start-dialect5-lowres 가 tmux 로 부른다)
#   src/finetune/run_dialect5_lowres.sh gyeongsang_B6p4h_GS-09                       # 이 실험 → 끝나면 대기 목록에서 다음 실험을 가져온다 → 없으면 끈다
#   QUEUE=0 src/finetune/run_dialect5_lowres.sh jeju_A3p2h_GS-09 jeju_A12p8h_GS-09   # 적어 준 실험만 차례로 돌리고 끈다
#   DRY_RUN=1 src/finetune/run_dialect5_lowres.sh <run_id> …                         # 설정을 풀어 찍기만 한다 (맥에서도 된다 · 아무것도 실행하지 않는다)
#   NO_SHUTDOWN=1 …                                                                  # 끝나도 끄지 않는다
#
# 설정  registry/finetune/dialect5_low_resource_runs_2026-10-04.json 한 곳 (make_dialect5_lowres_runs.py 가 만든다) — run_id 마다 학습 목록·val·test·결과 경로.
#       공통: GS-09 · 에포크 20% 검증 · 5회 연속 미개선 · 500 step 하한 · 스모크 없이 처음부터(direct).
# 실행  학습은 run_dialect5_smoke.sh, 채점은 run_dialect5_score.sh(SPLIT=test)를 NO_SHUTDOWN=1 로 그대로 부른다 — 업로드·DONE 규율은 그 둘의 것이다.
#       저장된 학습 상태(train_state.pt)가 로컬이나 S3 에 있으면 PHASE=full 로 재개한다. 학습이 끝난 실험(DONE exit 0 + best)은 채점부터 한다.
#       첫 검증 전에 죽어 상태가 없는 실험은 자동으로 다시 시작하지 않는다 (direct 는 S3 에 같은 RUN_ID 가 있으면 거부한다) — 사람이 본다.
# 데이터 지역 원본·중첩·B 묶음을 실험마다 aws s3 sync 로 맞춘다. 한 번 받은 파일은 다음 실험에서 다시 받지 않는다 (기본 모델은 HF 캐시).
# 대기열 QUEUE=1(기본)이면 설정의 queue.initial → queue.pending 순서대로 S3 조건부 쓰기(--if-none-match)로 실험을 하나씩 가져온다 — 두 서버가 같은 실험을 잡지 않는다.
#       표식: s3://…/results_dialect5/_lowres_queue/claims/<run_id>.json. 초기 배치도 목록에 넣는다 — 서버를 12대 다 못 켠 경우(2026-10-04 용량 부족) 켜진 서버가 이어받는다.
#       인자로 적어 준 실험도 표식을 먼저 잡는다. 이미 다른 서버가 잡았으면 건너뛰고 목록에서 다음 것을 가져온다. 중단된 실험을 다시 돌릴 때만 FORCE=1 로 표식을 무시한다.
# 디스크 JOB_DONE 까지 S3 에서 확인한 실험은 로컬 어댑터 폴더를 지운다 (KEEP_LOCAL_ADAPTERS=1 이면 남긴다) — 한 서버가 여러 실험을 이어 돌려도 디스크가 차지 않게.
#       지우는 것은 이 러너가 돌린 그 실험의 폴더뿐이다. 옛 실험의 어댑터·데이터 캐시는 건드리지 않는다.
# 실패  학습·채점·S3 대조 중 하나라도 실패하면 그 실험은 JOB_DONE 을 쓰지 않고, 이 서버는 다음 실험을 가져오지 않는다.
#       업로드 실패(UPLOAD_FAILED)면 끄지 않는다 (기존 러너와 같다). 그 밖의 실패는 끈다 — 디스크는 남는다.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ACCENTURY_DATA="${ACCENTURY_DATA:-$HOME/AccenturyData}"
CFG="${LOWRES_CONFIG:-registry/finetune/dialect5_low_resource_runs_2026-10-04.json}"
QUEUE="${QUEUE:-1}"; DRY_RUN="${DRY_RUN:-0}"; FREE_GB_MIN="${FREE_GB_MIN:-15}"   # B 묶음 7.4 GB + 실험 하나의 어댑터·상태 약 4 GB + 여유 (2026-10-04 실측: 서버 여유 25~27 GB)
BUCKET=accentury-track2-325771561913
CLAIMS="results_dialect5/_lowres_queue/claims"
HOST_ID="$(hostname)"
[ $# -ge 1 ] || { echo "run_id 가 하나 이상 필요하다" >&2; exit 2; }
[ -f "$CFG" ] || { echo "설정 파일이 없다: $CFG" >&2; exit 2; }
say() { echo "$(date -u +%FT%TZ) [lowres] $*"; }

# run_id → 셸 변수 (설정에 없으면 실패)
load_run() {
  python3 - "$CFG" "$1" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
run = next((r for r in cfg["runs"] if r["run_id"] == sys.argv[2]), None)
if run is None:
    sys.exit(f"설정에 없는 run_id: {sys.argv[2]}")
c, t, v, s = cfg["common"], run["train"], run["val"], run["test"]
out = {"REGION": run["region"], "COMBO": c["combo"], "PATIENCE": c["patience"],
       "TRAIN_S3": t["dataset_s3"], "TRAIN_ROOT": t["dataset_root"], "TRAIN_SET": t["train_set"], "TRAIN_SHA": t["manifest_sha256"],
       "VAL_S3": v["dataset_s3"], "VAL_ROOT": v["dataset_root"],
       "TEST_SHA": s["manifest_sha256"], "TEST_ROWS": s["rows"], "REMOTE": run["results_s3"]}
for k, val in out.items():
    print(f"{k}={shlex.quote(str(val))}")
print("EXTRA=(" + " ".join(shlex.quote(str(x)) for x in c["trainer_extra_args"]) + ")")
PY
}
for run in "$@"; do load_run "$run" >/dev/null || exit 2; done                      # 인자 오류는 아무것도 하기 전에 거른다
PENDING=()
while IFS= read -r line; do PENDING+=("$line"); done < <(python3 -c 'import json,sys; q=json.load(open(sys.argv[1], encoding="utf-8"))["queue"]; print("\n".join(q["initial"] + q["pending"]))' "$CFG")

# S3 조건부 쓰기로 실험을 잡는다. 반환: 0 잡음 · 1 이미 누가 잡음 · 2 그 밖의 실패
claim() {
  local key="$CLAIMS/$1.json" body err i
  body="$(mktemp)"; printf '{"run": "%s", "host": "%s", "at": "%s"}\n' "$1" "$HOST_ID" "$(date -u +%FT%TZ)" > "$body"
  for i in 1 2 3; do
    if err=$(aws s3api put-object --bucket "$BUCKET" --key "$key" --body "$body" --if-none-match '*' 2>&1 >/dev/null); then rm -f "$body"; return 0; fi
    case "$err" in
      *PreconditionFailed*) rm -f "$body"; return 1 ;;
      *ConditionalRequestConflict*) sleep $((i * 5)) ;;                              # 동시에 쓰는 중 — 잠깐 뒤 다시
      *) rm -f "$body"; say "claim 실패 $1: ${err:0:200}"; return 2 ;;
    esac
  done
  rm -f "$body"; return 1
}

# 로컬 파일과 S3 객체의 바이트가 같은지
same_bytes() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(aws s3 cp "$2" - 2>/dev/null | sha256sum | cut -d' ' -f1)" ]; }

KEEP_ON=0
run_one() {
  local run="$1" vars phase rc step adapter out
  vars="$(load_run "$run")" || return 2
  eval "$vars"
  adapter="$ACCENTURY_DATA/work/finetune/dialect5_lora/$run"; out="registry/finetune/dialect5_lora/$run"
  if [ "$DRY_RUN" = 1 ]; then
    echo "== $run"
    echo "  학습  NO_SHUTDOWN=1 PHASE=direct FULL_PATIENCE=$PATIENCE RUN_ID=$run TRAIN_SET=$TRAIN_SET"
    echo "        DATASET_S3=$TRAIN_S3"
    echo "        DATASET_ROOT=\$ACCENTURY_DATA/$TRAIN_ROOT"
    echo "        VAL_DATASET_S3=${VAL_S3:-(train 과 같은 완성본)} VAL_DATASET_ROOT=${VAL_ROOT:+\$ACCENTURY_DATA/$VAL_ROOT}"
    echo "        src/finetune/run_dialect5_smoke.sh $REGION $COMBO ${EXTRA[*]}"
    echo "  채점  NO_SHUTDOWN=1 SPLIT=test RUN_ID=$run src/finetune/run_dialect5_score.sh $REGION $COMBO <best step>   (test $TEST_ROWS 행)"
    echo "  저장  $REMOTE/{adapters,tb,report}/ · 로컬 $adapter · $out"
    return 0
  fi
  if aws s3 ls "$REMOTE/report/JOB_DONE.json" >/dev/null 2>&1; then say "$run: JOB_DONE 이 이미 있다 — 건너뛴다"; return 0; fi
  say "$run 시작 — train $TRAIN_SET · 디스크 여유 $(df -BG --output=avail "$ACCENTURY_DATA" | tail -1 | tr -dc 0-9) GB"
  mkdir -p "$out"

  # ---- 학습 (이미 끝난 학습이면 건너뛴다)
  phase=""
  if aws s3 cp "$REMOTE/report/DONE.json" - 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("exit") == 0 and d.get("best") and d.get("phase") in ("direct", "full") else 1)' 2>/dev/null; then
    say "$run: 학습은 끝나 있다 (DONE exit 0) — 채점부터"
    aws s3 cp "$REMOTE/adapters/best.json" "$adapter/best.json" >/dev/null
    aws s3 sync "$REMOTE/report/" "$out/" --exclude "DONE.json" --exclude "heartbeat" >/dev/null      # 다른 서버에서 학습했으면 학습 보고가 로컬에 없다
  else
    phase=direct
    if [ -f "$adapter/train_state.pt" ] || aws s3 ls "$REMOTE/adapters/train_state.pt" >/dev/null 2>&1; then phase=full; say "$run: 저장된 학습 상태에서 재개 (PHASE=full)"; fi
    set +e
    NO_SHUTDOWN=1 PHASE="$phase" FULL_PATIENCE="$PATIENCE" RUN_ID="$run" TRAIN_SET="$TRAIN_SET" \
      DATASET_S3="$TRAIN_S3" DATASET_ROOT="$ACCENTURY_DATA/$TRAIN_ROOT" \
      VAL_DATASET_S3="$VAL_S3" VAL_DATASET_ROOT="${VAL_ROOT:+$ACCENTURY_DATA/$VAL_ROOT}" \
      src/finetune/run_dialect5_smoke.sh "$REGION" "$COMBO" "${EXTRA[@]}"
    rc=$?
    set -e
    if [ "$rc" != 0 ]; then
      tail -n 8 "$out/summary.txt" 2>/dev/null | grep -q UPLOAD_FAILED && KEEP_ON=1
      say "$run: 학습 실패 (exit $rc) — 채점하지 않는다"; return 1
    fi
  fi
  step="$(python3 -c 'import json,sys,pathlib; print(pathlib.Path(json.load(open(sys.argv[1], encoding="utf-8"))["경로"]).name)' "$adapter/best.json")" || { say "$run: best.json 을 못 읽었다"; return 1; }
  case "$step" in step[0-9][0-9][0-9][0-9][0-9][0-9]) ;; *) say "$run: best 가 step 폴더가 아니다: $step"; return 1 ;; esac

  # ---- 목표 지역 test 채점 (CER·WER 는 한 번의 받아쓰기에서 같이 나온다)
  set +e
  NO_SHUTDOWN=1 SPLIT=test RUN_ID="$run" src/finetune/run_dialect5_score.sh "$REGION" "$COMBO" "$step"
  rc=$?
  set -e
  [ "$rc" = 0 ] || { say "$run: test 채점 실패 (exit $rc)"; return 1; }

  # ---- 내용 대조 + S3 저장 확인 → JOB_DONE.json
  python3 - "$CFG" "$run" "$out" "$adapter" "$step" "$HOST_ID" > "$out/JOB_DONE.json.tmp" <<'PY' || { say "$run: 결과 대조 실패 — JOB_DONE 을 쓰지 않는다"; return 1; }
import datetime, glob, json, os, sys
cfg_path, run_id, out, adapter, step, host = sys.argv[1:7]
cfg = json.load(open(cfg_path, encoding="utf-8"))
run = next(r for r in cfg["runs"] if r["run_id"] == run_id); c = cfg["common"]
reports = sorted(glob.glob(f"{out}/lora_{run_id}_*.json"), key=os.path.getmtime)
if not reports:
    sys.exit("학습 보고가 없다")
rep = json.load(open(reports[-1], encoding="utf-8")); s = rep["설정"]; done = rep.get("학습") or {}
test = json.load(open(f"{out}/test_cer_{step}.json", encoding="utf-8"))
best = json.load(open(f"{adapter}/best.json", encoding="utf-8"))
want = {"train manifest sha256": (rep["manifest_sha256"], run["train"]["manifest_sha256"]),
        "val 발화": (s["val_발화"], run["val"]["trainer_rows"]),
        "patience": (s["val_loss_patience"], c["patience"]), "하한 step": (s["val_loss_min_steps"], c["min_stop_steps"]),
        "검증 간격 비율": (s["eval_every_frac"], c["eval_every_frac"]), "고정 step 간격 미사용": (s["eval_every_steps"] <= 0, True),
        "test manifest sha256": (test["manifest_sha256"], run["test"]["manifest_sha256"]), "test 발화": (test["발화"], run["test"]["rows"]),
        "최종 어댑터 = val loss 최저": (os.path.basename(done.get("최종_어댑터") or ""), step),
        "채점한 어댑터": (os.path.basename((test.get("어댑터") or {}).get("path", "")), step)}
bad = {k: v for k, v in want.items() if v[0] != v[1]}
if bad:
    sys.exit(f"기대와 다르다 (실제, 기대): {bad}")
print(json.dumps({"run": run_id, "host": host, "best_step": step, "best_val_loss": best["val_loss"], "best_epoch": best["epoch"],
                  "global_step": done.get("global_step"), "종료_이유": done.get("종료_이유"), "벽시계_h": done.get("벽시계_h"),
                  "학습_보고": os.path.basename(reports[-1]), "train_manifest_sha256": rep["manifest_sha256"],
                  "test": {"파일": f"test_cer_{step}.json", "발화": test["발화"], "실패": test["실패"], "manifest_sha256": test["manifest_sha256"],
                           "어댑터_sha256": test["어댑터"]["sha256"]},
                  "주의": "test 요약은 제외 규칙 적용 전이다 — 충청은 집계 단계에서 DCDG20003365 를 뺀다 (dialect5_exclusions.py)",
                  "대조": sorted(want), "done": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
PY
  local f
  for f in "test_cer_$step.json" "test_cer_$step.csv" "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["학습_보고"])' "$out/JOB_DONE.json.tmp")"; do
    same_bytes "$out/$f" "$REMOTE/report/$f" || { say "$run: S3 의 $f 가 로컬과 다르다"; KEEP_ON=1; return 1; }
  done
  [ "$(aws s3api head-object --bucket "$BUCKET" --key "${REMOTE#s3://$BUCKET/}/adapters/$step/adapter_model.safetensors" --query ContentLength --output text 2>/dev/null)" = "$(stat -c %s "$adapter/$step/adapter_model.safetensors")" ] \
    || { say "$run: S3 의 best 어댑터($step) 크기가 로컬과 다르다"; KEEP_ON=1; return 1; }
  mv -f "$out/JOB_DONE.json.tmp" "$out/JOB_DONE.json"
  aws s3 cp "$out/JOB_DONE.json" "$REMOTE/report/JOB_DONE.json" >/dev/null && same_bytes "$out/JOB_DONE.json" "$REMOTE/report/JOB_DONE.json" \
    || { say "$run: JOB_DONE 업로드 실패"; KEEP_ON=1; return 1; }
  if [ "${KEEP_LOCAL_ADAPTERS:-0}" != 1 ]; then
    case "$adapter" in "$ACCENTURY_DATA/work/finetune/dialect5_lora/$run") rm -rf "$adapter" && say "$run: 로컬 어댑터 폴더 정리 (S3 에 있다)" ;; esac
  fi
  say "$run 끝 — best $step · test 채점·S3 저장 확인 완료"
}

if [ "$DRY_RUN" = 1 ]; then
  for run in "$@"; do run_one "$run"; done
  echo "대기 목록(QUEUE=$QUEUE): ${PENDING[*]}"
  exit 0
fi

# ---- 시작 전 확인 (여기서 실패하면 끄지 않고 바로 알린다)
free_gb="$(df -BG --output=avail "$ACCENTURY_DATA" 2>/dev/null | tail -1 | tr -dc 0-9)"
[ "${free_gb:-0}" -ge "$FREE_GB_MIN" ] || { echo "디스크 여유 ${free_gb:-?} GB < $FREE_GB_MIN GB — 옛 어댑터를 정리한 뒤 다시 시작한다" >&2; exit 2; }
probe_body="$(mktemp)"; echo '{}' > "$probe_body"
perr=$(aws s3api put-object --bucket "$BUCKET" --key "$CLAIMS/_probe_$HOST_ID.json" --body "$probe_body" --if-none-match '*' 2>&1 >/dev/null) || case "$perr" in
  *PreconditionFailed*) ;;                                                           # 이미 있는 표식 — 조건부 쓰기가 동작한다는 뜻이다
  *) echo "S3 조건부 쓰기(--if-none-match)가 안 된다 — aws cli 판을 확인한다: ${perr:0:200}" >&2; exit 2 ;;
esac
rm -f "$probe_body"

status=0
finish() {
  rc=$?; set +e
  [ "$rc" = 0 ] && rc="$status"
  say "=== 끝 (exit $rc · KEEP_ON $KEEP_ON) ==="
  if [ "${NO_SHUTDOWN:-0}" = 1 ]; then say "NO_SHUTDOWN=1 — 끄지 않는다"; exit "$rc"; fi
  if [ "$KEEP_ON" = 1 ]; then say "업로드·저장 확인 실패 — 끄지 않는다. 사람이 S3 로 올린 뒤 끈다"; exit "$rc"; fi
  sudo shutdown -h now
  exit "$rc"
}
trap finish EXIT

for run in "$@"; do
  c=0; claim "$run" || c=$?
  [ "$c" != 2 ] || { status=3; exit 3; }
  if [ "$c" != 0 ]; then
    if [ "${FORCE:-0}" = 1 ]; then say "$run: 표식이 이미 있다 — FORCE=1 이라 그대로 돌린다"
    else say "$run: 다른 서버가 이미 잡았다 — 건너뛰고 대기 목록에서 가져온다"; continue; fi
  fi
  run_one "$run" || { status=1; exit 1; }
done
if [ "$QUEUE" = 1 ]; then
  while :; do
    next=""
    for cand in "${PENDING[@]}"; do
      c=0; claim "$cand" || c=$?
      [ "$c" != 2 ] || { status=3; exit 3; }
      if [ "$c" = 0 ]; then next="$cand"; break; fi
    done
    [ -n "$next" ] || { say "대기 목록에 남은 실험이 없다"; break; }
    say "대기 목록에서 가져옴: $next"
    run_one "$next" || { status=1; exit 1; }
  done
fi
exit 0
