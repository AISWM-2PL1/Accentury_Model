#!/usr/bin/env python3
"""트랙 2 val 세트 채점 · 체크포인트 선택 — `docs/TRACK2_BENCHMARK_V2_SPEC.md` §4.3

  왜  epoch 는 논문에서 빌릴 수 없다 (§4.3 「왜 필요한가」). 시험지로 고르면 시험지가 성적표 구실을
      못하므로 별도 val 8세트로 고른다. 이 스크립트가 그 채점과 §4.3 의 3단 선택 규칙을 돌린다.
  뭘  ① 채점 모드 — val 8세트를 채점해 세트별 CER·WER·CI 와 판정 묶음(고령 3칸 합산 micro ·
                    경상 10~40대 · 서울 3칸)을 낸다. 모델(체크포인트)마다 registry JSON 한 장.
      ② 선택 모드(--select) — 무튜닝 baseline 과 체크포인트 여럿을 함께 받아 §4.3 의
                    탈락 A / 경고 B / 선택 · 조기 종료 · 동률 처리를 적용하고 고른 근거를 적는다.
                    규칙 7(2026-09-09 보험): 고령 3칸 **방언 자리 정답률**을 표에 병기하고 경고 C 를 낸다 —
                    CER 하락이 간투사·표기 관례 학습일 수 있어서다. 결정은 여전히 CER 이 한다.
  입력  --model <라벨>=<조각 폴더>   run_track2_stt.py 산출 (`A__v2val_*.csv.gz` + `_run_meta.json`)
        --sets-dir                 기본 `$ACCENTURY_DATA/work/finetune/track2_bench_v2_val/sets`
        registry/benchmark/track2_bench_v2_val_sets.json   (세트 정본 — 발화 수 대조용)

  val 추론 배선 (2026-09-03 뚫음)
     ~~`run_track2_stt.py --bench track2_bench_v2_val` 는 죽는다~~ — `track2_data.set_bench` 가
     `work/benchmark` 와 `work/finetune` 두 자리를 찾도록 고쳤다(val 세트는 후자에 있다).
     어댑터는 `run_track2_stt.py --adapter <폴더>` 로 얹는다 — `track2_asr.build(device, adapter)`
     가 peft 로 붙이고 **병합하지 않으며**, sha256 을 조각 메타에 남겨 무튜닝 조각과 섞이지 않게 한다.
     학습 중 자동 호출은 `src/finetune/val_hook.py` 가 맡는다 (추론 → 이 채점기 → `val_score.json`).
  출력  registry/benchmark/track2_bench_v2_val_stt__<라벨>.json  (모델마다)
        registry/benchmark/track2_bench_v2_val_select__<run>.json (--select)
        표는 stdout
  주의
    - **이 세트로 성능을 보고하지 않는다** (registry `용도`). 체크포인트 선택 전용이다.
    - 서울 3칸은 화자가 3/2/2 명뿐이라 §4.3 3번대로 **경고**로만 쓴다. 참고선·성능으로 인용 금지.
    - 채점 규칙은 `score_track2.score_combo` 를 그대로 쓴다. CI 는 둘을 병기한다 —
      `ci_group`(재표집 `group_id` · 2000회 · seed 0)이 val 의 정본이고(§4.3 한계: 같은 화자의
      여러 대화를 독립으로 보면 안 된다), `ci_dialogue`(score_combo · 대화 400회 · seed 20260826)는
      시험지 표와의 연속성을 위해 병기한다. 경상 60·70·80대+ 는 대화 151·385·338 개 대 group
      10·31·21 개라 두 값이 크게 다르다.
    - 짝지은 Δ 와 CI 는 `paired_delta_track2.py` 가 낸다. 탈락·경고는 그 CI 하한 > 0 으로 판정한다 — 마진 τ 는 없다.
    - GPU 를 쓰지 않는다 · 추론을 돌리지 않는다.

  ┌ 닫힌 미결 (2026-09-09 박재영) ─────────────────────────────────────────────────────────┐
  │ 미결 1·2(τ 를 어느 체크포인트에서 뽑나 · 폭인가 반폭인가) · 미결 3(τ 적용 범위) · 미결 4(조기 종료 │
  │ 적용)는 **τ 와 조기 종료를 없애며 함께 닫혔다.** 1차 실측: τ(ep02 Δ CI 폭 1.95)가 epoch 간 개선폭  │
  │ (0.2~1.1)보다 커서 옛 규칙은 w_ep02(CER 17.10 · 방언 자리 34%)를 골랐고, 조기 종료를 빼도 τ 동률이  │
  │ ep04(폭)·ep06(반폭)을 골랐다. 실제 최저는 ep14(14.59 · 46%). 조기 종료를 안 하면 τ 는 동률에도 필요 │
  │ 없고, 「나빠졌는가」는 이미 계산하는 Δ CI 로 판정하면 된다. 옛 인자 --tau·--tau-scope·--tau-rule·   │
  │ --tau-from·--early-stop 은 제거했다. 옛 결과 정본은 2026-09-09 오전 이전 판이다.                   │
  └──────────────────────────────────────────────────────────────────────────────────┘

  # 무튜닝 채점 (τ 를 정하기 전 단계 — 절대 CER 과 CI 만 나온다)
  python3 src/stt/score_val_track2.py --model baseline=registry/benchmark/track2_stt_bench_v2_val

  # 체크포인트 선택
  python3 src/stt/score_val_track2.py --select --run-label weighted --baseline baseline \\
      --model baseline=registry/benchmark/track2_stt_bench_v2_val \\
      --model ep02=registry/benchmark/track2_stt_bench_v2_val_w_ep02 \\
      --model ep04=registry/benchmark/track2_stt_bench_v2_val_w_ep04
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.score_track2 import check_meta, fmt, score_combo                   # 채점 규칙 복제 금지
from stt.paired_delta_track2 import (CLUSTER_COL, ITERS, SEED, V, boot_ci_group,
                                     cluster_map, list_sets, macro_cer, micro_cer, micro_wer,
                                     paired_delta, read_checked, read_manifest, score_rows,
                                     sets_root)

ROOT = Path(__file__).resolve().parents[2]
BENCH = "track2_bench_v2_val"
SETS_JSON = ROOT / f"registry/benchmark/{BENCH}_sets.json"
OUT_DIR = ROOT / "registry/benchmark"
SHORT = 2.0
# group 이 이보다 적으면 클러스터 재표집이 만들 수 있는 서로 다른 표본이 몇 가지뿐이다
# (2개면 3가지). §4.3 한계 「화자 묶음이 적다 — 서울은 2~3개」. 문턱은 이 스크립트가 정한 경고선일 뿐
# 명세에 있는 값이 아니다.
MIN_GROUP = 5

GY = "v2val_gyeongsang_"
SE = "v2val_seoul_"
EXPECTED = [f"{GY}10to40s", f"{GY}50s", f"{GY}60s", f"{GY}70s", f"{GY}80s_plus",
            f"{SE}10to40s", f"{SE}50s", f"{SE}60s_plus"]

# 판정 묶음 — §4.3 「체크포인트 선택 규칙」 2·3·4번. 역할을 코드에 적어 규칙과 표를 잇는다.
BUNDLES = {
    "고령_3칸":     {"세트": [f"{GY}60s", f"{GY}70s", f"{GY}80s_plus"], "역할": "선택(4번) 지표 · 경고 C(7번) 병기"},
    "경상_10to40s": {"세트": [f"{GY}10to40s"], "역할": "탈락 A — hard veto (2번)"},
    "경상_50s":     {"세트": [f"{GY}50s"], "역할": "규칙에 없다 — 병기만"},
    "서울_10to40s": {"세트": [f"{SE}10to40s"], "역할": "경고 B (3번)"},
    "서울_50s":     {"세트": [f"{SE}50s"], "역할": "경고 B (3번)"},
    "서울_60s_plus": {"세트": [f"{SE}60s_plus"], "역할": "경고 B (3번)"},
}
VETO_BUNDLE = "경상_10to40s"
WARN_BUNDLES = ["서울_10to40s", "서울_50s", "서울_60s_plus"]
PICK_BUNDLE = "고령_3칸"

# 규칙 7 — 경고 C (보험 · 박재영 결정 2026-09-09). 고령 3칸 **방언 자리 정답률**(방언층≠표준층 어절 중 출력이
# 방언층과 같은 비율 · `dialect_pos_val_track2.tally` · 공백 무시 정렬)을 CER 옆에 병기한다. 선택(4번)·조기 종료(5번)는
# 그대로 CER 이 정한다. 고른 체크포인트의 정답률이 후보 최고보다 DACC_WARN_PT 이상 낮거나 무튜닝 이하면 **표시만** 한다.
# 왜: CER 하락은 간투사·표기 관례 학습으로도 생긴다(`track2_bench_v2_val_convention.json` — 서울 개선의 52%가 간투사).
# CER 하나로 고르면 관례를 더 외운 쪽을 고를 수 있다. 문턱 1.0pt 는 이 스크립트가 정한 경고선일 뿐이다 — 정답률에는 CI 가 없다.
DACC_WARN_PT = 1.0

NOTE_USE = ("이 세트로 성능을 보고하지 않는다 — 체크포인트 선택 전용이다 "
            "(registry/benchmark/track2_bench_v2_val_sets.json 「용도」).")
NOTE_SEOUL = ("서울 3칸은 화자 3/2/2 명·대화 9/6/6 개다. §4.3 3번대로 경고로만 쓰고 참고선·성능으로 "
              "인용하지 않는다. 서울(NIKL)과 경상(AI Hub)은 말뭉치가 달라 절대값을 직접 견주지 않는다.")
NOTE_PREC = ("val 1h/칸의 절대 CER CI 는 ±1.49 로 추정된 값이다 (§4.3 한계 — 추정이며 실측 아님). "
             "체크포인트 비교는 paired Δ 라 이보다 정밀하다 — 실측은 paired_delta_track2.py 가 낸다.")
NOTE_OVERLAP = ("②겹침 = ② 교재(work/finetune/track2_tempbench_v1/sets)와 겹치는 val 발화 505개. "
                "무튜닝·v2 교재로 학습한 모델에는 무해하다 (v2 교재와의 발화 겹침은 0). "
                "② 산출물과 비교할 때만 「②겹침 제외」 열을 본다 (§4.3 한계).")


# ── 인자 ─────────────────────────────────────────────────
def kv(s: str, what: str) -> tuple:
    if "=" not in s:
        sys.exit(f"--{what} 는 <라벨>=<값> 형식이다: {s}")
    k, v = s.split("=", 1)
    if not k or not v:
        sys.exit(f"--{what} 의 라벨·값이 비었다: {s}")
    return k, v


SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
EPOCH_RE = re.compile(r"(?:^|[^0-9a-zA-Z])(?:ep|epoch)[_-]?(\d+)", re.I)


def epoch_of(label: str, given: dict):
    """체크포인트의 epoch. 규칙 4 의 동률 처리(「epoch 가 작은 쪽」)와 5 의 연속 판정에 쓴다.
    라벨에서 `ep12`·`epoch_12` 를 읽고, 없으면 --epoch 로 받는다. 둘 다 없으면 멈춘다 —
    epoch 를 모르면 §4.3 4·5번을 적용할 수 없다."""
    if label in given:
        return given[label]
    m = EPOCH_RE.search(label)
    if m:
        return int(m.group(1))
    sys.exit(f"체크포인트 '{label}' 의 epoch 를 못 읽었다 — 라벨에 ep<N> 을 넣거나 --epoch {label}=<N> 을 준다")


# ── 채점 ─────────────────────────────────────────────────
def pack_set(rows, attr, sc, cluster, uids_overlap2, iters, seed) -> dict:
    """한 세트의 채점 묶음. 방언 정답은 CI 포함, 표준어 정답은 병기 (score_bench_v2 와 같은 모양)."""
    d = score_combo(rows, "ref_dialect", want_ci=True)      # 대화 단위 CI (병기)
    s = score_combo(rows, "ref_standard", want_ci=False)
    ok = [r["utterance_id"] for r in rows if r["utterance_id"] in sc]
    dur = {u: float(attr[u]["duration_label_sec"]) for u in attr}
    long_u = [u for u in ok if dur[u] >= SHORT]
    male = [u for u in ok if attr[u]["gender"] == "m"]
    female = [u for u in ok if attr[u]["gender"] == "f"]
    no2 = [u for u in ok if u not in uids_overlap2]
    return {
        "발화": len(attr), "채점가능": len(ok),
        "시간_h": round(sum(dur.values()) / 3600, 3),
        "group": len({attr[u][CLUSTER_COL] for u in attr}),
        "대화": len({attr[u]["dialogue_id"] for u in attr}),
        "cer": d["cer_micro"], "wer": d.get("wer_micro"),
        "cer_macro": d.get("cer_macro"), "ser": d.get("ser"),
        "ci_group": boot_ci_group(sc, ok, cluster, iters=iters, seed=seed),
        "ci_dialogue": d.get("ci"),
        "cer_2초이상": round(micro_cer(sc, long_u), 2) if long_u else None,
        "cer_남": round(micro_cer(sc, male), 2) if male else None,
        "cer_여": round(micro_cer(sc, female), 2) if female else None,
        "표준어_cer": s["cer_micro"],
        "②겹침_발화": len(ok) - len(no2),
        "cer_②겹침제외": round(micro_cer(sc, no2), 2) if no2 else None,
        "cer_no_outlier": d.get("cer_micro_no_outlier"), "n_outlier": d.get("n_outlier"),
        "failure_rate": d.get("failure_rate"), "scoreable": d.get("scoreable"),
        "2초미만_pct": round(100 * sum(dur[u] < SHORT for u in attr) / max(1, len(attr)), 1),
        "방언_집계": d, "표준어_집계": s,
    }


def pack_bundle(name, sets, attrs, sc, cluster, uids_overlap2, iters, seed) -> dict:
    """판정 묶음(여러 세트 합산). **칸별 평균이 아니라 글자 가중 micro** 다 (§4.3 4번)."""
    uids = [u for s in sets for u in attrs[s] if u in sc]
    no2 = [u for u in uids if u not in uids_overlap2]
    dur = sum(float(attrs[s][u]["duration_label_sec"]) for s in sets for u in attrs[s])
    return {
        "역할": BUNDLES[name]["역할"], "세트": sets,
        "발화": sum(len(attrs[s]) for s in sets), "채점가능": len(uids),
        "시간_h": round(dur / 3600, 3),
        "group": len({attrs[s][u][CLUSTER_COL] for s in sets for u in attrs[s]}),
        "cer": round(micro_cer(sc, uids), 2) if uids else None,
        "wer": round(micro_wer(sc, uids), 2) if uids else None,
        "cer_macro": round(macro_cer(sc, uids), 2) if uids else None,
        "ci_group": boot_ci_group(sc, uids, cluster, iters=iters, seed=seed),
        "②겹침_발화": len(uids) - len(no2),
        "cer_②겹침제외": round(micro_cer(sc, no2), 2) if no2 else None,
    }


def score_model(label, shard_dir, sets, attrs, cond, cluster, uids_overlap2, iters, seed) -> dict:
    """한 모델(무튜닝 또는 체크포인트)의 val 채점 전체."""
    shard_dir = Path(shard_dir)
    if not shard_dir.is_absolute():
        shard_dir = ROOT / shard_dir
    shards = [f"{cond}__{s}" for s in sets]
    sh, devices, dtypes, warn = check_meta(shard_dir, shards)     # 코덱스 — 메타 없는 조각으로 채점하지 않는다
    per_set, sc = {}, {}
    for s in sets:
        rows = read_checked(shard_dir, s, cond, attrs[s])         # 조각↔manifest 대조 (코덱스)
        part = score_rows(rows, "ref_dialect")
        sc.update(part)
        per_set[s] = (rows, part)
    out_sets = {}
    for s in sets:
        rows, _ = per_set[s]
        out_sets[s] = pack_set(rows, attrs[s], sc, cluster, uids_overlap2, iters, seed)
        p = out_sets[s]
        if not p["scoreable"]:
            warn.append(f"🚨 {label} · {s}: 채점 가능 행 0")
        if p["failure_rate"]:
            warn.append(f"{label} · {s}: 실패율 {p['failure_rate']}%")
        if p["n_outlier"]:
            warn.append(f"{label} · {s}: CER>100% {p['n_outlier']}건")
    bundles = {n: pack_bundle(n, [x for x in b["세트"] if x in sets], attrs, sc, cluster,
                              uids_overlap2, iters, seed)
               for n, b in BUNDLES.items() if any(x in sets for x in b["세트"])}
    for n, b in bundles.items():
        if b["group"] < MIN_GROUP:
            # 모델이 아니라 세트의 성질이다 — 라벨을 붙이지 않아야 모델 수만큼 되풀이되지 않는다
            warn.append(f"{n}: group {b['group']}개 — group 클러스터 재표집이 만들 수 있는 서로 다른 표본이 "
                        f"몇 가지뿐이라 CI·τ 를 믿을 수 없다 (§4.3 한계 「화자 묶음이 적다」)")
    meta = {"model": sorted({sh[k]["model"] for k in shards}), "device": devices, "dtype": dtypes,
            "n_shards": len(shards), "cond": cond,
            "in_dir": str(shard_dir.relative_to(ROOT)) if shard_dir.is_relative_to(ROOT) else str(shard_dir)}
    return {"라벨": label, "meta": meta, "세트": out_sets, "묶음": bundles, "경고": warn, "_scored": sc}


# ── 표 ───────────────────────────────────────────────────
def ci_str(c):
    return "—" if not c else f"{c['lo']:.2f}~{c['hi']:.2f}"


def dedup(xs) -> list:
    """같은 경고를 모델 수만큼 되풀이하지 않는다. 순서는 유지한다."""
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def set_table(m) -> str:
    lines = ["| 세트 | 방언 CER | CI(group) | CI(대화) | WER | 2초 이상 | 남 / 여 | 표준어 CER | 발화 | group | 대화 | 시간h | ②겹침 제외 |",
             "|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for s, p in m["세트"].items():
        lines.append(f"| {s} | {fmt(p['cer'])} | {ci_str(p['ci_group'])} | {ci_str(p['ci_dialogue'])} | "
                     f"{fmt(p['wer'])} | {fmt(p['cer_2초이상'])} | {fmt(p['cer_남'])} / {fmt(p['cer_여'])} | "
                     f"{fmt(p['표준어_cer'])} | {p['발화']:,} | {p['group']} | {p['대화']} | "
                     f"{p['시간_h']:.2f} | {fmt(p['cer_②겹침제외'])} |")
    return "\n".join(lines)


def bundle_table(m) -> str:
    lines = ["| 판정 묶음 | 역할 | micro CER | CI(group) | WER | macro | 발화 | group | 시간h | ②겹침 제외 |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for n, b in m["묶음"].items():
        lines.append(f"| {n} | {b['역할']} | {fmt(b['cer'])} | {ci_str(b['ci_group'])} | {fmt(b['wer'])} | "
                     f"{fmt(b['cer_macro'])} | {b['발화']:,} | {b['group']} | {b['시간_h']:.2f} | "
                     f"{fmt(b['cer_②겹침제외'])} |")
    return "\n".join(lines)


# ── §4.3 선택 규칙 ────────────────────────────────────────
def deltas_for(base, ck, cluster, attrs, sets, iters, seed) -> dict:
    """판정 묶음마다 짝지은 Δ(체크포인트 − 무튜닝)와 CI."""
    out = {}
    for n, b in BUNDLES.items():
        uids = {u for s in b["세트"] if s in sets for u in attrs[s]}
        if not uids:
            continue
        out[n] = paired_delta(base["_scored"], ck["_scored"], cluster, uids=uids,
                              iters=iters, seed=seed, label_a=base["라벨"], label_b=ck["라벨"])
    return out


def dialect_acc(shard_dir, cond: str) -> dict:
    """규칙 7 병기값 — 고령 3칸 합산 방언 자리 정답률 %. 규칙은 dialect_pos_val_track2.tally 를 그대로 쓴다 (복제 금지).
    지연 import — 그 모듈이 이 파일의 BUNDLES 를 import 하므로 위에서 부르면 순환한다."""
    from stt.dialect_pos_val_track2 import tally
    d = Path(shard_dir) if Path(shard_dir).is_absolute() else ROOT / shard_dir
    hit = tot = 0
    for s in BUNDLES[PICK_BUNDLE]["세트"]:
        shard = d / f"{cond}__{s}.csv.gz"
        if not shard.exists():
            return {"정답률": None, "방언_어절": 0, "정답": 0, "경고": f"조각 없음: {shard}"}
        b, _ = tally(shard, cond)
        hit += b["정답"]; tot += b["방언_어절"]
    return {"정답률": round(100 * hit / tot, 2) if tot else None, "방언_어절": tot, "정답": hit}


def choose(base, cks, epochs, deltas, dacc=None) -> dict:
    """§4.3 「체크포인트 선택 규칙」 2~7번 (2026-09-09 개정판). 결정적으로 적는다 — 같은 입력이면 같은 답이 나온다.

    두 종류의 수를 섞어 쓴다. 규칙 4 는 **그 모델 자신의 절대 micro CER**(「고령 3칸을 합친
    micro CER 최저」)을 보고, 규칙 2·3 은 **무튜닝과의 짝지은 Δ 의 CI** 를 본다. 모델 실패(status≠PASS)
    가 없으면 두 집합이 같아 값도 같지만, 한쪽만 실패한 발화가 생기면 Δ 는 교집합, 절대값은
    각자 채점 가능한 전체가 되어 미세하게 어긋난다. 그 건수는 Δ 의 `n_a_only`·`n_b_only` 에 남는다.

    2026-09-09 개정 (박재영): 마진 τ 와 조기 종료(옛 5번)·τ 동률(옛 4번)을 없앴다. 1차 실측에서 τ(1.95)가
    epoch 간 개선폭(0.2~1.1)보다 커서 ep02 를 골랐다 — 실제 최저는 ep14. 나빠졌는가는 Δ CI 하한 > 0 으로 판정한다.
    """
    order = sorted(cks, key=lambda l: (epochs[l], list(cks).index(l)))   # epoch → 준 순서(저장 순서)
    verdict, reasons = {}, []

    # 2·3번 — 탈락 A / 경고 B: 짝지은 Δ(체크포인트 − 무튜닝)의 CI 하한이 0 보다 크면 「유의하게 나빠졌다」
    def worse(dn) -> bool:
        return bool(dn) and dn.get("lo") is not None and dn["lo"] > 0
    def comparable(l) -> list:
        """필수 비교(탈락 판정 묶음 · 선택 묶음)의 Δ CI 가 없으면(공통 채점 발화 0) 비교 불가 — 통과로 치지 않는다 (코덱스 5차 6번)."""
        return [n for n in (VETO_BUNDLE, PICK_BUNDLE) if not deltas[l].get(n) or deltas[l][n].get("lo") is None]
    for l in order:
        d = deltas[l]
        v = {"epoch": epochs[l], "탈락": [], "경고": []}
        miss = comparable(l)
        if miss:
            v["탈락"].append(f"비교 불가 — {'·'.join(miss)} 의 paired Δ CI 없음 (공통 채점 발화 0 또는 채점 누락) → 선택 보류 (코덱스 5차 6번)")
        dv = d.get(VETO_BUNDLE)
        if worse(dv):
            v["탈락"].append(f"{VETO_BUNDLE} Δ {dv['delta']:+.2f} (CI {dv['lo']:.2f}~{dv['hi']:.2f}) 하한 > 0 (규칙 2 hard veto)")
        hit = []
        for n in WARN_BUNDLES:
            dn = deltas[l].get(n)
            if worse(dn):
                hit.append(n)
                v["경고"].append(f"{n} Δ {dn['delta']:+.2f} (CI {dn['lo']:.2f}~{dn['hi']:.2f}) 하한 > 0 (규칙 3 경고 B)")
        if hit and len(hit) == len([n for n in WARN_BUNDLES if n in deltas[l]]):
            v["탈락"].append(f"서울 {len(hit)}칸이 모두 유의하게 나빠짐 — 규칙 3 「세 칸이 모두 그러면 그때는 탈락」")
        verdict[l] = v

    # (옛 5번 조기 종료는 2026-09-09 삭제 — 후보는 언제나 전부) · 참고용 추적만 남긴다
    best, trace = None, []
    for l in order:
        cur = cks[l]["묶음"][PICK_BUNDLE]["cer"]
        gain = None if (best is None or cur is None) else round(best - cur, 2)
        if best is None or (cur is not None and cur < best):
            best = cur
        trace.append({"라벨": l, "epoch": epochs[l], "고령_micro": cur, "직전_최저_대비_개선": gain})
    pool = list(order)

    # 4번 — 선택
    alive = [l for l in pool if not verdict[l]["탈락"]]
    if not alive:
        return {"고름": None, "실패": True,
                "근거": [f"후보 {len(pool)}개가 모두 탈락 — §4.3 6번대로 파인튜닝 1차를 실패로 보고한다. "
                       "무튜닝(epoch 0)을 고르지 않는다."],
                "판정": verdict, "추적": trace, "순서": order}
    scored = [(l, cks[l]["묶음"][PICK_BUNDLE]["cer"]) for l in alive]
    scored = [(l, c) for l, c in scored if c is not None]
    if not scored:
        return {"고름": None, "실패": True, "근거": ["고령 3칸 micro CER 이 없다 — 채점부터 다시 본다"],
                "판정": verdict, "추적": trace, "순서": order}
    lo = min(c for _, c in scored)
    tie = [l for l, c in scored if round(c, 2) == round(lo, 2)]     # 소수 둘째 자리까지 같을 때만 동률
    tie.sort(key=lambda l: (epochs[l], pool.index(l)))              # epoch 작은 쪽 → 먼저 저장된 쪽
    pick = tie[0]
    reasons.append(f"후보 {len(pool)}개 중 탈락 {len(pool)-len(alive)}개를 뺀 {len(alive)}개에서 고른다 (규칙 4).")
    reasons.append(f"고령 3칸 합산 micro CER 최저 = {lo:.2f}: " + ", ".join(f"{l}(ep{epochs[l]})" for l in tie)
                   + (f" — 동률 {len(tie)}개라 epoch 가 작은 쪽" if len(tie) > 1 else ""))
    reasons.append(f"**{pick}**(epoch {epochs[pick]}) 를 고른다.")
    if verdict[pick]["경고"]:
        reasons.append("경고 B (veto 아님): " + " · ".join(verdict[pick]["경고"]))
    # 7번 — 경고 C (보험): 방언 자리 정답률 병기. 결정은 바꾸지 않는다.
    warn_c = []
    if dacc:
        acc = {l: dacc[l]["정답률"] for l in alive if dacc.get(l, {}).get("정답률") is not None}
        mine, base_acc = acc.get(pick), (dacc.get("_baseline") or {}).get("정답률")
        if mine is not None and acc:
            top = max(acc, key=acc.get)
            if acc[top] - mine >= DACC_WARN_PT:
                warn_c.append(f"고른 {pick} 의 방언 자리 정답률 {mine:.2f} 가 후보 최고 {top}(ep{epochs[top]}) {acc[top]:.2f} 보다 "
                              f"{acc[top]-mine:.2f}pt 낮다 — CER 하락이 관례 학습일 수 있다 (규칙 7 경고 C · 표시만)")
            if base_acc is not None and mine <= base_acc:
                warn_c.append(f"고른 {pick} 의 방언 자리 정답률 {mine:.2f} 가 무튜닝 {base_acc:.2f} 이하 — 방언 자리를 못 움직였다 (규칙 7 경고 C)")
        verdict[pick]["경고_C"] = warn_c
        if warn_c:
            reasons.append("경고 C (veto 아님): " + " · ".join(warn_c))
        else:
            reasons.append(f"경고 C 없음 — 방언 자리 정답률 {mine} (무튜닝 {base_acc} · 후보 최고 {max(acc.values()) if acc else None})")
    return {"고름": pick, "실패": False, "epoch": epochs[pick], "고령_micro": dict(scored)[pick],
            "방언자리_정답률": (dacc or {}).get(pick, {}).get("정답률"), "경고_C": warn_c,
            "동률_후보": tie, "근거": reasons, "판정": verdict, "추적": trace, "순서": order}


def select_table(cks, epochs, deltas, verdict, dacc=None) -> str:
    lines = ["| 체크포인트 | epoch | 고령 micro | Δ 고령 | CI | 경상10~40 Δ | 서울 3칸 Δ | 방언 자리 정답 % | 판정 |",
             "|---|---:|---:|---:|---|---:|---|---:|---|"]
    for l in sorted(cks, key=lambda x: (epochs[x], list(cks).index(x))):
        dh = deltas[l].get(PICK_BUNDLE, {})
        dv = deltas[l].get(VETO_BUNDLE, {})
        se = " / ".join(fmt(deltas[l].get(n, {}).get("delta")) for n in WARN_BUNDLES)
        v = verdict[l]
        tag = "🚨 탈락" if v["탈락"] else ("⚠️ 경고" if (v["경고"] or v.get("경고_C")) else "통과")
        da = fmt((dacc or {}).get(l, {}).get("정답률"))
        lines.append(f"| {l} | {epochs[l]} | {fmt(cks[l]['묶음'][PICK_BUNDLE]['cer'])} | "
                     f"{fmt(dh.get('delta'))} | {fmt(dh.get('lo'))}~{fmt(dh.get('hi'))} | "
                     f"{fmt(dv.get('delta'))} | {se} | {da} | {tag} |")
    lines.append("")
    lines.append("Δ 는 무튜닝 대비 · **양수면 나빠졌다** · 탈락·경고는 Δ CI 하한 > 0 (규칙 2·3 · 2026-09-09 개정, τ 없음)")
    if dacc and dacc.get("_baseline"):
        lines.append(f"방언 자리 정답 % 는 병기(규칙 7 · 결정에 안 쓴다) · 무튜닝 {fmt(dacc['_baseline'].get('정답률'))} · "
                     f"방언 어절 {dacc['_baseline'].get('방언_어절', 0):,}개 · CI 없음")
    return "\n".join(lines)


# ── 실행 ─────────────────────────────────────────────────
def overlap_uids(train_roots) -> set:
    """② 교재와 겹치는 val 발화. 없으면 빈 집합 (열이 「—」로 나온다)."""
    uids = set()
    for root in train_roots:
        for m in sorted(Path(root).glob("sets/*/manifest.csv")):
            for r in csv.DictReader(open(m, encoding="utf-8")):
                uids.add(r["utterance_id"])
    return uids


def main():
    ap = argparse.ArgumentParser(description="val 8세트 채점 · §4.3 체크포인트 선택")
    ap.add_argument("--model", dest="models", action="append", default=[], required=True,
                    help="<라벨>=<조각 폴더>. 반복해서 체크포인트를 여럿 준다 (준 순서 = 저장 순서)")
    ap.add_argument("--baseline", help="무튜닝 모델의 라벨 (--select 에 필수)")
    ap.add_argument("--epoch", dest="epochs", action="append", default=[],
                    help="<라벨>=<epoch>. 라벨에서 ep<N> 을 못 읽을 때만 준다")
    ap.add_argument("--select", action="store_true", help="§4.3 3단 규칙으로 체크포인트를 고른다")
    ap.add_argument("--run-label", help="선택 결과 파일 이름 (--select 에 필수) 예: weighted")
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--sets-dir")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="세트 이름 (반복·쉼표). 스모크용")
    ap.add_argument("--cond", default="A")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", help="registry 대신 여기에 쓴다 (스모크 전용)")
    ap.add_argument("--allow-partial", action="store_true", help="8세트가 아니어도 채점한다 (스모크 전용)")
    ap.add_argument("--train-root", dest="train_roots", action="append", default=[],
                    help="②겹침을 잴 교재 세트 루트 (기본 $ACCENTURY_DATA/work/finetune/track2_tempbench_v1)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else OUT_DIR
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    sets_dir = Path(a.sets_dir) if a.sets_dir else sets_root(a.bench)
    have = list_sets(sets_dir)
    sets = [x for v in a.sets for x in v.split(",") if x] or [s for s in EXPECTED if s in have]
    gaps = [s for s in EXPECTED if s not in sets]
    if gaps and not a.allow_partial:
        sys.exit(f"val 8세트 중 {len(gaps)}개가 빠졌다: {gaps} — --allow-partial 은 스모크 전용")
    if a.allow_partial and out_dir.resolve() == OUT_DIR.resolve():
        sys.exit("--allow-partial 은 정본 경로에 쓰지 않는다 — --out-dir 을 registry 밖으로 준다")
    unknown = [s for s in sets if s not in EXPECTED]
    if unknown:
        sys.exit(f"val 세트가 아니다: {unknown}")

    attrs = {s: read_manifest(sets_dir, s) for s in sets}
    cluster = cluster_map(attrs)
    models = dict(kv(x, "model") for x in a.models)
    if len(models) != len(a.models):
        sys.exit("--model 라벨이 겹친다")
    for l in models:
        if not SAFE.match(l):
            sys.exit(f"라벨은 [A-Za-z0-9._-] 만 쓴다 (파일 이름이 된다): {l}")

    # 세트 정본과 발화 수 대조 — 조각이 아니라 세트 자체가 바뀐 경우를 잡는다
    chk = []
    if SETS_JSON.exists():
        ref = json.load(open(SETS_JSON, encoding="utf-8")).get("세트", {})
        for s in sets:
            want = ref.get(s, {}).get("발화")
            if want is not None and want != len(attrs[s]):
                chk.append(f"🚨 {s}: manifest {len(attrs[s])}행 ≠ 세트 정본 {want}행 ({SETS_JSON.name})")
    else:
        chk.append(f"세트 정본이 없어 발화 수를 대조하지 못했다: {SETS_JSON}")

    roots = a.train_roots or [V / "work/finetune/track2_tempbench_v1"]
    ov_all = overlap_uids(roots)
    ov = {u for s in sets for u in attrs[s] if u in ov_all}
    if not ov_all:
        chk.append(f"② 교재를 못 찾아 ②겹침을 재지 못했다: {[str(r) for r in roots]}")

    scored = {}
    for l, d in models.items():
        scored[l] = score_model(l, d, sets, attrs, a.cond, cluster, ov, a.iters, a.seed)

    # ── 모델마다 채점 JSON 한 장
    for l, m in scored.items():
        st, bt = set_table(m), bundle_table(m)
        doc = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
               "스크립트": "src/stt/score_val_track2.py",
               "명세": "docs/TRACK2_BENCHMARK_V2_SPEC.md §4.3", "벤치마크": a.bench, "라벨": l,
               "용도": NOTE_USE,
               "규칙": {"채점": "score_track2.score_combo (asr_metrics.score_one · kspon=False)",
                      "정답": "ref_dialect 정본 · ref_standard 병기",
                      "ci_group": f"재표집 {CLUSTER_COL} · {a.iters}회 · seed {a.seed} · 2.5~97.5 (val 정본)",
                      "ci_dialogue": "score_combo — 대화 단위 400회 seed 20260826 (시험지 표와의 연속성용 병기)",
                      "합산": "칸별 평균이 아니라 글자 가중 micro (§4.3 4번)",
                      "2초 이상": "라벨 길이(duration_label_sec) 기준"},
               "meta": m["meta"], "partial": bool(gaps), "빠진_세트": gaps,
               "세트": m["세트"], "묶음": m["묶음"],
               "표": {"세트": st, "묶음": bt},
               "②겹침": NOTE_OVERLAP, "서울_주의": NOTE_SEOUL, "정밀도_주의": NOTE_PREC,
               "경고": chk + m["경고"],
               "판정": "판정은 사람이 한다. 표는 숫자만 낸다."}
        p = out_dir / f"{BENCH}_stt__{l}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\n### {l} — 세트별\n{st}\n\n### {l} — 판정 묶음\n{bt}\n→ {p}")

    if not a.select:
        for w in dedup(chk + [w for m in scored.values() for w in m["경고"]]):
            print(f"경고: {w}")
        print(f"\n{NOTE_USE}\n{NOTE_SEOUL}\n{NOTE_PREC}\n{NOTE_OVERLAP}")
        if len(models) > 1:
            print("\n체크포인트를 고르려면 --select --baseline <무튜닝 라벨> --run-label <이름> 을 준다.")
        return 0

    # ── 선택 모드
    if not a.baseline or a.baseline not in models:
        sys.exit(f"--select 에는 --baseline <라벨> 이 필요하다 (준 라벨: {sorted(models)})")
    if not a.run_label or not SAFE.match(a.run_label):
        sys.exit("--select 에는 --run-label 이 필요하다 ([A-Za-z0-9._-])")
    cks = {l: m for l, m in scored.items() if l != a.baseline}
    if not cks:
        sys.exit("체크포인트가 없다 — --model 로 baseline 말고 하나 이상 더 준다")
    given = {k: int(v) for k, v in (kv(x, "epoch") for x in a.epochs)}
    epochs = {l: epoch_of(l, given) for l in cks}
    if len(set(epochs.values())) != len(epochs):
        print(f"경고: epoch 가 겹치는 체크포인트가 있다 {epochs} — 동률은 준 순서(저장 순서)로 가른다")

    base = scored[a.baseline]
    deltas = {l: deltas_for(base, cks[l], cluster, attrs, sets, a.iters, a.seed) for l in cks}

    # 7번 병기 — 방언 자리 정답률 (무튜닝은 "_baseline" 키)
    dacc = {l: dialect_acc(models[l], a.cond) for l in cks}
    dacc["_baseline"] = dialect_acc(models[a.baseline], a.cond)
    for l, v in dacc.items():
        if v.get("경고"):
            chk.append(f"방언 자리 정답률 못 냄 ({l}): {v['경고']}")

    res = choose(base, cks, epochs, deltas, dacc)
    tbl = select_table(cks, epochs, deltas, res["판정"], dacc)
    print(f"\n### 체크포인트 선택 — {a.run_label}\n{tbl}")
    print("\n" + "\n".join(f"  {r}" for r in res["근거"]))
    if res["실패"]:
        print("\n🚨 고른 체크포인트 없음 — §4.3 6번: 파인튜닝 1차를 실패로 보고한다.")
    else:
        print(f"\n고른 체크포인트: {res['고름']} (epoch {res['epoch']} · 고령 micro {res['고령_micro']:.2f} · "
              f"방언 자리 정답률 {res['방언자리_정답률']})")

    doc = {"생성": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "스크립트": "src/stt/score_val_track2.py --select",
           "명세": "docs/TRACK2_BENCHMARK_V2_SPEC.md §4.3 「체크포인트 선택 규칙」",
           "벤치마크": a.bench, "run": a.run_label, "baseline": a.baseline,
           "체크포인트": {l: {"epoch": epochs[l], "in_dir": scored[l]["meta"]["in_dir"],
                        "고령_micro": cks[l]["묶음"][PICK_BUNDLE]["cer"],
                        "방언자리_정답률": dacc[l]["정답률"]} for l in cks},
           "방언자리_정답률": {"무튜닝": dacc["_baseline"], "규칙": "dialect_pos_val_track2.tally · 공백 무시 정렬 · 고령 3칸 합산 · CI 없음",
                        "용도": f"규칙 7 병기 — 결정에 안 쓴다 · 경고선 {DACC_WARN_PT}pt 는 스크립트가 정한 값"},
           "개정": "2026-09-09 박재영 — 마진 τ·조기 종료·τ 동률 삭제. 옛 규칙 결과는 2026-09-09 오전 이전 판",
           "규칙": {"1번": "2 epoch 마다 + 마지막 저장 · 상한 15 전부 (조기 종료 없음)",
                  "2번": f"{VETO_BUNDLE} 짝지은 Δ CI 하한 > 0 → 탈락 A", "3번": "서울 3칸 각각 Δ CI 하한 > 0 → 경고 · 셋 다면 탈락",
                  "4번": f"{PICK_BUNDLE} 합산 micro 최저 · 둘째 자리까지 같으면 epoch 작은 쪽 · 그래도 같으면 먼저 저장된 쪽",
                  "5번": "삭제 (조기 종료 없음)",
                  "6번": "모두 탈락이면 파인튜닝을 실패로 보고한다 (무튜닝을 고르지 않는다)",
                  "7번": f"경고 C (보험 · 2026-09-09) — 고령 3칸 방언 자리 정답률 병기. 고른 것이 후보 최고보다 {DACC_WARN_PT}pt 이상 낮거나 무튜닝 이하면 표시만",
                  "Δ": "체크포인트 − 무튜닝 · 양수면 나빠졌다 · 같은 발화 위 짝지은 값"},
           "Δ": deltas, "결과": res, "표": tbl,
           "서울_주의": NOTE_SEOUL, "②겹침": NOTE_OVERLAP, "용도": NOTE_USE,
           "경고": dedup(chk + [w for m in scored.values() for w in m["경고"]]),
           "판정": "규칙 적용은 결정적이다. 규칙 자체를 바꾸는 판단은 사람이 한다."}
    p = out_dir / f"{BENCH}_select__{a.run_label}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {p}")
    for w in doc["경고"]:
        print(f"경고: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
