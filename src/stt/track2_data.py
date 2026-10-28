#!/usr/bin/env python3
"""트랙2 세트 로더 — `docs/TRACK2_STT_PIPELINE_SPEC.md` §6 B1-① 의 구현체

  세트 폴더를 읽어 (소리, 정답) 을 내놓는다. 그게 전부다.

하지 않는 것 (§6 「하지 말아야 할 것」)
  모델을 부르지 않는다 · 파일을 쓰지 않는다 · 채점하지 않는다

규칙을 바꾸려면 명세를 먼저 고친다.
"""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 맥과 EC2 의 데이터 위치가 다르다. 경로를 코드에 박지 않는다 (`run_asr.py:20`).
V = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))
BENCH_DEFAULT = "track2_tempbench_v1"
SETS = V / "work/benchmark" / BENCH_DEFAULT / "sets"
SR = 16000


SET_ROOTS = ("work/benchmark", "work/finetune")   # 찾는 순서. 앞이 이기게 둔다.


def set_bench(name: str) -> Path:
    """세트 루트를 다른 벤치마크 이름으로 바꾼다 (② 보강 세트 `track2_tempbench_v1_addon`,
    `docs/TRACK2_STT_ADDON_SETS_SPEC.md` §5). 기본값은 ②의 16세트다.

    **두 자리를 찾는다** — 시험지·참고선은 `work/benchmark/<name>/sets` 에 있고
    교재와 val 은 `work/finetune/<name>/sets` 에 있다 (`build_track2_v2.py` 가 그렇게 쓴다).
    둘 다 있으면 benchmark 쪽이 이긴다. 둘 다 없으면 옛 동작대로 benchmark 경로를
    돌려준다 — 「없다」는 판정은 부르는 쪽이 한다 (`run_track2_stt.py` 가 존재를 검사한다)."""
    global SETS
    for r in SET_ROOTS:
        cand = V / r / name / "sets"
        if cand.is_dir():
            SETS = cand
            return SETS
    SETS = V / SET_ROOTS[0] / name / "sets"
    return SETS

# 조건 B: 클립마다 그 클립의 RMS 를 -20 dBFS 로 맞춘다 (inference 명세 §4).
TARGET_DBFS = -20.0
TARGET_RMS = 10.0 ** (TARGET_DBFS / 20.0)   # 0.1


@dataclass
class Item:
    """①→③ 계약 (§6 데이터 계약)"""
    utterance_id: str
    audio: np.ndarray          # float32 1-D · 16 kHz · 조건 B 면 정규화된 값
    ref_dialect: str           # 정본
    ref_standard: str          # 병기
    dialogue_id: str           # 부트스트랩 클러스터 (§8)
    dur: float                 # 라벨 기준 길이 초
    normalization_note: str    # 빈칸 또는 RMS_ZERO


def set_dir(set_name: str) -> Path:
    d = SETS / set_name
    if not d.is_dir():
        raise FileNotFoundError(f"세트 폴더가 없다: {d}")
    return d


def list_sets() -> list:
    return sorted(p.name for p in SETS.iterdir() if p.is_dir())


def read_manifest(set_name: str, limit: int = 0) -> list:
    """manifest.csv(27열) 를 그대로 읽는다. 거르지 않는다."""
    with open(set_dir(set_name) / "manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def n_rows(set_name: str) -> int:
    """③ 의 완성본 판정에 쓰는 발화 수."""
    return len(read_manifest(set_name))


def refs(row: dict) -> tuple:
    """정답 두 개. `_clean` 이 비면 원문으로 떨어진다 (inference 명세 §5 결정).

    🚨 fallback 없이 짜면 `set01_age_10s` 8,567행 중 1,958행으로만 CER 이 나온다 —
    오염이 없던 행의 `_clean` 은 빈 문자열이기 때문이다
    (`build_track2_manifest.py:193~207`).
    원문마저 비면 fallback 하지 않는다. `aggregate()` 가 `empty_reference` 로 뺀다.
    """
    d = (row.get("dialect_text_clean") or "").strip() or (row.get("dialect_text") or "")
    s = (row.get("standard_text_clean") or "").strip() or (row.get("standard_text") or "")
    return d, s


def load_wav(path: Path) -> np.ndarray:
    """16 kHz mono float32. **아니면 예외** (§6 B1-①).

    `run_asr.py:load_wav` 는 다채널을 평균으로 내리지만 여기서는 예외로 올린다 —
    명세가 「16 kHz mono, 아니면 예외」이고, 조용히 다운믹스하면 다른 소리를
    추론해 놓고 성공으로 기록된다 (코덱스 지적 2026-08-27).
    """
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim != 1:
        raise ValueError(f"{path.name}: {x.ndim}차원 — mono가 아니다")
    if sr != SR:
        raise ValueError(f"{path.name}: {sr}Hz — 16kHz가 아니다")
    return x


def normalize_level(x: np.ndarray) -> tuple:
    """조건 B. 원본 wav 는 고치지 않는다 — 메모리 안에서만 곱한다.

    무음(RMS 0)은 나누기가 죽는다. 정규화를 건너뛰고 `RMS_ZERO` 를 남기되
    **status 는 PASS 로 둔다** — 바꾸면 `asr_metrics.py:159~163` 이 모델 실패로
    보고 채점에서 뺀다 (코덱스 지적 2026-08-26).

    클리핑은 하지 않는다. 명세가 정한 것은 RMS 를 맞추는 것뿐이다.
    """
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    if rms <= 0.0:
        return x, "RMS_ZERO"
    return (x * (TARGET_RMS / rms)).astype(np.float32), ""


def load_item(row: dict, cond: str) -> Item:
    """manifest 한 행 → Item. 로드 실패는 예외로 올린다 (③이 LOAD_FAIL 로 기록)."""
    if cond not in ("A", "B"):
        raise ValueError(f"cond 는 A 또는 B 다: {cond}")
    # `clip_file` 은 세트 폴더 기준 상대경로다 (실측: `clips/XXX.wav`).
    x = load_wav(set_dir(row["set"]) / row["clip_file"])
    note = ""
    if cond == "B":
        x, note = normalize_level(x)
    d, s = refs(row)
    return Item(
        utterance_id=row["utterance_id"],
        audio=x,
        ref_dialect=d,
        ref_standard=s,
        dialogue_id=row.get("dialogue_id", ""),
        dur=float(row.get("duration_label_sec") or 0.0),
        normalization_note=note,
    )
