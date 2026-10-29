#!/usr/bin/env python3
"""트랙2 무튜닝 STT 실행 진입점 — `docs/TRACK2_STT_PIPELINE_SPEC.md` §6 B1-③

  루프를 돌며 결과를 저장한다. 채점하지 않는다 · 표를 만들지 않는다.

산출물 1 (inference 명세 §1)
  registry/benchmark/track2_stt/<cond>__<set>.csv.gz      # 2 × 16 = 32조각

사용
  python src/stt/run_track2_stt.py --set set01_age_10s --cond A --limit 20 \
         --out-dir work/scratch/track2_stt_smoke
  python src/stt/run_track2_stt.py --all
  python src/stt/run_track2_stt.py --verify

② 보강 세트 (`docs/TRACK2_STT_ADDON_SETS_SPEC.md` §5 · 세트 4개 · 조건 A 만 · 별도 결과 폴더)
  python src/stt/run_track2_stt.py --bench track2_tempbench_v1_addon --all --cond A
  python src/stt/run_track2_stt.py --bench track2_tempbench_v1_addon --verify --cond A
  → registry/benchmark/track2_stt_addon/A__set17_….csv.gz  (②의 track2_stt/ 와 섞지 않는다)
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt import track2_data as D

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "registry/benchmark/track2_stt"

COLUMNS = ["utterance_id", "set", "cond", "dialogue_id", "ref_dialect", "ref_standard",
           "hypothesis", "status", "normalization_note", "sec", "dur", "err"]


def shard_path(out_dir: Path, cond: str, set_name: str) -> Path:
    return out_dir / f"{cond}__{set_name}.csv.gz"


def shard_rows(path: Path) -> int:
    """조각의 행 수. 헤더가 계약과 다르면 -1 (완성본으로 보지 않는다)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            head = next(rd, None)
            if head != COLUMNS:
                return -1
            return sum(1 for _ in rd)
    except Exception:
        return -1


def is_complete(path: Path, expect: int) -> bool:
    """파일 존재만으로 판정하지 않는다 — 헤더와 행 수가 manifest 와 같을 때만 완성본이다."""
    return path.exists() and shard_rows(path) == expect


def write_shard(path: Path, rows: list):
    """임시 파일에 쓴 뒤 교체한다 — 죽다 만 조각이 완성본으로 보이면 안 된다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def save_meta(out_dir: Path, meta: dict):
    """임시 파일 + os.replace. 조각을 쓸 때마다 부른다."""
    tmp = out_dir / "_run_meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "_run_meta.json")


def load_meta(out_dir: Path) -> dict:
    p = out_dir / "_run_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"shards": {}}


def run_one(set_name: str, cond: str, run, out_dir: Path, limit: int = 0,
            meta: dict = None, stamp: dict = None) -> dict:
    man = D.read_manifest(set_name, limit)
    path = shard_path(out_dir, cond, set_name)
    if is_complete(path, len(man)):
        print(f"  건너뜀 (완성본 {len(man):,}행) {path.name}", flush=True)
        return {"set": set_name, "cond": cond, "skipped": True, "n": len(man)}

    rows, audio_sec, proc_sec, n_fail = [], 0.0, 0.0, 0
    t_set = time.perf_counter()
    for i, m in enumerate(man):
        try:
            it = D.load_item(m, cond)
        except Exception as e:                      # 로드 실패도 멈추지 않는다
            d, s = D.refs(m)
            rows.append({"utterance_id": m["utterance_id"], "set": set_name, "cond": cond,
                         "dialogue_id": m.get("dialogue_id", ""), "ref_dialect": d,
                         "ref_standard": s, "hypothesis": "", "status": "LOAD_FAIL",
                         "normalization_note": "", "sec": 0,
                         # 로드가 실패해도 라벨 길이는 manifest 에 이미 있다 (Row 계약)
                         "dur": round(float(m.get("duration_label_sec") or 0.0), 3),
                         "err": str(e)[:80]})
            n_fail += 1
            continue
        t1 = time.perf_counter()
        try:
            hyp, st, err = run(it.audio), "PASS", ""
        except Exception as e:
            hyp, st, err = "", f"FAIL:{type(e).__name__}", str(e)[:80]
            n_fail += 1
        el = time.perf_counter() - t1
        audio_sec += len(it.audio) / D.SR
        proc_sec += el
        rows.append({"utterance_id": it.utterance_id, "set": set_name, "cond": cond,
                     "dialogue_id": it.dialogue_id, "ref_dialect": it.ref_dialect,
                     "ref_standard": it.ref_standard, "hypothesis": hyp, "status": st,
                     "normalization_note": it.normalization_note,
                     "sec": round(el, 4), "dur": round(it.dur, 3), "err": err})
        if (i + 1) % 200 == 0:
            # 이 RTFx 는 진행률 표시용이다. `registry/rtf` 에 넣지 않는다 —
            # 속도 계측 규약과 조건이 다르다 (inference 명세 §9).
            print(f"  {i+1:,}/{len(man):,}  RTFx={audio_sec/max(1e-9, proc_sec):.1f}", flush=True)

    write_shard(path, rows)
    # 🚨 메타는 조각과 같이 남긴다. 루프 끝에서만 쓰면 중단된 실행이 「메타 없는
    #    완성 조각」을 남겨, 다른 device 로 재개해도 혼합 검사가 통과한다
    #    (코덱스 2판 지적 2026-08-27).
    if meta is not None:
        meta["shards"][f"{cond}__{set_name}"] = dict(
            stamp or {}, n=len(rows), n_fail=n_fail)
        save_meta(out_dir, meta)
    el = time.perf_counter() - t_set
    print(f"  저장 {path.name}  {len(rows):,}행 · 실패 {n_fail} · {el/60:.1f}분", flush=True)
    return {"set": set_name, "cond": cond, "skipped": False, "n": len(rows),
            "n_fail": n_fail, "sec": round(el, 1)}


def verify(out_dir: Path, sets: list = None, conds: tuple = ("A", "B")) -> int:
    """조각(세트 × 조건)과 행 수를 검사한다. 어긋나면 종료 코드 1.
    기본은 전 세트 × A/B (②의 32조각). addon 은 `--cond A` 로 범위를 좁힌다 — 안 좁히면
    없는 B 조각 때문에 실패하고 `set -e` 래퍼가 업로드·종료 전에 멈춘다 (코덱스 2026-09-01 #2)."""
    bad = []
    sets = sets or D.list_sets()
    for s in sets:
        want = D.n_rows(s)
        for c in conds:
            p = shard_path(out_dir, c, s)
            got = shard_rows(p) if p.exists() else None
            mark = "OK" if got == want else "MISMATCH" if p.exists() else "없음"
            if mark != "OK":
                bad.append(f"{c}__{s}: {mark} (want {want:,} · got {got})")
            print(f"  {mark:9} {c}__{s}  {want:,}행", flush=True)
    print(f"\n조각 {len(sets)*len(conds)}개 중 문제 {len(bad)}개", flush=True)
    for b in bad:
        print("  " + b, flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_names", action="append", default=[],
                    help="세트 이름. 반복하거나 쉼표로 여러 개")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bench", default=D.BENCH_DEFAULT,
                    help="세트 루트 이름 (기본 ②의 16세트 · 보강은 track2_tempbench_v1_addon)")
    ap.add_argument("--cond", choices=["A", "B", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 어댑터 폴더 (train_lora_whisper.py 산출). 주면 원본 위에 얹어 돌린다 — "
                         "병합하지 않는다. 조각 메타에 sha256 을 남겨 무튜닝 조각과 섞이지 않게 한다")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="기존 조각과 다른 device/dtype 으로 이어 돌린다 (§9 위반)")
    a = ap.parse_args()
    a.set_names = [x for v in a.set_names for x in v.split(",") if x]

    # 🚨 --limit 를 준 실행이 기본 경로에 20행짜리 조각을 남기면, 이후 전량 실행이
    #    그것을 완성본으로 보고 건너뛴다 (코덱스 지적 2026-08-26).
    if a.limit and not a.out_dir:
        sys.exit("--limit 를 주면 --out-dir 도 반드시 준다 (registry/ 밖으로)")

    D.set_bench(a.bench)
    # 벤치마다 결과 폴더를 나눈다 — addon 조각이 ②의 32조각 폴더에 섞이면 ② 채점의
    # 기대 세트 검사와 정본이 흔들린다 (addon 명세 §8).
    out_default = OUT_DEFAULT if a.bench == D.BENCH_DEFAULT else \
        OUT_DEFAULT.with_name(f"track2_stt_{a.bench.removeprefix(D.BENCH_DEFAULT + '_')}")
    out_dir = Path(a.out_dir) if a.out_dir else out_default
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir.resolve()
    # 존재 확인만으로는 부족하다 — `--out-dir registry/...` 를 주면 부분 조각이
    # 정식 산출물 자리에 남는다 (코덱스 지적 2026-08-27).
    if a.limit and out_dir.is_relative_to((ROOT / "registry").resolve()):
        sys.exit(f"--limit 실행은 registry/ 밖에 쓴다: {out_dir}")

    conds = ["A", "B"] if a.cond == "both" else [a.cond]
    if a.verify:
        sys.exit(verify(out_dir, a.set_names or None, tuple(conds)))

    if not (a.all or a.set_names):
        sys.exit("--set 또는 --all 중 하나가 필요하다")

    sets = D.list_sets() if a.all else a.set_names

    from stt import track2_asr as A
    dev = A.pick_device(a.device)
    dt_want = str(A.dtype_for(dev)).replace("torch.", "")
    meta = load_meta(out_dir)
    ad_want = A.adapter_id(a.adapter)["sha256"] if a.adapter else None
    prev = {(v.get("device"), v.get("dtype"), v.get("adapter_sha256"))
            for v in meta["shards"].values()}
    orphan = [f"{c}__{s}" for s in D.list_sets() for c in conds
              if is_complete(shard_path(out_dir, c, s), D.n_rows(s))
              and f"{c}__{s}" not in meta["shards"]]
    if orphan and not a.allow_mixed:
        sys.exit(f"메타 없는 완성 조각 {len(orphan)}개: {orphan[:3]}… — 어떤 device 로 만든 것인지"
                 " 알 수 없다. 지우고 다시 돌리거나 --allow-mixed")
    if prev and (dev, dt_want, ad_want) not in prev and not a.allow_mixed:
        # 🚨 어댑터도 이 검사에 든다 — 무튜닝 조각과 파인튜닝 조각이 한 폴더에 섞이면
        #    그 폴더의 CER 은 두 모델의 평균이 되고 아무도 눈치채지 못한다.
        sys.exit(f"기존 조각은 (device, dtype, adapter) {sorted(map(str, prev))} 로 만들어졌다. "
                 f"지금은 ({dev}, {dt_want}, {ad_want}) 다 — 한 방법으로 끝까지 간다 "
                 "(inference 명세 §9). 섞으려면 --allow-mixed")

    # 완성 여부를 먼저 가른다. 재개할 것이 없는데 모델을 올리면 캐시·네트워크·의존성
    # 문제로 애먼 곳에서 죽는다 (코덱스 3판 지적 2026-08-27).
    todo = [(x, c) for x in sets for c in conds
            if not is_complete(shard_path(out_dir, c, x), len(D.read_manifest(x, a.limit)))]
    if not todo:
        print(f"완성본 {len(sets)*len(conds)}개 — 돌릴 것이 없다 (모델을 올리지 않는다)", flush=True)
        return

    print(f"세트 {len(sets)} × 조건 {len(conds)} · 남은 조각 {len(todo)} · device={dev} · "
          "모델 로드 중…", flush=True)
    run, repo, dt, ad = A.build(dev, a.adapter)
    if ad:
        print(f"어댑터 {ad['path']} · {ad['file']} {ad['bytes']:,}B · sha256 {ad['sha256'][:12]}…", flush=True)

    stamp = {"model": repo, "device": dev, "dtype": dt, "limit": a.limit}
    if ad:
        stamp |= {"adapter": ad["path"], "adapter_sha256": ad["sha256"], "adapter_bytes": ad["bytes"]}
    done = []
    for x, c in todo:
        print(f"[{x} · {c}]", flush=True)
        done.append(run_one(x, c, run, out_dir, a.limit, meta, stamp))

    # ④가 meta 에 실을 실행 조건 (§9 수용 기준 6).
    # 🚨 조각마다 따로 적는다. 전부 건너뛴 재실행이 이전 실행의 device·dtype 을
    #    덮어쓰면 혼합 실행이 감춰지고 실행 조건이 거짓으로 기록된다
    #    (코덱스 지적 2026-08-27).
    for d in done:                                    # 소요 시간만 마지막에 채운다
        k = f"{d['cond']}__{d['set']}"
        if not d.get("skipped") and k in meta["shards"]:
            meta["shards"][k]["sec"] = d.get("sec")
    save_meta(out_dir, meta)
    n_new = sum(1 for d in done if not d.get("skipped"))
    print(f"\n조각 {len(done)}개 (신규 {n_new}) · meta → {out_dir/'_run_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
