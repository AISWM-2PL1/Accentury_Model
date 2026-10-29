#!/usr/bin/env python3
"""ASR 평가 지표 — 볼트 [[평가 프로토콜 - STT]] §4 의사코드 그대로

이 파일은 명세의 구현체다. 규칙을 바꾸려면 **명세를 먼저 고친다.**

  §4.1 정규화     NFC → 한글·영숫자·공백만 → 숫자를 한글로 → 공백 압축
  §4.1b kspon    KsponSpeech 전용 전처리(이중표기·잡음태그·간투사).
                 방언 전사에는 해당 사항이 0건이므로 걸지 않아도 결과가 같다.
  §4.2 공식       CER(공백 제외) · WER(어절) · SER
                  sWER은 CER과 계산이 겹치므로 별도 산출하지 않는다
  §4.3 집계       micro = 편집거리 합 / 정답 단위 합
                  macro = 발화별 지표 평균 (단위 = 발화)
  §4.4 경계       정답 빈 문자열 → 제외 · 모델 실패 → failure_rate 분리
  §4.5 outlier    본값 미제외 · CER>100% 제외분을 별도 병기
  §4.6 단위       오류율은 퍼센트 소수 둘째 자리
"""
import re
import unicodedata

WS = re.compile(r"\s+")


# ── §4.1b KsponSpeech 전용 전처리 ────────────────────────
# OpenKoASR `openkoasr/dataset/KsponSpeech/utils.py` (MIT) 의 규칙을 옮긴 것.
# 리더보드 수치와 견주려면 같은 전처리를 써야 한다.
NOISE_LABELS = ["/b", "/l", "/o", "/n", "b/", "l/", "o/", "n/", "u/"]
_DUAL = re.compile(r"\(([^/]+)/([^)]+)\)")
_ENG_ONLY = re.compile(r"[A-Za-z\s&\-.]+")
_FILLER_SLASH = re.compile(r"([가-힣])/")


def kspon_pre(text: str) -> str:
    """KsponSpeech 전사 표기를 푼다. 방언 전사에는 해당 표기가 없다.

    이중표기 (A)/(B) → 영어면 앞(A), 아니면 뒤(B, 발음).
    잡음 태그 제거. 간투사는 슬래시만 떼고 글자는 남긴다.
    """
    def pick(m):
        front, back = m.group(1), m.group(2)
        return front.strip() if _ENG_ONLY.fullmatch(front.strip()) else back.strip()

    t = _DUAL.sub(pick, text or "")
    for lab in NOISE_LABELS:
        t = t.replace(lab, "")
    return _FILLER_SLASH.sub(r"\1", t)


# ⚠️ _DUAL 정규식은 괄호를 하나씩 흘린다 — `(A)/(B)` 에서 B 앞의 `(` 가 남는다.
#    OpenKoASR 원본도 똑같고, 뒤이은 정규화가 괄호를 지우므로 결과는 같다.
#    **고치지 마라.** 고치면 리더보드와 대조가 어긋난다.


# ── 숫자 → 한글 ─────────────────────────────────────────
_N1 = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_N2 = ["", "십", "백", "천"]
_BIG = ["", "만", "억", "조"]


def number_to_korean(s: str) -> str:
    """OpenKoASR 구현을 그대로 옮겼다.

    ⚠️ `123` → `일백이십삼` 처럼 십·백·천 앞의 `일` 을 남긴다.
       자연스러운 한국어는 `백이십삼` 이지만, **리더보드와 같은 값을 내려면
       같은 변환을 써야 한다.** 자연스럽게 고치지 마라.
    """
    s = s.lstrip("0")
    if not s:
        return "공"
    k = (len(s) + 3) // 4
    s = s.zfill(k * 4)
    out = []
    for i in range(k):
        part, buf = s[i * 4:(i + 1) * 4], []
        for j, dg in enumerate(part):
            d = int(dg)
            if d:
                buf.append(_N1[d] + _N2[3 - j])
        if buf:
            out.append("".join(buf) + _BIG[k - i - 1])
    return "".join(out)


# ── §4.1 정규화 ──────────────────────────────────────────
def normalize(text: str) -> str:
    """NFC → 한글·영숫자·공백만 → 숫자를 한글로 → 공백 압축.

    정답·가설 양쪽에 동일 적용한다.

    ⚠️ 숫자 한글화는 2026-08-06에 추가했다. KsponSpeech 전사의 7~14%가
       숫자를 포함하는데, 모델이 「30만 원」을 쓰고 정답이 「삼십만 원」이면
       전부 오류로 세게 된다. 방언 전사에는 숫자가 4건(0.06%)뿐이라
       우리 벤치마크 결과는 사실상 바뀌지 않는다.

    ⚠️ 문장부호는 **전부 제거**한다. OpenKoASR는 `, . ? !` 를 남기므로
       리더보드 수치와 완전히 같지는 않다. 모델마다 문장부호 출력 습관이
       달라 남기면 특정 모델에 불리해지기 때문이다. 재현 보고 때 명시한다.
    """
    v = unicodedata.normalize("NFC", text or "")
    out = []
    for ch in v:
        # `ch.isalnum()` 은 「영숫자」보다 넓어 한자·가나·키릴·낱자모(`ㅎ`)까지 통과시켰다.
        # 명세 §6 은 「한글·영숫자·공백만」이므로 완성형 한글과 ASCII 영숫자로 좁힌다
        # (코덱스 2026-08-30 지적 5번: 그 문자 1,335개가 살아남아 CER 을 부풀렸다).
        if ch.isspace() or ("가" <= ch <= "힣") or ("a" <= ch <= "z") \
                or ("A" <= ch <= "Z") or ("0" <= ch <= "9"):
            out.append(ch)
        else:
            out.append(" ")
    t = WS.sub(" ", "".join(out)).strip()
    t = re.sub(r"\d+", lambda m: number_to_korean(m.group()), t)
    return WS.sub(" ", t).strip()


# ── 편집거리 ─────────────────────────────────────────────
def edit_counts(ref: list, hyp: list):
    """Levenshtein. (거리, 치환, 삽입, 삭제) 를 돌려준다."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, m, 0
    if m == 0:
        return n, 0, 0, n
    # (거리, 치환, 삽입, 삭제)
    prev = [(j, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
            else:
                d_sub = prev[j - 1]
                d_ins = cur[j - 1]
                d_del = prev[j]
                best = min(d_sub[0], d_ins[0], d_del[0])
                if d_sub[0] == best:
                    cur[j] = (best + 1, d_sub[1] + 1, d_sub[2], d_sub[3])
                elif d_ins[0] == best:
                    cur[j] = (best + 1, d_ins[1], d_ins[2] + 1, d_ins[3])
                else:
                    cur[j] = (best + 1, d_del[1], d_del[2], d_del[3] + 1)
        prev = cur
    return prev[m]


# ── §4.2 발화 단위 지표 ──────────────────────────────────
def score_one(reference: str, hypothesis: str, status: str = "PASS",
              kspon: bool = False):
    """발화 하나를 채점한다.

    status != "PASS" 면 모델 실패로 보고 지표를 내지 않는다(§4.4).
    정답이 빈 문자열이면 집계에서 제외한다(§4.4).
    kspon=True 면 정답에 KsponSpeech 전용 전처리를 먼저 건다(§4.1b).
    가설(모델 출력)에는 걸지 않는다 — 모델은 그 표기를 뱉지 않는다.
    """
    ref = kspon_pre(reference) if kspon else reference
    R, H = normalize(ref), normalize(hypothesis)
    ref_c = list(R.replace(" ", ""))
    hyp_c = list(H.replace(" ", ""))
    ref_w, hyp_w = R.split(), H.split()

    row = {"reference_norm": R, "hypothesis_norm": H,
           "ref_chars": len(ref_c), "ref_words": len(ref_w)}

    if status != "PASS":
        row.update(status="FAIL", include=False, reason="model_failure")
        return row
    if not ref_c:
        row.update(status="PASS", include=False, reason="empty_reference")
        return row

    cd, cs, ci, cdel = edit_counts(ref_c, hyp_c)
    wd, ws_, wi, wdel = edit_counts(ref_w, hyp_w)
    row.update(
        status="PASS", include=True, reason="",
        cer_dist=cd, cer=cd / len(ref_c),
        wer_dist=wd, wer=wd / max(1, len(ref_w)),
        ser=0 if R == H else 1,
        sub=cs, ins=ci, dele=cdel,
    )
    return row


# ── §4.3 집계 ────────────────────────────────────────────
def aggregate(rows: list, outlier_cer: float = 1.0):
    """발화별 결과 → micro·macro. 퍼센트 소수 둘째 자리(§4.6).

    outlier_cer: 별도 병기용 임계값. 본값은 미제외(§4.5).
    """
    n_all = len(rows)
    fail = [r for r in rows if r.get("status") == "FAIL"]
    empty = [r for r in rows if r.get("reason") == "empty_reference"]
    use = [r for r in rows if r.get("include")]

    def pack(rs, tag):
        if not rs:
            return {}
        rc = sum(r["ref_chars"] for r in rs)
        rw = sum(r["ref_words"] for r in rs)
        pc = lambda x: round(x * 100, 2)
        return {
            f"n{tag}": len(rs),
            f"cer_micro{tag}": pc(sum(r["cer_dist"] for r in rs) / max(1, rc)),
            f"cer_macro{tag}": pc(sum(r["cer"] for r in rs) / len(rs)),
            f"wer_micro{tag}": pc(sum(r["wer_dist"] for r in rs) / max(1, rw)),
            f"wer_macro{tag}": pc(sum(r["wer"] for r in rs) / len(rs)),
            f"ser{tag}": pc(sum(r["ser"] for r in rs) / len(rs)),
        }

    out = {"n_input": n_all,
           "n_scored": len(use),
           "n_excluded_empty_ref": len(empty),
           "failure_rate": round(len(fail) / max(1, n_all) * 100, 2)}
    out.update(pack(use, ""))
    kept = [r for r in use if r["cer"] <= outlier_cer]
    out.update(pack(kept, "_no_outlier"))
    out["n_outlier"] = len(use) - len(kept)
    out["outlier_rule"] = f"CER > {outlier_cer*100:.0f}%"
    return out


# ── 자체 점검 ────────────────────────────────────────────
if __name__ == "__main__":
    T = [
        ("완전 일치", "여개는 봄만 되만", "여개는 봄만 되만", 0.0, 0.0, 0),
        ("한 글자 치환", "가나다", "가나라", 1/3, 1.0, 1),
        ("가설 비어 있음", "가나다", "", 1.0, 1.0, 1),
        ("구두점만 다름", "안녕, 하세요.", "안녕 하세요", 0.0, 0.0, 0),
        ("띄어쓰기만 다름", "안녕 하세요", "안녕하세요", 0.0, 1.0, 1),
    ]
    print(f"{'항목':18s} {'CER':>7s} {'WER':>7s} {'SER':>4s}   기대")
    ok = True
    for name, r, h, ec, ew, es in T:
        x = score_one(r, h)
        good = abs(x["cer"]-ec) < 1e-9 and abs(x["wer"]-ew) < 1e-9 and x["ser"] == es
        ok &= good
        print(f"{name:18s} {x['cer']:7.4f} {x['wer']:7.4f} {x['ser']:4d}   "
              f"{ec:.4f}/{ew:.4f}/{es} {'OK' if good else '❌'}")

    # 경계 조건
    e = score_one("", "뭐라도")
    print(f"\n정답 빈 문자열 → include={e['include']} reason={e['reason']}")
    f = score_one("가나다", "가나다", status="ERROR")
    print(f"모델 실패      → include={f['include']} reason={f['reason']}")

    rows = [score_one(r, h) for _, r, h, *_ in T] + [e, f]
    import json
    print("\n집계:")
    print(json.dumps(aggregate(rows), ensure_ascii=False, indent=1))
    print("\n" + ("✅ 자체 점검 통과" if ok else "❌ 실패"))
