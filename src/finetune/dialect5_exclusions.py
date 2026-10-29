"""방언5 test 집계에서 빼는 대화 — 라벨 결함 (2026-10-03, `docs/DIALECT5_RESULTS_2026-10-02.md` §17.1).

  왜   AI Hub 원본 라벨의 발화 시각이 실제 음성과 어긋난 대화는 모델과 무관하게 CER 이 80% 대로 나와 세트 평균과 연령 칸을 왜곡한다.
       v2 manifest·클립·채점 CSV 는 그대로 두고(재현성), **집계 단계에서만** 뺀다. 집계 스크립트들은 이 목록을 기본으로 적용하고
       `--no-exclude` 로 옛 값(전체)을 다시 낼 수 있다.
  찾는 법 `src/finetune/detect_dialect5_misalignment.py` — 가설이 자기 정답보다 이웃 발화 정답과 더 맞는 발화가 많은 대화.
       2026-10-03 에 5지역 val·test 10개 세트(발화 20개 이상 대화)를 훑어 아래 1개만 걸렸다.
"""

EXCLUDE_DIALOGUES = {
    "DCDG20003365": (
        "충청 test · 60대 미만(30대 남성 1명) 54발화 900글자. 838초까지의 발화에서 클립에 같은 화자의 이웃(대개 다음) 발화가 들어 있다"
        "(가설이 이웃 정답과 유사도 0.6 이상으로 맞는 24발화에서 클립에 들린 말의 라벨 시작이 클립 시작보다 2.1~6.8초 뒤"
        " = 라벨 시각이 실제 말보다 그만큼 늦음 · 발화마다 다름)."
        " 1,146초 이후 발화는 자기 정답과 맞는다. 무튜닝·방언별·통합 모두 CER 80~82%."
        " 원천 WAV 는 S3 raw/aihub-dialect-free-cc/…/(비식별화완료)충청도_11.zip 의 DCDG20003365.wav (16 kHz mono, 1,231.9초)."
    ),
}


def excluded_utterances(manifest_rows):
    """manifest 행(dict: utterance_id·dialogue_id) 목록에서 제외 대상 utterance_id 집합."""
    return {r["utterance_id"] for r in manifest_rows if r.get("dialogue_id") in EXCLUDE_DIALOGUES}
