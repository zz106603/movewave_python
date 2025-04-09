from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import traceback
import random

app = Flask(__name__)

# 감정 분석 모델
emotion_model_name = "nlp04/korean_sentiment_analysis_kcelectra"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
classifier = pipeline("sentiment-analysis", model=emotion_model, tokenizer=emotion_tokenizer)

# 감정별 추천 키워드 맵
keyword_map = {
    "기쁨(행복한)": ["행복한 기분이 드는 노래", "따뜻한 감성 음악"],
    "고마운": ["감사한 마음을 표현한 노래", "고마움을 느끼게 하는 발라드"],
    "설레는(기대하는)": ["설레는 감정을 담은 노래", "기대되는 순간에 어울리는 음악"],
    "사랑하는": ["사랑 노래", "달달한 분위기의 음악"],
    "즐거운(신나는)": ["신나는 댄스곡", "에너지 넘치는 노래"],
    "일상적인": ["편안한 일상 브금", "루틴에 어울리는 배경음악"],
    "생각이 많은": ["생각을 정리할 수 있는 음악", "잔잔한 피아노 연주곡"],
    "슬픔(우울한)": ["위로가 되는 노래", "슬픔을 달래주는 감성곡"],
    "힘듦(지침)": ["지친 마음을 위한 힐링 음악", "응원이 되는 노래"],
    "짜증남": ["스트레스를 풀어주는 음악", "강한 비트의 힙합"],
    "걱정스러운(불안한)": ["마음을 안정시켜주는 음악", "편안한 자연 소리"]
}

@app.route('/api/emotion/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': '텍스트가 없습니다.'}), 400

        # 감정 분석
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']

        # 키워드 선택 (해당 감정 키워드 1개 이상 무작위 선택)
        keywords = keyword_map.get(label, ["감정에 어울리는 음악"])
        selected_keywords = random.sample(keywords, min(len(keywords), 2))  # 2개 랜덤 선택

        return jsonify({
            'text': text,
            'prediction': label,
            'confidence': round(score, 2),
            'keywords': selected_keywords
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
