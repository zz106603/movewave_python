import json
import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import traceback
import random

app = Flask(__name__)

# 감정 분석 모델 로드
emotion_model_name = "nlp04/korean_sentiment_analysis_kcelectra"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
classifier = pipeline("sentiment-analysis", model=emotion_model, tokenizer=emotion_tokenizer)

# ✅ JSON 파일에서 키워드 맵 불러오기
with open(os.path.join(os.path.dirname(__file__), 'emotion_keywords.json'), encoding='utf-8') as f:
    keyword_map = json.load(f)

@app.route('/api/emotion/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        genre = data.get('type', '').strip()

        if not text:
            return jsonify({'error': '텍스트가 없습니다.'}), 400

        # 감정 분석
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']

        emotion_keywords = keyword_map.get(label, {})
        if genre == "" or genre == "전체":
            candidates = emotion_keywords.get("공통", [])
        else:
            candidates = emotion_keywords.get(genre, [])

        if not candidates:
            candidates = ["감정에 어울리는 음악"]

        selected_keywords = random.sample(candidates, min(len(candidates), 2))

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
