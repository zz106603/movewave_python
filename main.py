import json
import os
import random
import traceback
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Optional

# uvicorn main:app --reload --port 5000

app = FastAPI()

# CORS 허용 (Spring 8080과 통신 가능하게 설정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안이 필요하면 프론트 주소만 명시: ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 감정 분석 모델 로드
emotion_model_name = "nlp04/korean_sentiment_analysis_kcelectra"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
classifier = pipeline("sentiment-analysis", model=emotion_model, tokenizer=emotion_tokenizer)

# 키워드 JSON 로딩
with open(os.path.join(os.path.dirname(__file__), 'emotion_keywords.json'), encoding='utf-8') as f:
    keyword_map = json.load(f)

# 입력 데이터 형식 정의
class EmotionRequest(BaseModel):
    text: str
    type: Optional[str] = ""

@app.post("/api/emotion/predict")
async def predict(request_data: EmotionRequest):
    try:
        text = request_data.text.strip()
        genre = request_data.type.strip()

        if not text:
            return {"error": "텍스트가 없습니다."}

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

        return {
            'text': text,
            'prediction': label,
            'confidence': round(score, 2),
            'keywords': selected_keywords
        }

    except Exception as e:
        return {
            'error': str(e),
            'trace': traceback.format_exc()
        }
