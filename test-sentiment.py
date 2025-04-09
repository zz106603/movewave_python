from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "nlp04/korean_sentiment_analysis_kcelectra"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

texts = [
    "오늘 너무 지치고 외로워.",
    "기분이 정말 좋아!",
    "그냥 그런 하루였어.",
    "짜증나는 일만 가득해.",
    "뭔가 기분이 애매하다."
]

label_map = {
    "LABEL_0": "부정",
    "LABEL_1": "중립",
    "LABEL_2": "긍정"
}

print(model.config.id2label)

results = sentiment_pipe(texts)

for text, result in zip(texts, results):
    label = label_map.get(result['label'], result['label'])
    print(f"{text} => 감정: {label}, 점수: {result['score']:.2f}")
