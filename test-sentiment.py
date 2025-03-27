from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "alsgyu/sentiment-analysis-fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 테스트할 문장 리스트
texts = [
    "이 제품 정말 마음에 들어요!",
    "기분이 너무 안 좋다",
    "그냥 평범한 하루였어요",
    "오늘 진짜 행복했어!",
    "짜증나 죽겠어",
    "생각보다 괜찮았어",
    "별로 기대 안 했는데 꽤 좋았어",
    "지루했어",
    "오늘 존나 피곤해"
]

# 라벨 매핑
label_map = {
    'LABEL_0': '부정',
    'LABEL_1': '중립',
    'LABEL_2': '긍정'
}

# 각 문장에 대해 결과 출력
threshold = 0.7

for text in texts:
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']

    # threshold 아래는 중립 처리
    if score < threshold:
        emotion = "중립 (애매함)"
    else:
        emotion = label_map[label]

    print(f"[{text}] → {emotion} (신뢰도: {score:.2f})")
