from transformers import pipeline

print("모델 로딩 중...")

classifier = pipeline("sentiment-analysis")

print("모델 준비 완료. 테스트 시작!")

text = "I love this music so much!"
result = classifier(text)

print("결과:", result)