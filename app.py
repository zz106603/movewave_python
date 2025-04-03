from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import traceback

app = Flask(__name__)

# 모델 불러오기
model_name = "alsgyu/sentiment-analysis-fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 라벨 매핑
label_map = {
    'LABEL_0': '부정',
    'LABEL_1': '중립',
    'LABEL_2': '긍정'
}
threshold = 0.7

@app.route('/api/emotion/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': '텍스트가 없습니다.'}), 400

        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']

        if score < threshold:
            emotion = "중립"
        else:
            emotion = label_map[label]

        return jsonify({
            'text': text,
            'prediction': emotion,
            'confidence': round(score, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
