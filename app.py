from flask import Flask, request, jsonify
from model_module import RobertaModelWrapper

app = Flask(__name__)

# Load model and tokenizer
model_wrapper = RobertaModelWrapper(model_path='weights_final.h5', tokenizer_name='roberta-base')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    sentiment = data.get('sentiment')

    if not text or not sentiment:
        return jsonify({'error': 'Invalid input'}), 400

    prediction = model_wrapper.predict(text, sentiment)

    return jsonify({'selected_text': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)