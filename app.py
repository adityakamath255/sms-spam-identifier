from flask import Flask, jsonify, render_template, request
from prediction import Predictor

app = Flask(__name__)
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor


@app.route('/')
def home():
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    result = get_predictor().predict(message)
    result["probability"] = f"{result['probability']:.2%}"
    result["confidence"] = f"{result['confidence']:.2%}"
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False)
