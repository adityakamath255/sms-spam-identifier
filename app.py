from flask import Flask, render_template, request, jsonify
from predictor import Predictor

app = Flask(__name__)
predictor = Predictor()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_spam():
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400

        prediction, probability = predictor.predict(message)
        confidence = probability if probability > 0.5 else 1 - probability

        return jsonify({
            "prediction": prediction,
            "probability": f"{probability:.2%}",
            "confidence": f"{confidence:.2%}",
            "message": message
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False)
