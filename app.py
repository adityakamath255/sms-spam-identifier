from flask import Flask, render_template, request, jsonify
import os
import subprocess
import threading
import time
from queue import Queue, Empty

app = Flask(__name__)
input_queue = Queue()
output_queue = Queue()

def read_output(process):
    for line in iter(process.stdout.readline, ''):
        if line.strip():
            print("Got output:", line.strip())  # debug print
            output_queue.put(line.strip())
    process.stdout.close()

def write_input(process):
    while True:
        message = input_queue.get()
        if message is None:
            break
        print("Sending input:", message)  # debug print
        process.stdin.write(message + "\n")
        process.stdin.flush()

def initialize_predictor():
    global predictor_process, output_thread, input_thread
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(current_dir, "predictor.py")
    
    print(f"Starting predictor at: {predictor_path}")  # debug print
    
    predictor_process = subprocess.Popen(
        ["python3", predictor_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=current_dir
    )
    
    output_thread = threading.Thread(target=read_output, args=(predictor_process,), daemon=True)
    input_thread = threading.Thread(target=write_input, args=(predictor_process,), daemon=True)
    
    output_thread.start()
    input_thread.start()

initialize_predictor()

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
        
        input_queue.put(message)
        
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = output_queue.get(timeout=0.5)
                if "Prediction:" in response:
                    response = response.replace('Enter message to classify (or \'quit\'): ', '')
                    response = response.replace('Prediction: ', '')
                    response = response.replace('Confidence: ', '')
                    response = response.replace(')', '')
                    parts = response.split('(')
                    prediction = parts[0].strip()
                    probability = parts[1].replace(')', '').strip()
                    
                    return jsonify({
                        'prediction': prediction,
                        'probability': probability,
                        'message': message
                    })
            except Empty:
                continue
        
        return jsonify({'error': 'Prediction service timeout'}), 504
    
    except Exception as e:
        print(f"Error in api_predict: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000, debug=False)
    finally:
        if predictor_process:
            input_queue.put(None)
            predictor_process.terminate()
            predictor_process.wait()
