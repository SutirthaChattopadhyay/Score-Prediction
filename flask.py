import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load your machine learning model
model = pickle.load(open('crck_model2.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        prediction = model.predict([features])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
