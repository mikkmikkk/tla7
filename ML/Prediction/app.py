from flask import Flask, request, render_template, jsonify
import pickle
import re

app = Flask(__name__)


with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the POST request
        text = request.json['text']
        
        # Preprocess the text (you may need to adjust this depending on how your model was trained)
        processed_text = preprocess_text(text)
        
        # Vectorize the processed text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        
        # Return the prediction as JSON
        return jsonify({'emotion': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def preprocess_text(text):
    
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

if __name__ == '__main__':
    app.run(debug=True)
