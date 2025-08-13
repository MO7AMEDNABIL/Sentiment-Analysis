from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load resources when starting the app
def load_resources():
    global tokenizer, model, sentiment_mapping, max_len
    
    # Load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load trained model
    model = load_model("sentiment_lstm.h5")
    
    # Define sentiment mapping (adjust according to your model)
    sentiment_mapping = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}
    
    # Set your max sequence length (adjust according to your model)
    max_len = 167

load_resources()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_text = request.form['text']
        
        # Process input
        new_seq = tokenizer.texts_to_sequences([user_text])
        new_pad = pad_sequences(new_seq, maxlen=max_len, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(new_pad)
        pred_label = prediction.argmax()
        
        # Reverse mapping
        reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
        
        # Format results
        sentiment = reverse_mapping[pred_label]
        confidence = f"{prediction[0][pred_label] * 100:.2f}%"
        
        return render_template('index.html', 
                             text=user_text, 
                             sentiment=sentiment, 
                             confidence=confidence)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)