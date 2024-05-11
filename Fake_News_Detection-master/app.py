word_tokenize = 5
stopwords = 6
# from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import numpy as np

# app = Flask(__name__)
# tokenizer = Tokenizer()
# model = load_model('Fake_News_Detection-master\LSModel.h5')  # Replace with the correct filename

# # Assuming you have already defined the function for preprocessing text
# def preprocess_text(text):
#     # Perform any necessary text preprocessing here
#     # Tokenization, removing stop words, etc.
#     return text

# def fake_news_det(news):
#     preprocessed_news = preprocess_text(news)
#     sequences = tokenizer.texts_to_sequences([preprocessed_news])
#     padded_sequence = pad_sequences(sequences, maxlen=512)  # Adjust maxlen based on your model
#     prediction = model.predict(np.array(padded_sequence))
#     return prediction


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         message = request.form['message']
#         pred = fake_news_det(message)
#         print(pred)
#         return render_template('index.html', prediction=pred)
#     else:
#         return render_template('index.html', prediction="Something went wrong")

# if __name__ == '__main__':
#     app.run(debug=True)

import string

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
tokenizer = Tokenizer()
model = load_model('LSModel.h5')  # Replace with the correct filename



def preprocess_and_tokenize(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Combine tokens back into a preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text



def fake_news_det(news):
    preprocessed_news = preprocess_and_tokenize(news)
    sequences = tokenizer.texts_to_sequences([preprocessed_news])
    padded_sequence = pad_sequences(sequences, maxlen=512)  # Adjust maxlen based on your model
    prediction = model.predict(np.array(padded_sequence))
    return prediction[0][0]

@app.route('/')
def home():
    return render_template('index.html', prediction=None)  # Initialize prediction as None

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Check if the message is empty
        if not message:
            return render_template('index.html', prediction="Please enter a message")

        pred = fake_news_det(message)

        # Interpret the model's output directly
        prediction_label = 'REAL' if pred >= 0.5 else 'FAKE'
        
        # Print for debugging purposes
        print(f"Model Output: {pred}, Predicted Label: {prediction_label}")

        return render_template('index.html', prediction=prediction_label)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
