import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.datasets import imdb

# Load the word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained RNN model
model = load_model('rnn_model.h5')

# Helper Function: Decode the encoded review to human-readable text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Helper Function: Preprocess user input text
def preprocess_text(text):
    # Convert the text to lowercase and split into words
    words = text.lower().split()
    
    # Encode the words into integers using the word index
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Pad the sequence to ensure uniform length (500 in this case)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    return padded_review

# Function to Predict Sentiment with Multiple Categories
def predict_sentiment(review):
    # Preprocess the input review
    preprocessed_input = preprocess_text(review)
    
    # Predict the sentiment using the model
    prediction = model.predict(preprocessed_input)
    
    # Determine sentiment based on the prediction confidence score
    confidence = prediction[0][0]
    
    if confidence >= 0.8:
        sentiment = 'Very Positive'
    elif 0.6 <= confidence < 0.8:
        sentiment = 'Positive'
    elif 0.4 <= confidence < 0.6:
        sentiment = 'Neutral'
    elif 0.2 <= confidence < 0.4:
        sentiment = 'Negative'
    else:
        sentiment = 'Very Negative'
    
    return sentiment, confidence

# Streamlit App Interface
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify its sentiment. The sentiment categories range from 'Very Negative' to 'Very Positive'.")

# Input field for user to enter a movie review
user_input = st.text_area('Movie Review', height=150, placeholder='Type your review here...')

# Handle the submission of the review
if st.button('Submit'):
    if user_input.strip():
        # Predict the sentiment
        sentiment, confidence = predict_sentiment(user_input)
        
        # Display the results
        st.subheader(f'Sentiment: {sentiment}')
        st.write(f'Prediction Confidence: {confidence * 100:.2f}%')
    else:
        st.error('Please enter a valid movie review.')
else:
    st.write('Please enter a movie review and click the Submit button.')

## this is Sample Review decoded
st.write("\n### Example of Decoded Review:")
sample_encoded_review = [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3948, 4412, 232, 33, 130, 12, 16, 34, 685, 41, 12, 216, 125, 24, 308, 5, 4, 2, 2, 33, 6, 58, 15, 4, 249, 2, 5, 4, 2, 7, 2, 2, 7, 11, 2, 5, 2, 11, 20, 2, 2, 22, 17, 2, 2, 11, 5, 4, 2, 2, 13, 2, 2, 2, 19, 2, 5, 20, 4, 5, 2, 2, 7, 2, 2, 15, 2, 2, 7, 2, 2, 2, 2, 11, 2, 14, 2, 2, 2, 5, 4, 5, 16, 2, 2, 2, 2, 15, 2, 7, 2, 5, 2, 11, 2, 22, 5, 2, 5, 2, 4, 2, 4, 2, 11, 2, 19, 2, 2, 8, 4, 15, 2, 23]
st.write(decode_review(sample_encoded_review))