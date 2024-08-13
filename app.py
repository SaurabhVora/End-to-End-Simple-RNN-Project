import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.datasets import imdb

# Load the word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained RNN model
model = load_model('my_model.keras')

# Helper Function: Decode the encoded review to human-readable text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Helper Function: Preprocess user input text
# Helper Function: Preprocess user input text
def preprocess_text(text):
    # Convert the text to lowercase and split into words
    words = text.lower().split()
    
    # Encode the words into integers using the word index
    encoded_review = []
    for word in words:
        # Get the index of the word or assign 2 (index for unknown words)
        index = word_index.get(word, 2) + 3
        if index >= 10000:
            index = 2  # Replace out-of-bounds index with index for unknown words
        encoded_review.append(index)
    
    # Pad the sequence to ensure uniform length (500 in this case)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    return padded_review


# Function to Predict Sentiment with Multiple Categories
def predict_sentiment(review):
    # Preprocess the input review
    preprocessed_input = preprocess_text(review)
    
    # Ensure the preprocessed input is a numpy array with the correct dtype
    preprocessed_input = np.array(preprocessed_input, dtype='int32')
    
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
        try:
            # Predict the sentiment
            sentiment, confidence = predict_sentiment(user_input)
            
            # Display the results
            st.subheader(f'Sentiment: {sentiment}')
            st.write(f'Prediction Confidence: {confidence * 100:.2f}%')
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error('Please enter a valid movie review.')
else:
    st.write('Please enter a movie review and click the Submit button.')