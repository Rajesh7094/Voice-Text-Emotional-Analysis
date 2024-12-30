import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import speech_recognition as sr
import time


# Function to load and preprocess the dataset
@st.cache_resource
def load_and_train_model():
    # Load dataset
    file_path = "D:\\Successed Project\\Voice-Emotional Analysis\\training data.csv"  # Update with your correct file path
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Rename columns for clarity
    df.columns = ['ID', 'Topic', 'Sentiment', 'Text']

    # Drop unnecessary columns
    df = df[['Sentiment', 'Text']]

    # Clean the data by dropping rows with missing values
    df.dropna(subset=['Sentiment', 'Text'], inplace=True)

    # Encode sentiment labels: Positive -> 1, Neutral -> 0, Negative -> -1
    df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

    # Check for any remaining NaN values in the Sentiment column after mapping
    df.dropna(subset=['Sentiment'], inplace=True)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

    # Convert text data into numerical form using TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, vectorizer


# Function to predict sentiment of a single tweet
def predict_sentiment(tweet, model, vectorizer):
    tweet_vec = vectorizer.transform([tweet])  # Transform the input tweet using the trained vectorizer
    prediction = model.predict(tweet_vec)  # Predict the sentiment
    sentiment = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}[prediction[0]]  # Map prediction to sentiment label
    return sentiment


# Function to convert live voice input to text with a 10-second limit
def live_voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Please speak clearly for up to 5 seconds.")
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Use a try-except block to catch timeout errors
        try:
            # Listen for audio input with a timeout of 10 seconds
            audio_data = recognizer.listen(source, timeout=10)  # 10 seconds timeout

            # Attempt to recognize the speech
            text = recognizer.recognize_google(audio_data)  # Convert speech to text
            st.success(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Error with the speech recognition service; {e}")
        except sr.WaitTimeoutError:
            st.warning("No speech detected within the time limit. Please try again.")

    return None


# Streamlit app
def main():
    st.title("Voice- Emotional  Analysis with Live Voice Input")
    st.write("Speak into your microphone, and the model will predict the sentiment (Positive, Neutral, Negative).")

    # Load the trained model and vectorizer
    model, vectorizer = load_and_train_model()

    # User input for the tweet
    user_input = st.text_area("Enter a text for sentiment prediction:")

    # If a tweet is entered, predict the sentiment
    if st.button("Predict Sentiment"):
        if user_input:
            sentiment = predict_sentiment(user_input, model, vectorizer)
            st.success(f"Predicted Sentiment from text: {sentiment}")
        else:
            st.warning("Please enter a text or use the live voice input below.")

    # Live voice input button
    if st.button("Record Live Voice for Sentiment Prediction"):
        voice_text = live_voice_to_text()  # Capture live voice input
        if voice_text:
            sentiment = predict_sentiment(voice_text, model, vectorizer)
            st.success(f"Predicted Sentiment from voice: {sentiment}")


# Run the app
if __name__ == '__main__':
    main()
