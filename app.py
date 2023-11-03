from flask import Flask, render_template, request, send_file, jsonify
from googleapiclient.discovery import build
import pandas as pd
import io
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
nltk.download('vader_lexicon')

# Your YouTube Data API key
YOUTUBE_API_KEY = "AIzaSyB6yRPzPcZLqHcICH8KfOoF9iMKOmpg9Ec"  # Replace with your YouTube API key

# Function to fetch video comments using the YouTube Data API
def get_video_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []

    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat="plainText",
        maxResults=100  # Adjust the number of comments as needed
    )

    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)

    return comments

# Initialize the VADER Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to determine emotion labels
def get_emotion_label(comment):
    sentiment_scores = sia.polarity_scores(comment)

    # Determine the emotion label based on sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        return 'happy'
    elif sentiment_scores['compound'] <= -0.05:
        return 'sad'
    elif sentiment_scores['compound'] < 0 and sentiment_scores['compound'] >= -0.05:
        return 'angry'
    else:
        return 'surprise'

# Load the emotion prediction model
model = load_model('model.h5')

# Tokenization and padding for the comment input
max_words = 5000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_comments', methods=['POST'])
def get_comments():
    video_id = request.form['video_id']
    comments = get_video_comments(video_id)

    # Add emotion labels to comments and create a DataFrame
    data = {'Comment': comments, 'Emotion': [get_emotion_label(comment) for comment in comments]}
    df = pd.DataFrame(data)

    # Create a BytesIO object for sending the CSV as a download
    csv_io = io.BytesIO()
    df.to_csv(csv_io, index=False, encoding='utf-8')

    csv_io.seek(0)

    return send_file(
        csv_io,
        as_attachment=True,
        download_name="comments_with_emotion.csv",
        mimetype='text/csv'
    )

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # Get the comment from the input
    input_comment = request.form['input_comment']

    # Preprocess the input comment
    input_sequences = tokenizer.texts_to_sequences([input_comment])
    input_data = pad_sequences(input_sequences, maxlen=max_sequence_length)

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Map the prediction to an emotion label
    emotion_labels = ["surprise", "sad", "angry", "happy"]
    predicted_emotion = emotion_labels[prediction.argmax()]

    # Return the predicted emotion without quotes
    return predicted_emotion

if __name__ == '__main__':
    app.run(debug=True)
