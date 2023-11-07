
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

# Your existing code here

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

# Your existing code here

# Define your HTML form in an HTML template, e.g., index.html
# This form allows users to input a YouTube Video ID and submit it to /get_comments

# In your Flask app routes:

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_comments', methods=['POST'])
def get_comments():
    video_id = request.form['video_id']
    comments = get_video_comments(video_id)

    # Add emotion labels to comments
    emotions = [get_emotion_label(comment) for comment in comments]

    # Create a DataFrame with comments and emotion labels
    data = {'Comment': comments, 'Emotion': emotions}
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

from flask import Flask, request, render_template, jsonify
import joblib



# Load the saved SVM model and TF-IDF vectorizer
svm_model = joblib.load('emotion_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
# Define a dictionary to map numerical labels to emotions
emotion_mapping = {
    0: 'sad',
    1: 'happy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Modify the predict_emotion function to return the emotion
def predict_emotion(comment):
    comment_vector = vectorizer.transform([comment])
    predicted_label = svm_model.predict(comment_vector)[0]
    predicted_emotion = emotion_mapping.get(predicted_label, 'Unknown')
    return predicted_emotion

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion_endpoint():
    comment = request.form['comment']
    predicted_emotion = predict_emotion(comment)
    return render_template('index.html', predicted_emotion=predicted_emotion)


if __name__ == '__main__':
    app.run(debug=True)




