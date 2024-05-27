from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/result.html')
def result():
    return send_from_directory('../frontend', 'result.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    mobile_name = data['text']
    # Filter reviews by mobile name (for demo purposes, using a simple contains filter)
    reviews = df[df['product'].str.contains(mobile_name, case=False, na=False)]
    review_texts = reviews['body'].tolist()
    predictions = model.predict(review_texts)
    results = [{'text': text, 'sentiment': pred} for text, pred in zip(review_texts, predictions)]
    return jsonify(results)

@app.route('/wordcloud', methods=['POST'])
def wordcloud():
    data = request.json
    text = " ".join(review['text'] for review in data)
    wordcloud = WordCloud().generate(text)
    img = io.BytesIO()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/piechart', methods=['POST'])
def piechart():
    data = request.json
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for review in data:
        sentiments[review['sentiment']] += 1
    
    labels = sentiments.keys()
    sizes = sentiments.values()
    img = io.BytesIO()
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['green', 'red', 'blue'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    # Load datasets for filtering in analyze function
    df1 = pd.read_csv('20191226-items.csv')
    df2 = pd.read_csv('20191226-reviews.csv')
    df = pd.merge(df2, df1, on='asin')
    df.rename(columns={'title_x': 'review_title', 'title_y': 'product'}, inplace=True)
    app.run(debug=True)
