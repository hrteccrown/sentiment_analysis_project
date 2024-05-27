import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Determine the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load datasets
df1 = pd.read_csv(os.path.join(script_dir, '20191226-items.csv'))
df2 = pd.read_csv(os.path.join(script_dir, '20191226-reviews.csv'))

# Merge datasets
df = pd.merge(df2, df1, on='asin')
df.rename(columns={'title_x': 'review_title', 'title_y': 'product'}, inplace=True)

# Fill missing values
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
for col in non_numeric_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

# Convert date column
df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y')

# Define sentiment
df['Sentiment'] = df['rating_x'].apply(lambda x: 'Positive' if x > 3 else ('Negative' if x < 3 else 'Neutral'))

# Split data
X = df['body']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, 'sentiment_model.pkl')

# Generate word cloud for demonstration
text = " ".join(review for review in df.body)
wordcloud = WordCloud(max_words=100, background_color="white").generate(text)
wordcloud.to_file("wordcloud.png")

