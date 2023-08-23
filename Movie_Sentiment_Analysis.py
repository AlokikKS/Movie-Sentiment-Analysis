# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load the dataset
data = pd.read_csv('movie_reviews.csv')  # Replace with your dataset file

# Preprocessing: Cleaning and Label Encoding
# (Assuming your dataset has a 'text' column for reviews and a 'label' column for sentiment)
data['text'] = data['text'].apply(lambda x: x.lower())  # Convert text to lowercase

# Split data into features (X) and labels (y)
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction: Bag of Words
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Selection and Training: Naive Bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Model Evaluation
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Predictions
new_review = ["This movie was fantastic! I loved every moment of it."]
new_review_vectorized = vectorizer.transform(new_review)
prediction = model.predict(new_review_vectorized)

if prediction[0] == 1:
    print("Predicted sentiment: Positive")
else:
    print("Predicted sentiment: Negative")
