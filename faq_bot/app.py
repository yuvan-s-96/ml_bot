from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Read data from CSV
df = pd.read_csv(r"C:\Users\yuvan\Downloads\Faq\faq_bot\BankFAQs (1).csv")

# Separate features (X) and labels (y)
X = df["Question"]
y = df["Class"]

# Convert labels to numerical format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# Preprocess text data
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, encoded_labels, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the model and vectorizer using joblib
joblib.dump(clf, 'decision_tree_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('user_input')

    # Load the saved model and vectorizer
    clf = joblib.load('decision_tree_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    # Transform the input question using the TF-IDF vectorizer
    new_X = vectorizer.transform([user_input])

    # Predict the class using the trained decision tree
    predicted_class = clf.predict(new_X)[0]

    # Get subset of questions belonging to the predicted class
    subset_questions = df[df['Class'] == label_encoder.inverse_transform([predicted_class])[0]]["Question"]

    # Calculate cosine similarity
    similarities = [cosine_similarity(new_X, vectorizer.transform([subset_question]))[0][0] for subset_question in subset_questions]

    # Find the question with the maximum cosine similarity
    max_similarity_index = np.argmax(similarities)
    max_similarity_question = subset_questions.iloc[max_similarity_index]

    # Get the corresponding answer
    answer = df[df['Question'] == max_similarity_question]['Answer'].values[0]

    response = {
        "answer": answer,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
