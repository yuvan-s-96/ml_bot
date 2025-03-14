# FAQ Chatbot with Flask and Decision Tree Classifier

## Overview
This project is a Flask-based chatbot that provides answers to frequently asked questions (FAQs) related to banking. It uses machine learning techniques such as TF-IDF vectorization and a Decision Tree classifier to categorize user queries and retrieve relevant responses based on cosine similarity.

## Features
- Uses a **Decision Tree Classifier** to classify user queries into predefined categories.
- Implements **TF-IDF Vectorization** for text processing.
- Computes **Cosine Similarity** to retrieve the most relevant answer.
- Saves and loads trained models using **joblib** for efficiency.
- Provides a simple **Flask API** to handle user queries.

## Installation

### Prerequisites
Ensure you have Python installed (>=3.7) and the following dependencies:

```
pip install flask pandas numpy scikit-learn joblib
```

### Clone the Repository
```
git clone https://github.com/ml_bot.git
cd faq-bot
```

## Dataset
The chatbot uses a CSV file (`BankFAQs.csv`) that contains banking-related questions and their corresponding answers. The CSV file should have the following columns:

- **Question**: The user question.
- **Answer**: The corresponding response.
- **Class**: The category of the question.

## Running the Application
1. Place your dataset in the appropriate directory and update the file path in `app.py`.
2. Train and save the model by running:
   ```
   python app.py
   ```
3. Start the Flask server:
   ```
   python app.py
   ```
4. Access the web interface at: `http://127.0.0.1:5000/`

## API Endpoint
### `/api/get_response`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "user_input": "How do I reset my password?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "You can reset your password by visiting the official banking portal and selecting 'Forgot Password'."
  }
  ```

## File Structure
```
faq-bot/
│── app.py
│── templates/
│   └── index.html
│── static/
│── models/
│   ├── decision_tree_model.joblib
│   ├── tfidf_vectorizer.joblib
│── BankFAQs.csv
│── README.md
```

## Future Enhancements
- Improve the accuracy using **more advanced models** like Random Forest or Deep Learning.
- Implement **Named Entity Recognition (NER)** to extract specific entities from queries.
- Deploy the chatbot as a **REST API** or integrate it with a **messaging platform**.



