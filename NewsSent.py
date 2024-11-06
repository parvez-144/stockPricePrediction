import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib  # For saving and loading the model

# Load dataset
print("Loading dataset...")
data = pd.read_csv('sentime.csv')  # Replace with your file path
data = data[['Title', 'Words', 'Decisions']].dropna()
print("Dataset loaded. Number of entries:", len(data))

# Combine Title and Words into one text column
print("Combining Title and Words into a single text column...")
data['Text'] = data['Title'].astype(str) + " " + data['Words'].astype(str)

# Function to clean and normalize the 'Decisions' column
def parse_decision(value):
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) and "{" in value else value
        if isinstance(parsed, dict):
            parsed = next(iter(parsed.values()))
        return str(parsed).strip().lower()
    except (ValueError, SyntaxError):
        return np.nan

print("Normalizing 'Decisions' values...")
data['Decisions'] = data['Decisions'].apply(parse_decision)
unique_values = data['Decisions'].unique()
print("Unique values after normalization:", unique_values)

# Define expected classes and map them to integers
expected_classes = {"positive", "neutral", "negative"}
data = data[data['Decisions'].isin(expected_classes)]
label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
data['Encoded_Decisions'] = data['Decisions'].map(label_mapping)

if len(data) == 0:
    raise ValueError("No valid samples found after filtering. Please ensure the 'Decisions' column has correct labels.")

X = data['Text']
y = data['Encoded_Decisions']

# Split data into training and test sets
print("Splitting data into training and test sets...")
test_size = 0.2 if len(data) >= 10 else 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
print("Data split completed. Training size:", len(X_train), "Test size:", len(X_test))

# Vectorize text data
print("Vectorizing text data using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Text vectorization completed.")

# Train the SVM model
print("Training the SVM model...")
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
print("SVM model training completed.")

# Save the model and vectorizer as .pkl files
print("Saving the model and vectorizer...")
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved as 'svm_model.pkl' and 'tfidf_vectorizer.pkl'.")

# Predict and evaluate
print("Predicting using SVM model...")
y_pred = svm_model.predict(X_test_tfidf)

# Generate classification report
target_names = ["positive", "neutral", "negative"]
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# Save classification report to CSV
report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)).transpose()
report_df.to_csv("svm_classification_report.csv", index=True)
print("Classification report saved as CSV.")