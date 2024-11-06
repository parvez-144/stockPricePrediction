import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Define stopwords manually (or use a common stopword list)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
    't', 'can', 'will', 'just', 'don', 'should', 'now'
])

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove non-alphabetic characters using regex
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text and remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    # Return the processed text as a string
    return " ".join(words)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Apply preprocessing to the 'processed' column
    data['processed'] = data['processed'].apply(preprocess_text)
    
    # Split the data into features (X) and target (y)
    X = data['processed']
    y = data['senti_label']
    
    return X, y

# Vectorize the text data using TF-IDF
def vectorize_text(X):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    return X_tfidf, vectorizer

# Train a Logistic Regression model
def train_logistic_regression(X_train, y_train):
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    return log_reg_model

# Train a Random Forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Train an SVM model
def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    return y_pred

# Save the model and predictions to CSV
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def save_predictions_to_csv(X_test, y_test, log_reg_pred, rf_pred, svm_pred, filename):
    results_df = pd.DataFrame({
        'Text': X_test,
        'True_Label': y_test,
        'Logistic_Regression_Pred': log_reg_pred,
        'Random_Forest_Pred': rf_pred,
        'SVM_Pred': svm_pred
    })
    results_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

# Main function to train and evaluate multiple models
def main():
    # Load and preprocess data
    file_path = 'train_stockemo.csv'  # Replace with the path to your dataset
    X, y = load_and_preprocess_data(file_path)
    
    # Vectorize the text data
    X_tfidf, vectorizer = vectorize_text(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression model
    print("Training Logistic Regression...")
    log_reg_model = train_logistic_regression(X_train, y_train)
    
    # Train Random Forest model
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    # Train SVM model
    print("\nTraining SVM...")
    svm_model = train_svm(X_train, y_train)
    
    # Evaluate models
    print("Evaluating Logistic Regression...")
    log_reg_pred = evaluate_model(log_reg_model, X_test, y_test)
    
    print("\nEvaluating Random Forest...")
    rf_pred = evaluate_model(rf_model, X_test, y_test)
    
    print("\nEvaluating SVM...")
    svm_pred = evaluate_model(svm_model, X_test, y_test)
    
    # Save predictions to CSV
    save_predictions_to_csv(X_test, y_test, log_reg_pred, rf_pred, svm_pred, 'model_predictions.csv')

    # Save the models and vectorizer
    save_model(log_reg_model, 'log_reg_senti_model.pkl')  # Save Logistic Regression model
    save_model(rf_model, 'random_forest_senti_model.pkl')  # Save Random Forest model
    save_model(svm_model, 'svm_senti_model.pkl')  # Save SVM model
    save_model(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer for future use

if _name_ == "_main_":
    main()