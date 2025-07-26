# train_models.py

import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import zipfile # This module is crucial for reading from ZIP files


try:
    nltk.data.find('corpora/stopwords')
except (LookupError, Exception):
    print("NLTK stopwords not found. Attempting to download...")
    try:
        nltk.download('stopwords')
        print("NLTK stopwords downloaded.")
    except Exception as e:
        print(f"Failed to download stopwords automatically: {e}")
        print("Please manually run: python -m nltk.downloader stopwords")

try:
    nltk.data.find('corpora/wordnet')
except (LookupError, Exception):
    print("NLTK wordnet not found. Attempting to download...")
    try:
        nltk.download('wordnet')
        print("NLTK wordnet downloaded.")
    except Exception as e:
        print(f"Failed to download wordnet automatically: {e}")
        print("Please manually run: python -m nltk.downloader wordnet")


def clean_text(text):
    """
    Performs initial cleaning on the email text.
    - Removes HTML tags.
    - Removes URLs.
    - Converts to lowercase.
    - Removes punctuation.
    """
 
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenizes the text and applies lemmatization.
    Removes stopwords.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split() 
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

def preprocess_email_for_training(email_content):
    """
    Combines all preprocessing steps for a given email content during training.
    """
    cleaned = clean_text(email_content)
    processed = tokenize_and_lemmatize(cleaned)
    return processed


def train_and_save_models(dataset_path, csv_file_in_zip, text_column, label_column):
    """
    Loads dataset from a ZIP file, preprocesses, trains TF-IDF vectorizer and ML models,
    and saves them to disk.

    Args:
        dataset_path (str): Path to your zipped dataset file (e.g., "my_data.zip").
        csv_file_in_zip (str): The name of the actual CSV file *inside* the ZIP archive (e.g., "data.csv").
        text_column (str): The name of the column containing email text.
        label_column (str): The name of the column containing the binary label (0/1).
    """
    print(f"Loading dataset from: {dataset_path} (CSV file inside: {csv_file_in_zip})")
    try:
       
        with zipfile.ZipFile(dataset_path, 'r') as z:
            if csv_file_in_zip not in z.namelist():
                raise FileNotFoundError(f"'{csv_file_in_zip}' not found inside '{dataset_path}'. "
                                        f"Available files: {z.namelist()}")
            with z.open(csv_file_in_zip) as f:
                df = pd.read_csv(f)

        emails = df[text_column].fillna('').tolist() 
        labels = df[label_column].tolist()

        print(f"Dataset loaded. Total emails: {len(emails)}, Total labels: {len(labels)}")
        if len(emails) == 0:
            print("Error: Dataset is empty after loading. Please check your dataset path and column names.")
            return

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your dataset ZIP file is in the same directory as this script, or provide the full path.")
        print("Also, verify the name of the CSV file *inside* the ZIP archive.")
        print("For demonstration, creating a small dummy dataset.")
        # Create a small dummy dataset if the actual file is not found
        emails = [
            "Subject: Your account has been suspended. Click here to reactivate.", 
            "Subject: Invoice attached for Q2 earnings report.", 
            "Subject: Urgent security alert! Verify your password now.", 
            "Subject: Meeting reminder for tomorrow at 10 AM.", 
            "Subject: Congratulations, you've won a prize! Claim it here.", 
            "Subject: Project update and next steps.", 
            "Subject: Verify your bank details to avoid account closure. Link: http://scam.ru",
            "Subject: New policy document for review.", 
            "Subject: PayPal: Unusual activity detected on your account. Login immediately.", 
            "Subject: Your order #12345 has shipped." 
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        print("Using dummy dataset for training.")
    except KeyError as e:
        print(f"Error: Missing expected column in dataset: {e}")
        print(f"Please check 'text_column' ('{text_column}') and 'label_column' ('{label_column}') in the script.")
        return
    except zipfile.BadZipFile:
        print(f"Error: '{dataset_path}' is not a valid ZIP file. Please check the file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return

    print("Preprocessing emails...")
    processed_emails = [preprocess_email_for_training(email) for email in emails]
    print("Preprocessing complete.")

    X_train, X_test, y_train, y_test = train_test_split(
        processed_emails, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Data split: Training samples={len(X_train)}, Testing samples={len(X_test)}")

    print("Fitting TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
    X_test_vectorized = tfidf_vectorizer.transform(X_test)
    print(f"TF-IDF Vectorizer fitted. Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Training data vector shape: {X_train_vectorized.shape}")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True) 
    }

    best_model_name = None
    best_accuracy = 0

    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"{name} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print("  Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_ml_model = model # Keep track of the best model instance

    print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # --- 6. Save the trained TF-IDF Vectorizer and the Best Model ---
    print("\nSaving TF-IDF Vectorizer and best ML model...")
    try:
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            joblib.dump(tfidf_vectorizer, f)
        joblib.dump(best_ml_model, 'phishing_model.pkl')
        print("TF-IDF Vectorizer saved as 'tfidf_vectorizer.pkl'")
        print(f"Best model ({best_model_name}) saved as 'phishing_model.pkl'")
        print("Model training and saving complete!")
    except Exception as e:
        print(f"Error saving models: {e}")

if __name__ == "__main__":
    
    train_and_save_models(
        dataset_path="CEAS_08.csv.zip",
        csv_file_in_zip="CEAS_08.csv",
        text_column='body',          
        label_column='label'         
    )
