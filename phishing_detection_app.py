# phishing_detection_app.py

import tkinter as tk
from tkinter import messagebox, scrolledtext
import re
import string
import pickle
import joblib # Recommended for scikit-learn models
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# You might need to download these if you haven't already:
# nltk.download('stopwords')
# nltk.download('wordnet')

# --- Global Variables for ML Components (Loaded on app start) ---
# These will hold your trained TF-IDF vectorizer and ML model.
# In a real application, you'd load these from files.
tfidf_vectorizer = None
ml_model = None # e.g., Naive Bayes, Logistic Regression, RandomForest, SVM

# --- Text Preprocessing Functions ---
def clean_text(text):
    """
    Performs initial cleaning on the email text.
    - Removes HTML tags.
    - Removes URLs.
    - Converts to lowercase.
    - Removes punctuation.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenizes the text and applies lemmatization.
    Removes stopwords.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split() # Simple split, consider nltk.word_tokenize for better tokenization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

def preprocess_email(email_content):
    """
    Combines all preprocessing steps for a given email content.
    """
    cleaned = clean_text(email_content)
    processed = tokenize_and_lemmatize(cleaned)
    return processed

# --- Model Loading Function ---
def load_ml_components():
    """
    Loads the pre-trained TF-IDF vectorizer and the machine learning model.
    These files would be generated during the model training phase.
    """
    global tfidf_vectorizer, ml_model
    try:
        # Replace 'tfidf_vectorizer.pkl' and 'phishing_model.pkl' with your actual file paths
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('phishing_model.pkl', 'rb') as f:
            ml_model = joblib.load(f) # Use joblib for scikit-learn models
        print("ML components loaded successfully.")
        return True
    except FileNotFoundError:
        messagebox.showerror("Error", "Model files (tfidf_vectorizer.pkl, phishing_model.pkl) not found.\n"
                                      "Please ensure you have trained and saved your models.")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load ML components: {e}")
        return False

# --- Prediction Function ---
def predict_phishing(email_text):
    """
    Takes raw email text, preprocesses it, and uses the loaded model to predict.
    Returns the prediction (0 or 1) and confidence score.
    """
    if tfidf_vectorizer is None or ml_model is None:
        messagebox.showwarning("Warning", "Machine learning components not loaded. Cannot predict.")
        return None, None

    # Preprocess the input email
    processed_text = preprocess_email(email_text)

    # Transform text using the loaded TF-IDF vectorizer
    # Note: use .transform() for new data, not .fit_transform()
    email_vector = tfidf_vectorizer.transform([processed_text])

    # Make prediction
    prediction = ml_model.predict(email_vector)[0]

    # Get confidence score (probability)
    # Check if the model has predict_proba method
    if hasattr(ml_model, 'predict_proba'):
        confidence_scores = ml_model.predict_proba(email_vector)[0]
        # For binary classification, confidence for class 1 (phishing)
        confidence = confidence_scores[1]
    else:
        # Some models (like SVM with default settings) don't have predict_proba
        confidence = "N/A" # Or implement a custom confidence measure if needed

    return prediction, confidence

# --- Tkinter GUI Functions ---
def analyze_email():
    """
    Handles the 'Analyze Email' button click event.
    Gets email content, calls prediction, and displays results.
    """
    email_content = email_text_area.get("1.0", tk.END).strip()

    if not email_content:
        messagebox.showwarning("Input Error", "Please enter email content to analyze.")
        return

    prediction, confidence = predict_phishing(email_content)

    if prediction is not None:
        result_text = "Legitimate" if prediction == 0 else "PHISHING/SPAM"
        result_color = "green" if prediction == 0 else "red"
        confidence_str = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)

        result_label.config(text=f"Classification: {result_text}", fg=result_color)
        confidence_label.config(text=f"Confidence: {confidence_str}")
    else:
        result_label.config(text="Classification: N/A", fg="black")
        confidence_label.config(text="Confidence: N/A")

def create_phishing_detection_app():
    """
    Creates and runs the Tkinter GUI for the phishing detection system.
    """
    global email_text_area, result_label, confidence_label

    root = tk.Tk()
    root.title("Intelligent Phishing Email Detection System")
    root.geometry("800x600") # Larger initial size
    root.resizable(True, True) # Allow resizing

    # --- Load ML components when the app starts ---
    if not load_ml_components():
        # If loading fails, disable prediction functionality
        messagebox.showerror("Setup Error", "Application cannot proceed without ML models. Exiting.")
        root.destroy()
        return

    # --- GUI Layout ---

    # Frame for input
    input_frame = tk.LabelFrame(root, text="Enter Email Content", padx=10, pady=10)
    input_frame.pack(padx=10, pady=10, fill="both", expand=True)

    email_text_area = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=80, height=15,
                                                font=("Arial", 10), bd=2, relief=tk.SUNKEN)
    email_text_area.pack(padx=5, pady=5, fill="both", expand=True)
    email_text_area.insert(tk.END, "Subject: Urgent Security Update\n\nDear customer,\n\nWe have detected unusual activity on your account. Please click the link below to verify your details immediately:\n\nhttp://malicious-site.com/login\n\nFailure to do so will result in account suspension.\n\nSincerely,\nYour Bank Security Team")

    # Frame for buttons and results
    control_frame = tk.Frame(root, padx=10, pady=10)
    control_frame.pack(padx=10, pady=10, fill="x")

    analyze_button = tk.Button(control_frame, text="Analyze Email", command=analyze_email,
                               bg="#007BFF", fg="white", font=("Arial", 12, "bold"),
                               activebackground="#0056b3", activeforeground="white",
                               relief=tk.RAISED, bd=3, padx=15, pady=8)
    analyze_button.pack(side=tk.LEFT, padx=10)

    # Result Labels
    result_label = tk.Label(control_frame, text="Classification: N/A", font=("Arial", 14, "bold"))
    result_label.pack(side=tk.LEFT, padx=20)

    confidence_label = tk.Label(control_frame, text="Confidence: N/A", font=("Arial", 12))
    confidence_label.pack(side=tk.LEFT, padx=10)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    # --- IMPORTANT: Placeholder for creating dummy model files ---
    # In a real scenario, you would train your models on your dataset
    # and save them to these files. This is just to prevent FileNotFoundError
    # when you first try to run this conceptual app without actual models.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        import numpy as np

        # Create dummy vectorizer
        dummy_corpus = ["legitimate email example", "phishing scam warning", "click this link now"]
        dummy_vectorizer = TfidfVectorizer()
        dummy_vectorizer.fit(dummy_corpus)
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(dummy_vectorizer, f)

        # Create dummy model
        dummy_X = dummy_vectorizer.transform(dummy_corpus)
        dummy_y = np.array([0, 1, 1]) # 0=legitimate, 1=phishing
        dummy_model = MultinomialNB()
        dummy_model.fit(dummy_X, dummy_y)
        joblib.dump(dummy_model, 'phishing_model.pkl')
        print("Dummy model files created for demonstration.")
    except ImportError:
        print("Scikit-learn not found. Please install it (`pip install scikit-learn`) to generate dummy models.")
        print("You will need to manually create 'tfidf_vectorizer.pkl' and 'phishing_model.pkl' if you proceed.")
    except Exception as e:
        print(f"Could not create dummy model files: {e}")
        print("Please ensure you have scikit-learn and joblib installed.")
    # --- End of dummy model creation ---

    create_phishing_detection_app()