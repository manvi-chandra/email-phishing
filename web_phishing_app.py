# web_phishing_app.py

import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os # For checking file existence

try:
    nltk.data.find('corpora/stopwords')
except (LookupError, Exception):
    with st.spinner("Downloading NLTK stopwords..."):
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            st.error(f"Failed to download stopwords automatically: {e}. Please run `python -m nltk.downloader stopwords` in your terminal.")

try:
    nltk.data.find('corpora/wordnet')
except (LookupError, Exception):
    with st.spinner("Downloading NLTK WordNet..."):
        try:
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            st.error(f"Failed to download wordnet automatically: {e}. Please run `python -m nltk.downloader wordnet` in your terminal.")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

def preprocess_email(email_content):
    cleaned = clean_text(email_content)
    processed = tokenize_and_lemmatize(cleaned)
    return processed

@st.cache_resource
def load_ml_components():
    vectorizer_path = 'tfidf_vectorizer.pkl'
    model_path = 'phishing_model.pkl'

    if not os.path.exists(vectorizer_path):
        st.error(f"Error: TF-IDF vectorizer file '{vectorizer_path}' not found.")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"Error: ML model file '{model_path}' not found.")
        return None, None

    try:
        tfidf_vectorizer_loaded = joblib.load(vectorizer_path)
        ml_model_loaded = joblib.load(model_path)
        return tfidf_vectorizer_loaded, ml_model_loaded
    except Exception as e:
        st.error(f"Failed to load ML components: {e}")
        return None, None

tfidf_vectorizer, ml_model = load_ml_components()


def predict_phishing(email_text, vectorizer, model):
    if vectorizer is None or model is None:
        st.warning("Machine learning components not loaded. Cannot predict.")
        return None, None

    processed_text = preprocess_email(email_text)
    email_vector = vectorizer.transform([processed_text])

    # Get the raw prediction from the model
    initial_prediction = model.predict(email_vector)[0]

    confidence = "N/A"
    final_prediction = initial_prediction # Start with the model's prediction

    if hasattr(model, 'predict_proba'):
        confidence_scores = model.predict_proba(email_vector)[0]
        # Confidence for the positive class (phishing, typically index 1)
        confidence = confidence_scores[1]

        # Apply the custom rule: if phishing score > 1.00%, classify as phishing
        if confidence * 100 > 1.00: # 1.00% threshold
            final_prediction = 1 # Force prediction to phishing (1)
    else:
        confidence = "N/A" # Model does not provide probabilities

    return final_prediction, confidence # Return the potentially adjusted prediction


st.set_page_config(
    page_title="Intelligent Phishing Email Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)


if 'page' not in st.session_state:
    st.session_state.page = 'start'

SHAKE_CSS = """
    <style>
    .shake-alert {
        animation: shake 0.5s;
        animation-iteration-count: 1;
    }
    @keyframes shake {
        0% { transform: translate(1px, 1px) rotate(0deg); }
        10% { transform: translate(-1px, -2px) rotate(-1deg); }
        20% { transform: translate(-3px, 0px) rotate(1deg); }
        30% { transform: translate(3px, 2px) rotate(0deg); }
        40% { transform: translate(1px, -1px) rotate(1deg); }
        50% { transform: translate(-1px, 2px) rotate(-1deg); }
        60% { transform: translate(-3px, 1px) rotate(0deg); }
        70% { transform: translate(3px, 1px) rotate(-1deg); }
        80% { transform: translate(-1px, -1px) rotate(1deg); }
        90% { transform: translate(1px, 2px) rotate(0deg); }
        100% { transform: translate(1px, -2px) rotate(-1deg); }
    }
    .phishing-score-text {
        font-size: 22px;
        color: white; /* Changed text color to white */
        background-color: #6c757d; /* Added grey background */
        padding: 10px 15px; /* Added padding for the box effect */
        border-radius: 8px; /* Rounded corners for the box */
        display: inline-block; /* Ensures background only covers text */
        margin-top: 15px; /* Add some space above */
        margin-bottom: 15px; /* Add some space below */
    }
    .magnifying-glass-icon {
        font-size: 2em; /* Adjust size as needed */
        margin-left: 10px; /* Space from the title */
        vertical-align: middle;
    }
    </style>
    """
st.markdown(SHAKE_CSS, unsafe_allow_html=True)


def start_page():
    """
    Displays the introductory start page.
    """
    st.title("Welcome to the Phishing Email Detection System")
    st.markdown("---")


    st.write("Protect your inbox from malicious emails with our intelligent detection system. Scan emails for threats like phishing, spam, and malware.")


    center_cols = st.columns([1, 3, 1])
    with center_cols[1]:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <a href="https://mail.google.com/" target="_blank">
                    <span style="font-size: 80px;">📧</span><br>
                    <p style="font-size: 1.2em; margin-top: 5px;">Your Inbox</p>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <p>Click the button below to start analyzing your email content.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Centered Start button
    start_button_col1, start_button_col2, start_button_col3 = st.columns([1, 1, 1])
    with start_button_col2:
        start_button_clicked = st.button("🚀 Start Analysis", use_container_width=True, key="start_button")

        if start_button_clicked:
            st.session_state.page = 'analyze'
            st.rerun()
def analyze_email_page():
    """
    Displays the email analysis page (your original app content).
    """
    st.title("🎣 Intelligent Phishing Email Detection System")
    st.markdown("---")


    st.markdown("<h1 style='display: inline-block;'>Email Analysis</h1> <span class='magnifying-glass-icon'>🔎</span>", unsafe_allow_html=True)
    st.markdown("---")


    if tfidf_vectorizer is None or ml_model is None:
        st.warning("Please ensure you have run `train_models.py` to generate `tfidf_vectorizer.pkl` and `phishing_model.pkl` in the same directory as this app.")
    else:
        st.write("Paste the content of an email below to check if it's legitimate or a phishing attempt.")

        email_content = st.text_area(
            "Enter Email Content Here:",
            value="Subject: Urgent Security Alert! Your account has been compromised.\n\nDear User,\n\nWe have detected suspicious login activity on your account. To prevent unauthorized access, please verify your account by clicking on the link below:\n\nhttp://secure-login-update.com/verify-now\n\nFailure to verify within 24 hours will result in permanent account suspension.\n\nThank you,\nSecurity Team",
            height=350,
            help="Paste the full email content, including subject and body."
        )

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("✨ Analyze Email ✨", use_container_width=True, key="analyze_button")

        st.markdown("---")

        if analyze_button:
            if email_content.strip() == "":
                st.warning("Please enter some email content to analyze.")
            else:
                with st.spinner('Analyzing...'):
                    prediction, confidence = predict_phishing(email_content, tfidf_vectorizer, ml_model)

                if prediction is not None:
                    if prediction == 0:
                        # Legitimate email: show success message with shake
                        st.markdown(
                            f"<div class='stAlert success shake-alert'><h2>✅ Classification: Legitimate Email</h2></div>",
                            unsafe_allow_html=True
                        )

                    else:
                        # PHISHING/SPAM: show error message with shake
                        st.markdown(
                            f"<div class='stAlert error shake-alert'><h2>🚨 PHISHING/SPAM Detected! 🚫</h2></div>",
                            unsafe_allow_html=True
                        )
                        st.warning("### This email exhibits characteristics of a phishing attempt. Exercise extreme caution!")


                    if isinstance(confidence, float):

                        st.markdown(
                            f"<div class='phishing-score-text'>Phishing Score: `{confidence * 100:.2f}%` (Higher score means more likely phishing)</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='phishing-score-text'>Phishing Score: '{confidence}' (Model does not provide probability scores)</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.write("Could not perform prediction. Please check the model loading status above.")

    st.markdown("---")
    st.caption("This application uses a machine learning model to help detect potential phishing emails. It is not foolproof and should be used as a supplementary tool.")


if st.session_state.page == 'start':
    start_page()
elif st.session_state.page == 'analyze':
    analyze_email_page()