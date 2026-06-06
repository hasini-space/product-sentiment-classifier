import pandas as pd
from text_cleaner import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def run_pipeline():
    print("📦 Loading dataset...")
    df = pd.read_csv("data/Womens_Clothing_E-Commerce_Reviews.csv")
    
    # Keep only the columns we need
    df = df[['Review Text', 'Rating']].dropna(subset=['Review Text'])
    
    print("🧹 Preprocessing text data (this may take a few moments)...")
    # Apply our cleaning script to every review
    df['Cleaned_Review'] = df['Review Text'].apply(clean_text)
    
    # Map ratings to binary sentiment: Stars 4-5 = Positive (1), Stars 1-2 = Negative (0)
    # Filter out neutral rating (3) to create a clean binary classification problem
    df = df[df['Rating'] != 3]
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x > 3 else 0)
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Review'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
    )
    
    print("🧮 Vectorizing text data using TF-IDF...")
    # Convert text to numeric vectors (limiting to top 5000 words for speed and memory efficiency)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("🏋️ Training Logistic Regression Classifier...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    print("\n📊 Evaluating Model Performance:")
    y_pred = model.predict(X_test_vec)
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Negative Sentiment', 'Positive Sentiment']))

if __name__ == "__main__":
    run_pipeline()
