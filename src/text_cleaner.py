import re
import nltk
from nltk.corpus import stopwords

# Download required NLTK resources safely
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text by removing punctuation, digits, converting to lowercase, 
    and removing stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize and remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in STOPWORDS]
    
    return " ".join(cleaned_words)

if __name__ == "__main__":
    test_review = "I absolutely LOVED this dress! It was beautifully made, but 10 times too big."
    print("Before:", test_review)
    print("After: ", clean_text(test_review))
