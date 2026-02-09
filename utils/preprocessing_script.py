import re
import emoji # type: ignore
import nltk
import html
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text, full=False):
    # 1. Basic Cleaning (Good for BERT)
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+', '[URL]', text)  # Replace URLs
    text = html.unescape(text) # Convert escaped characters back to original (&quot; -> ")
    
    # 2. Handle Emojis (Convert to text descriptions)
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # 3. 'Full' Processing (For Logistic Regression / Traditional ML)
    if full:
        # Define allowed punctuation: keep ? and !
        # This regex removes punctuation EXCEPT ? and !
        text = re.sub(r'[^\w\s\?\!]', '', text)
        
        # Tokenize for processing
        words = text.split()
        
        # Initialize Stopwords and Lemmatizer
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Filter stopwords and Lemmatize
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        
        text = " ".join(words)
    
    # 4. Final Cleanup of extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text