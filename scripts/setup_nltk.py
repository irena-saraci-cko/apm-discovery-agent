import nltk
import ssl

def main():
    """
    Downloads the necessary NLTK data (stopwords and punkt)
    and handles potential SSL certificate issues.
    """
    print("Attempting to download NLTK data...")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    print("✅ NLTK data downloaded successfully.")

if __name__ == "__main__":
    main()
