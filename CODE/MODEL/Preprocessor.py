
class PreProcessor:
    """Takes in a text and preprocesses it."""

    def __init__(self):
        # Add some properties to the stemmer.

    def remove_stop_words(words, stopwords=None):
        if stopwords is None:
            stopwords = set(nltk.corpus.stopwords.words('english'))
        return list(filter(lambda x: x not in stopwords, words))
        
    
