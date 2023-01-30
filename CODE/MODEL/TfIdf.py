import collections
from typing import Dict

class TF_IDF:

    word_frequency_dictionary : Dict[str, float]

    def compute_tf(self, words) -> Dict[str, float]: 
        """ 
        Takes in a list of words and creates a dictionary containing each words frequency. 
        """

        term_counts = collections.defaultdict(int)
        for word in words:
            term_counts[word] += 1
        total_count = sum(term_counts.values())
        tf = collections.defaultdict(int)
        tf.update({word: count/total_count for word, count in term_counts.items()})
        
        return tf

    def compute_term_frequencies(file_name, stemmer=None, stopwords=None):
        """
        Compute term frequencies after stemming and removal of stopwords.
        """
        with open(file_name) as file:
            text = file.read()
        words = remove_stop_words(normalize(text, stemmer=stemmer), stopwords=stopwords)
            
        return compute_tf(words)