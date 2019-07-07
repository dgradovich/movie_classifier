from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords


class LemmaTokenizer:
    """
    """
    def __init__(self):
        """
        Tokenzier class
        """
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokens =  [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        return [i for i in tokens if len(i)>1]

def get_stopwords_punct():
    """
    Generate a set of stopwords and punctiation symbols
    to be excluded from the main vocabulary.

    :return: set of stopwords and punctuations symbols
    """
    stop = set(stopwords.words('english'))
    punct = set(punctuation)
    return stop.union(punct)
