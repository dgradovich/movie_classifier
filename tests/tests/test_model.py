import sys
import unittest
import pandas as pd
import numpy as np
import logging
sys.path.append('../..')
sys.path.append('.')
from src.utils.load_save import load_obj
from src.utils.paths import get_models_path


# Sample input to predict
sample_input = """
The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.
"""

class ModelUnitTest(unittest.TestCase):

    def test(self):
        """
        Test Base Line model performance
        """
        # Make sure there are pre-trained models saved
        try:
            logging.info("Loading model...")
            model = load_obj(get_models_path('model.joblib'))
            logging.info("Loading labels map...")
            labeler = load_obj(get_models_path('labeler.joblib'))
            logging.info("Loading tfidf...")
            vectorizer = load_obj(get_models_path('tfidf.joblib'))
        except Exception as e:
            logging.error(e)
            raise e
        # Drama genre should be at least one of the predicted class
        tfidf_input = vectorizer.transform(pd.Series(sample_input))
        prediction = model.predict(tfidf_input)
        prediction = [i for i in labeler.classes_[np.asarray(prediction.todense()).astype('bool')[0]]]
        self.assertTrue('Drama' in prediction)
        # Output shouldn't be empty
        self.assertNotEqual(prediction, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)