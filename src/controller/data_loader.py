import sys
sys.path.append('..')
sys.path.append('.')

import logging
logging.root.setLevel(logging.NOTSET)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import ast
from src.utils.paths import get_data_path, get_models_path
from src.utils.words import LemmaTokenizer, get_stopwords_punct  
from src.utils.load_save import save_obj
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


class DataLoader:

    def __init__(self):

        self.data_file_name = 'movies_metadata.csv'
        self.columns = ['overview', 'genres', 'original_title']
        self.selected_genres = ['Action','Adventure', 'Animation',
                                'Comedy','Crime', 'Documentary', 'Drama',
                                'Family','Fantasy', 'Foreign',
                                'History','Horror',
                                'Music', 'Mystery',
                                'Romance','Science Fiction',
                                'Thriller','War', 'Western']


    def load_raw_data(self):
        """
        Load csv file from the deafault data folder

        :return: Raw dataframe
        """
        data_path = get_data_path(self.data_file_name)

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            logging.error(f'Loading data failed - check filename {data_path}')
            raise e

        if [i for i in self.columns if i in df.columns]\
            != self.columns:
            raise KeyError("Unknown columns in the data")

        df = self.remove_missing_obs(df)
        return df[self.columns]

    def remove_missing_obs(self, df):
        """
        Remove samples with missing description

        :param df: Raw data

        :return: Data without missing descriptions
        """
        df = df[~df['overview'].isna()]
        return df

    def remove_empty_labels(self, df, genres):
        """
        Remove samples with missing labels

        :param df: Raw data
        :param genres: Nested list of genre names

        :return: Data without missing labels
        """

        df = df[genres.apply(lambda x: x!=[])]
        genres = genres[genres.apply(lambda x: x!=[])]

        return df, genres

    def parse_genres(self, genres):
        """
        Parse genres from dictionaries to lists of genre names

        :param genres: Original genre format

        :return: Nested list of genre names
        """
        genres = ast.literal_eval(genres)
        return [genre['name'] for genre in genres if genre['name'] in self.selected_genres]

    def multi_label_binarise(self, genres_list):
        """
        Creaty dummy variables for each label

        :param genres: Nested list of genres

        :return: Genres represented as dummy varaibles
        :return: Dummy variable encoder object
        """
        if len(genres_list) == 0:  
            raise ValueError("No genres are found")

        labeler = MultiLabelBinarizer()
        labeler.fit(genres_list)
        labels = labeler.transform(genres_list)
        return labels, labeler


    def tfidf_vectorise(self, features):
        """
        Create tf-idf representation of the text features

        :param features: text features

        :return: Tf-idf representation of the text features
        :return: Tf-idf encoder object
        """
        if len(features) == 0:
            raise ValueError("No features found")
        stop_punct = get_stopwords_punct()
        vect = TfidfVectorizer(tokenizer=LemmaTokenizer(),
            stop_words=stop_punct, 
            min_df = 100) # remove infrequent words from vocabulary
        vect.fit(features)
        features = vect.transform(features)
        return features, vect
            

    def backup(self, labeler, tfidf):
        """
        Save labeler and vetorizer objects locally

        :param: Labeler encoder
        :param: Tfidf encoder
        """
    
        save_obj(labeler, get_models_path('labeler.joblib'))
        save_obj(tfidf, get_models_path('tfidf.joblib'))

    def load_preprocess(self):
        """
        Load and preprocess data

        :return: Tf-idf representation of the text features
        :return: Tf-idf encoder object
        :return: Genres represented as dummy varaibles
        :return: Dummy variable encoder object
        """ 
        logging.info('Loading raw data...')
        df = self.load_raw_data()
        

        genres_list = df['genres'].apply(self.parse_genres)
        logging.info('Processing genres...')

        logging.info('Removing empty labels...')
        df, genres_list = self.remove_empty_labels(df, genres_list)
        

        logging.info('Binarizing labels...')
        labels, labeler = self.multi_label_binarise(genres_list)
        

        features = df['original_title'] + ' ' + df['overview']
        logging.info('Generating tfidf features...')
        features, vectorizer = self.tfidf_vectorise(features)
        

        if features.shape[0] != len(labels):
            raise ValueError('The dimensions of labels and features do not match')

        logging.info('Saving tfidf and labeler objects...')
        self.backup(labeler=labeler, tfidf=vectorizer)
        
        return features, vectorizer, labels, labeler


if __name__ == '__main__':
    dl = DataLoader()
    features, vectorizer, labels, labeler = dl.load_preprocess()

    print(features.shape)
    print(labels.shape)