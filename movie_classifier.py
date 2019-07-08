import sys
import os
import pandas as pd
import numpy as np

import logging
logging.root.setLevel(logging.NOTSET)

import argparse
from src.controller.data_loader import DataLoader
from src.controller.model import BaseLineModel
from src.utils.load_save import load_obj, save_json
from src.utils.paths import get_models_path, get_output_path

parser=argparse.ArgumentParser()

parser.add_argument('--title', help='The movie title', type=str)
parser.add_argument('--description', help='The movie description', type=str)
parser.add_argument('--retrain', help='Retrain the current model', type=bool)


try:
    args = parser.parse_args()
except Exception as e:
    logging.error("Cannot parse arguments")
    logging.error(e)


def check_inputs(args):
    """
    Check the arguments to see if both title and description are present

    :param args: arguments
    """
    if not (args.title and args.description):
        logging.error(f"{args} are not correct arguments")
        raise Exception('The inputs are not correct or missing')

    return args.title + ' ' + args.description


def save_output(predictions):
    """
    Save the most recent prediction
    
    :param predictions:
    """

    output = {"title": args.title,
            "description": args.description,
            "genre": predictions}
    
    logging.info(output)
    logging.info('Saving output...')
    save_json(output,
        get_output_path('output.json'))


def retrain():
    """
    Re-trian the baseline model

    :return: model
    :return: labeler
    :return: vectorizer
    """
    dl = DataLoader()
    base_line_model = BaseLineModel()
    features, vectorizer, labels, labeler = dl.load_preprocess()
    model = base_line_model.train_evaluate_save(features, labels)
    return model, labeler, vectorizer


def load_existing():
    """
    Load existing model, labeler map and tfidf matrix

    :return: model
    :return: labeler
    :return: vectorizer
    """
    
    try:
        logging.info("Loading model...")
        model = load_obj(get_models_path('model.joblib'))
        logging.info("Loading labels map...")
        labeler = load_obj(get_models_path('labeler.joblib'))
        logging.info("Loading tfidf...")
        vectorizer = load_obj(get_models_path('tfidf.joblib'))
    except Exception as e:
        logging.error(e)
        logging.warning("Cannot load the model, labeler map and tfidf matrix")
        model, labeler, vectorizer = retrain()

    return model, labeler, vectorizer


def make_prediction(inputs,
    tfidf,
    model,
    labeler):
    """
    Make a prediction based on user input

    :return: model
    :return: labeler
    :return: vectorizer
    """
    tfidf_input = tfidf.transform(pd.Series(inputs))
    prediction = model.predict(tfidf_input)
    return ' '.join([i for i in labeler.classes_[np.asarray(prediction.todense()).astype('bool')[0]]])


if __name__ == "__main__":
    inputs = check_inputs(args)
    if args.retrain:
        # retrain model
        model, labeler, vectorizer = retrain()
    else:
        model, labeler, vectorizer = load_existing()
    prediction = make_prediction(inputs, vectorizer, model, labeler)
    if len(prediction) > 0:
        save_output(prediction)
    else:
        logging.error("Cannot make a prediction. Please try again with more detailed description or retrain the model.")

