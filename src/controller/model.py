import sys
sys.path.append('..')
sys.path.append('.')

import os
import logging
logging.root.setLevel(logging.NOTSET)

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import hamming_loss, make_scorer
from skmultilearn.problem_transform import LabelPowerset
import numpy as np

from src.utils.load_save import save_obj, find_default
from src.utils.paths import get_models_path


class BaseLineModel:
    def __init__(self):
        """
        A baseline model for movie classification
        """
        self.type = find_default('model')
        self.cv = 5
        self.score = make_scorer(hamming_loss, greater_is_better=False)
        if self.type == 'random_forest':
            self.parameter_search = {'classifier': [RandomForestClassifier(random_state=42)],
                                    'classifier__n_estimators': [10, 50, 100],
                                    'classifier__max_depth': [2, 5, 10]}
            self.default_parameters = {'classifier': [RandomForestClassifier(random_state=42)],
                                    'classifier__n_estimators': [10],
                                    'classifier__max_depth': [2]}
        else:
            raise NotImplementedError("Unknonw model type")
        
  
    def train_model(self,
        features,
        labels, 
        use_default_params = True):
        """
        Train a baseline model, which 
        either uses the deafult classifier parameters or
        uses grid search and cross validation to find the best
        hyper parameter values

        :param features: tfidf transformed feautures
        :param labels: binary vectorised labels
        :param use_params: Whether or not to use default variables

        :return: A trained classifier 
        """
        if use_default_params:
            try:
                logging.info('Training the model')
                clf = GridSearchCV(LabelPowerset(),
                    param_grid=self.default_parameters)
                clf.fit(features, labels)
                logging.info('Model trained with default parameters')
            except Exception as e:
                logging.error("Fitting default model has failed")
                raise e
        else: 
            # Could take a long time
            try:
                logging.info('Tuning the hyperparameters')
                clf = GridSearchCV(clf,
                    param_grid=self.parameter_search,
                    cv=self.cv)
                clf = clf.best_estimator_.fit(features, labels)
            except Exception as e:
                logging.error("Grid search for the best hyperparameters failed")
                raise e
            
        return clf

    def backup(self, model):
        """
        Save model object locally

        :param: Trained model object
        """
        save_obj(model, get_models_path('model.joblib'))

    def evaluate_model(self, model, features, labels):
        """
        Evaluate model using 5-fold cross validation and Hamming loss

        :param model: trained classifier
        :param features: tfidf transformed feautures
        :param labels: binary vectorised labels
        """
        logging.info(f"Average 5 fold cross validation score using Hamming Loss is \
         {-np.mean(cross_val_score(model, features, labels, cv=self.cv, scoring=self.score))}")

    def train_evaluate_save(self, features, labels, evaluate=False):
        """
        Train, evaluate and save the classifier

        :param features: tfidf transformed feautures
        :param labels: binary vectorised labels

        :return: A trained classifier 
        """
        model = self.train_model(features, labels)
        
        self.evaluate_model(model, features, labels)
        self.backup(model)
        return model
