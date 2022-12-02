import nltk
import warnings
import pandas as pd
import numpy as np
from pyparsing import OneOrMore

import sklearn.metrics as mtr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow
import keras
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer


class Engine:
    def __init__(self):
        self.model = keras.models.load_model(
            '/Users/fabio/Documents/Progetto IR/models/RNNLSTM')

        self.models = {
            'Logistic Regression': LogisticRegression(),
            'MLP': MLPClassifier(random_state=1, max_iter=500, n_iter_no_change=5),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'RNN LSTM': self.model
        }
        self.cross_val = {}
        self.predictions = {}

    def fit(self, X, y):
        for model_name, model in self.models.items():
            if (model_name != 'RNN LSTM'):
                print("Training: {}".format(model_name))
                model.fit(X, y)

    def _from_categorical(self, vec):
        new_vec = []
        for v in vec:
            max = v[0]
            index = 0
            for i, val in enumerate(v):
                if (val > max):
                    max = val
                    index = i
            new_vec.append(index)
        return new_vec

    def predict(self, X, y, y_k):
        self.predictions = {}
        for model_name, model in self.models.items():
            print("Predicting: {}".format(model_name))
            if (model_name != 'RNN LSTM'):
                self.predictions[model_name] = model.predict(X)
                self.cross_val[model_name] = cross_val_score(model, X, y, cv=5)
            else:
                self.predictions[model_name] = self._from_categorical(
                    model.predict(X))
                self.cross_val[model_name] = model.evaluate(X, y_k)

        E = []
        for estimator, y_pred in self.predictions.items():
            report = mtr.classification_report(
                y, y_pred, output_dict=True, zero_division=0)

            E.append({
                'Model': estimator, 'Accuracy': report['accuracy'],
                'Avg Precision (macro)': report['macro avg']['precision'],
                'Avg Recall (macro)': report['macro avg']['recall'],
                'Avg F1-score (macro)': report['macro avg']['f1-score'],
                'Avg Precision (weighted)': report['weighted avg']['precision'],
                'Avg Recall (weighted)': report['weighted avg']['recall'],
                'Avg F1-score (weighted)': report['weighted avg']['f1-score']
            })
        E = pd.DataFrame(E).set_index('Model', inplace=False)

        return E
