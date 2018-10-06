"""
Module defining class mapping similarity vector to similarity score.
The score is our measure for whether a plan is a match to the input string.
"""

import numpy as np
import pickle

MODEL_FILENAME = '../Models/logreg_default.sav'

def inv_sigmoid(x):
    return np.log(x/(1. - x))

def construct_features_targets(X_match_in, X_nomatch_in, n_match_out, n_nomatch_out):
    n_match_in = X_match_in.shape[0]
    indices_match = np.random.choice(range(n_match_in), size=n_match_out) # RdP
    X = X_match_in[indices_match, :]
    n_nomatch_in = X_nomatch_in.shape[0]
    indices_nomatch = np.random.choice(range(n_nomatch_in), size=n_nomatch_out)
    X = np.concatenate((X, X_nomatch_in[indices_nomatch,:]), axis=0)
    Y = np.concatenate(( np.ones((n_match_out)), np.zeros((n_nomatch_out)) ))
    indices_out = np.random.permutation(range(n_match_out + n_nomatch_out))
    return X[indices_out, :], Y[indices_out]

class SimilarityVectorToScore():
    """Object that does the mapping from similarity vector to score.

    The workhorse is the method compute_score. It takes the matching probability
    predicted by the classification model (default: log reg) and transforms it using
    an inverse sigmoid. This gives the similarity score based on which it is decided
    if an input-plan pair is a match.

    Important attributes:
        model (sklearn model): model with a predict_proba method.
        compute_score (function): function mapping sim vec to score.
    """

    def set_model(self, model):
        self.model = model
        self.compute_score = lambda x: inv_sigmoid(self.model.predict_proba(x.reshape(1,-1))[0,1])

    def load_model(self, filename = MODEL_FILENAME):
        self.model = pickle.load(open(filename, 'rb'))
        self.compute_score = lambda x: inv_sigmoid(self.model.predict_proba(x.reshape(1,-1))[0,1])

    def set_logreg_model_hardcoded_coefs(self):
        logreg = LogisticRegression()
        logreg.coef_ = [[ 2.39742147, -0.47638306, 2.55820895, 1.87910186, 3.80069801, 2.08620501,
            -0.0242531, 6.75999312]]
        logreg.intercept_ = [-7.39899214]
        self.model = logreg
        self.compute_score = lambda x: inv_sigmoid(self.model.predict_proba(x.reshape(1,-1))[0,1])

    def set_score_fn(self, fn):
        self.compute_score = fn

    def save_model(self, filename = MODEL_FILENAME):
        pickle.dump(logreg, open(filename, 'wb'))

    """
    def train_logreg_model(self, mapper):
        mapper.load_mapped_plans()
        mapped_plans_train, mapped_plans_test = train_test_split(mapper.mapped_plans, random_state=42, test_size=0.15)
        X_train_match, X_train_nomatch = mapper.compute_comparison_features_match_nomatch(mapped_plans_train, 100)
        X_train, Y_train = construct_features_targets(X_train_match, X_train_nomatch, 1000, 1000)
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        self.model = logreg
        self.compute_score = lambda x: inv_sigmoid(self.model.predict_proba(x.reshape(1,-1))[0,1])
    """
