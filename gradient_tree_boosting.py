import numpy as np
import pandas as pd
import copy

from sklearn.ensemble import *
from sklearn.metrics import *
from xgboost.sklearn import *
import time

from sklearn.cross_validation import KFold
from sklearn import grid_search

import scipy as sp

class Kobe_Solver:
    def __init__(self, default_args, model_name, input_filename, output_filename):
        self.default_args = default_args
        self.best_args = copy.copy(self.default_args)

        self.model_name = model_name
        self.input_filename = input_filename
        self.output_filename = output_filename

        raw = pd.read_csv(input_filename)
        seed = 19931130

        train = raw[pd.notnull(raw['shot_made_flag'])]
        self.train_x = train.drop('shot_made_flag', 1)
        self.train_y = train['shot_made_flag']

        self.submission = raw[pd.isnull(raw['shot_made_flag'])].drop('shot_made_flag', 1)
        self.kfold = KFold(len(self.train_x), n_folds=10, random_state=seed)

    def get_score(self, args):
        model = globals()[self.model_name](**args)
        model_score = 0
        for train_k, validate_k in self.kfold:
            model.fit(self.train_x.iloc[train_k], self.train_y.iloc[train_k])
            if "Classifier" in self.model_name:
                pred = model.predict_proba(self.train_x.iloc[validate_k])[:,1]
            else:
                pred = model.predict(self.train_x.iloc[validate_k])
            ls = log_loss(self.train_y.iloc[validate_k], pred)
            print 'loss:', ls
            model_score += ls / 10
        return model, model_score

    def find_best_rec(self, args, args_range, depth):
        if depth == len(args_range):
            print args, ':'
            start_time = time.time()
            model, model_score = self.get_score(args)
            end_time = time.time()
            print args
            print 'Model score: {0:10f}, Time pass: {1:10f}'.format(model_score, end_time - start_time)
            if model_score < self.best_score:
                self.best_score = model_score
                self.best_args = copy.copy(args)
            return
        parament_name = args_range.keys()[depth]
        parament_value_range = args_range.values()[depth]
        for parament_value in parament_value_range:
            args[parament_name] = parament_value
            self.find_best_rec(args, args_range, depth + 1)

    def find_best(self, args_range):
        print self.model_name
        self.best_score = 10000
        self.find_best_rec(self.default_args, args_range, 0)

    def test(self):
        model = globals()[self.model_name](**self.best_args)

        print 'best args:', self.best_args
        print 'best score:', self.best_score

        t1 = time.time()
        model.fit(self.train_x, self.train_y)
        if "Classifier" in self.model_name:
            pred = model.predict_proba(self.submission)[:,1]
        else:
            pred = model.predict(self.submission)
        t2 = time.time()

        sub = pd.read_csv('sample_submission.csv')
        sub['shot_made_flag'] = pred
        sub.to_csv(self.output_filename, index=False)

        print 'OK! time pass: {0}'.format(t2 - t1)

# XGBClassifier
# XGBRegressor
# GradientBoostingRegressor
# GradientBoostingClassifier
default_args = {
        'n_estimators': 1000,
        'subsample': .7,
        # 'max_features': .5,
        'colsample_bytree': .7,
        'max_depth': 6,
        'learning_rate': .008,
        }
solver = Kobe_Solver(default_args, "XGBClassifier", "my.in", "GBDT.out")
solver.find_best({
    })
solver.test()
