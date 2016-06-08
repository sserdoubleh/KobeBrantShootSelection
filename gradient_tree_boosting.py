import numpy as np
import pandas as pd
import copy

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import *
import time

from sklearn.cross_validation import KFold
from sklearn import grid_search

import scipy as sp

class Regression:
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
            pred = model.predict(self.train_x.iloc[validate_k])
            ls = log_loss(self.train_y.iloc[validate_k], pred)
            print 'loss:', ls
            model_score += ls / 10
        return model, model_score

    def find_best_parament(self, parament_name, parament_range):
        print('Finding best {0}...'.format(parament_name))
        print parament_range
        args = self.default_args
        min_score = 100000
        best_parament = args[parament_name]
        for parament_value in parament_range:
            print("{0} : {1}".format(parament_name, parament_value))
            args[parament_name] = parament_value

            t1 = time.time()
            model_score = self.get_score(args)
            t2 = time.time()
            print 'Done! {0} = {1} time : {2} score : {3}'.format(parament_name, parament_name, t2 - t1, model_score)

            if model_score < min_score:
                min_score = model_score
        self.best_parament[parament_name] = best_parament
        print 'best {0} : {1} score : {2}'.format(parament_name, best_parament, min_score)

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
        self.best_score = 10000
        self.find_best_rec(self.default_args, args_range, 0)

    def test(self):
        model = globals()[self.model_name](**self.best_args)

        print 'best args:', self.best_args
        print 'best score:', self.best_score

        t1 = time.time()
        model.fit(self.train_x, self.train_y)
        pred = model.predict(self.submission)
        t2 = time.time()

        sub = pd.read_csv('sample_submission.csv')
        sub['shot_made_flag'] = pred
        sub.to_csv(self.output_filename, index=False)

        print 'OK! time pass: {0}'.format(t2 - t1)

default_args = {}
solver = Regression(default_args, "GradientBoostingRegressor", "in", "GDBT_out")
solver.find_best({
    'n_estimators': [1000],
    'subsample': [.7],
    'max_depth': [6],
    'learning_rate': [.0075],
    })
solver.test()
