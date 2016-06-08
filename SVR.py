import numpy as np
import pandas as pd

filename = "in"
raw = pd.read_csv(filename)

df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag', 1)

train_x = df.drop('shot_made_flag', 1).as_matrix()
train_y = df['shot_made_flag'].as_matrix()

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

from sklearn import svm
import time

t1 = time.time()
model = svm.SVR(cache_size=2000, max_iter=-1)
model.fit(train_x, train_y)
t2 = time.time()

print 'time : {0:3f}'.format(t2 - t1)

pred = model.predict(submission)

sub = pd.read_csv('sample_submission.csv')
sub['shot_made_flag'] = pred
sub.to_csv('out', index=False)

print 'OK!'
