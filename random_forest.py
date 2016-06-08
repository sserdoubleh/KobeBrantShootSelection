import numpy as np
import pandas as pd

filename = "in"
raw = pd.read_csv(filename)

df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag', 1)

train_x = df.drop('shot_made_flag', 1)
train_y = df['shot_made_flag']

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import time

from sklearn.cross_validation import KFold

default_n = 100
default_m = 11

best_m = default_m
# find best max_depth for RandomForestRegressor
# print('Finding best max_depth for RandomForestRegressor')
# min_score = 100000
# scores_m = []
# range_m = np.linspace(8, 13, num=6).astype(int)
# for m in range_m:
#     print('the max depth: {0}'.format(m))
#     t1 = time.time()
# 
#     rfc_score = 0
#     rfc = RandomForestRegressor(n_estimators=default_n, max_depth=m)
#     for train_k, test_k in KFold(len(train_x), n_folds=10, shuffle=True):
#         rfc.fit(train_x.iloc[train_k], train_y.iloc[train_k])
#         pred = rfc.predict(train_x.iloc[test_k])
#         print logloss(train_y.iloc[test_k], pred)
#         rfc_score += logloss(train_y.iloc[test_k], pred) / 10
#     scores_m.append(rfc_score)
#     if rfc_score < min_score:
#         min_score = rfc_score
#         best_m = m
# 
#     t2 = time.time()
#     print('Done processing {0} max_dep with cost ({1:.3f}) ({2:.3f})'.format(m, rfc_score, t2 - t1))
# print(best_m, min_score)

best_n = default_n
# find the best n_estimators for RandomForestRegressor
print('Finding best n_estimators for RandomForestRegressor')
min_score = 100000
scores_n = []
range_n = np.linspace(200, 250, num=2).astype(int)
print(range_n)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0
    rfc = RandomForestRegressor(n_estimators=n, max_depth=best_m)
    for train_k, test_k in KFold(len(train_x), n_folds=10, shuffle=True):
        rfc.fit(train_x.iloc[train_k], train_y.iloc[train_k])
        pred = rfc.predict(train_x.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()
    print('Done processing {0} tree with cost ({1:.3f}) ({2:.3f})'.format(n, rfc_score, t2 - t1))
print(best_n, min_score)

model = RandomForestRegressor(n_estimators=best_n, max_depth=best_m)
model.fit(train_x, train_y)
pred = model.predict(submission)

sub = pd.read_csv('sample_submission.csv')
sub['shot_made_flag'] = pred
sub.to_csv('real_submission.csv', index=False)

print 'OK!'
