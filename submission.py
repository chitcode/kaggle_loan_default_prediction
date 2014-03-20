 	import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler


from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('data/train_v2.csv',low_memory=True)
train_loss = train.loss

train_featured = train[['f527','f528','f271','f9']]
f2_unique = train['f2'].unique()
for feature_val in f2_unique:
    train_featured['f2_'+str(feature_val)] = 1.* (train['f2'] == feature_val)


imp1 = Imputer()
imp1.fit(train_featured)
train_featured = imp1.transform(train_featured)


scalaer1 = StandardScaler()
scalaer1.fit(train_featured)
train_featured = scalaer1.transform(train_featured)

train = train[['f527','f528','f271','f2','f9','f39','f57', u'f13', u'f25', u'f26', u'f31', u'f63', 
               u'f67', u'f68', u'f142', u'f230', 
               u'f258', u'f260', u'f263', u'f270', u'f281', u'f282', u'f283', u'f314', 
               u'f315', u'f322', u'f323', u'f324', u'f376', u'f377', u'f395', u'f396', 
               u'f397', u'f400', u'f402', u'f404', u'f405', u'f406', u'f424', u'f442', 
               u'f443', u'f516', u'f517', u'f596', u'f597', u'f598', u'f599', u'f629', 
               u'f630', u'f631', u'f671', u'f675', u'f676', u'f765', u'f766', u'f767', u'f768']]
imp2 = Imputer()
imp2.fit(train)
train = imp2.transform(train)

scalaer2 = StandardScaler()
scalaer2.fit(train)
train = scalaer2.transform(train)


test = pd.read_csv('data/test_v2.csv',low_memory=True)

test_featured = test[['f527','f528','f271','f9']]

for feature_val in f2_unique:
    test_featured['f2_'+str(feature_val)] = 1.* (test['f2'] == feature_val)


test_featured = imp1.transform(test_featured)
test_featured = scalaer1.transform(test_featured)


test = test[['f527','f528','f271','f2','f9','f39','f57', u'f13', u'f25', u'f26', u'f31', u'f63', u'f67', 
               u'f68', u'f142', u'f230', 
               u'f258', u'f260', u'f263', u'f270', u'f281', u'f282', u'f283', u'f314', 
               u'f315', u'f322', u'f323', u'f324', u'f376', u'f377', u'f395', u'f396', 
               u'f397', u'f400', u'f402', u'f404', u'f405', u'f406', u'f424', u'f442', 
               u'f443', u'f516', u'f517', u'f596', u'f597', u'f598', u'f599', u'f629', 
               u'f630', u'f631', u'f671', u'f675', u'f676', u'f765', u'f766', u'f767', u'f768']]
test = imp2.transform(test)
test = scalaer2.transform(test)

gbr = GradientBoostingRegressor(loss='lad',n_estimators = 200,min_samples_leaf = 6,
                                min_samples_split=2,max_features=10)

train_loss_b = train_loss.apply(lambda x:1 if x>0 else 0)


print 'Predicting classification....'

clf = LogisticRegression(penalty='l2',C=1e10, dual = False,class_weight = 'auto')
clf.fit(train_featured,train_loss_b.values)
    
y_clf_pred = np.zeros(test_featured.shape[0])
y_clf_pred[clf.predict_proba(test_featured)[:,1] > 0.65] = 1
clf.predict(test_featured)
    
print 'Predicting regression....'
    
X_train_reg = train[train_loss.values > 0,:]
y_train_reg = train_loss.values[train_loss.values > 0]
    
X_test_reg = test[y_clf_pred == 1,:]


gbr.fit(X_train_reg,y_train_reg)
y_reg_pred = gbr.predict(X_test_reg)

y_reg_pred[y_reg_pred < 1] = 1
y_reg_pred[y_reg_pred > 100] = 100
    
y_clf_pred[y_clf_pred == 1] = y_reg_pred

submission = pd.read_csv('data/sampleSubmission.csv')
submission['loss'] = y_clf_pred
submission.to_csv('submissions/beat_benchmark_gbr_527_528_271_9_2fact_1.csv',index=False)
print 'Submission File Created.'
