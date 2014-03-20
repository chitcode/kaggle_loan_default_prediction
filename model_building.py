## Author: Chitrasen
## 
## train_master has to be commented after first run to reduce the data loading time.



import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,mean_absolute_error,f1_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier

# following 2 lines has to be commented after first run, to reduce the data loading time for interatively
# experimenting with different models.
train_master = pd.read_csv('data/train_v2.csv', low_memory=True)
train_master = train_master.drop(['id','f53','f137','f206','f276','f419','f466','f639','f703','f731','f743'],1)

train = train_master.copy()

train_loss = train.loss
train = train.drop(['loss'],1)
features = train.columns

train_featured = train[['f527','f528','f271','f9']]#,'f2', 'f57', 'f9','f39','f271','f274'
f2_unique = train_master['f2'].unique()
for feature_val in f2_unique:
    train_featured['f2_'+str(feature_val)] = 1.* (train_master['f2'] == feature_val)
                             
imp1 = Imputer()

imp1.fit(train_featured)
train_featured = imp1.transform(train_featured)

scalaer = StandardScaler()
scalaer.fit(train_featured)
train_featured = scalaer.transform(train_featured)


train = train[['f527','f528','f271','f9','f39','f57','f275', u'f13', u'f25', u'f26', u'f31', u'f63', u'f67', 
               u'f68', u'f142', u'f230', 
               u'f258', u'f260', u'f263', u'f270', u'f281', u'f282', u'f283', u'f314', 
               u'f315', u'f322', u'f323', u'f324', u'f376', u'f377', u'f395', u'f396', 
               u'f397', u'f400', u'f402', u'f404', u'f405', u'f406', u'f424', u'f442', 
               u'f443', u'f516', u'f517', u'f596', u'f597', u'f598', u'f599', u'f629', 
               u'f630', u'f631', u'f671', u'f675', u'f676', u'f765', u'f766', u'f767', u'f768']]

for feature_val in f2_unique:
    train['f2_'+str(feature_val)] = 1.* (train_master['f2'] == feature_val)


imp2 = Imputer()
imp2.fit(train)
train = imp2.transform(train)


scalaer = StandardScaler()
scalaer.fit(train)
train = scalaer.transform(train)


##### models
clf = LogisticRegression(penalty='l2',C=1e10, dual = False,class_weight ='auto')

regr = GradientBoostingRegressor(loss='lad',n_estimators = 200,max_depth = 3,
                              min_samples_leaf = 4,min_samples_split=2,learning_rate = 0.1, alpha = 0.9)


train_loss_b = train_loss.apply(lambda x:1 if x>0 else 0)

auc_score = []
mae_score = []
f1 = []
kf = StratifiedKFold(train_loss,n_folds=4)

for i,(train_idx,cv_idx) in enumerate(kf):
    print '='*20
    X_train = train_featured[train_idx,:]
    X_cv = train_featured[cv_idx,:]
    
    y_train = train_loss_b.values[train_idx]
    y_cv = train_loss_b.values[cv_idx]
    
    #sgd.fit_transform(X_train,y_train)
    #X_train = sgd.transform()

    clf.fit(X_train,y_train)
    
    auc = roc_auc_score(y_cv,clf.predict_proba(X_cv)[:,1])    
    
    auc_score.append(auc)
    
    #y_clf_pred = clf.predict(X_cv)
    y_clf_pred = np.zeros(y_cv.shape[0])
    y_clf_pred[clf.predict_proba(X_cv)[:,1] > 0.65] = 1
    f1_local = f1_score(y_cv,y_clf_pred)
    f1.append(f1_local)
    
    print 'AUC score for ',i,auc
    print 'F1 score for ',i, f1_local
    #print classification_report(y_cv,y_clf_pred)
    
    X_train_reg = train[train_idx,:][y_train == 1,:]
    y_train_reg = train_loss.values[train_idx][y_train == 1]
    
    X_cv_reg = train[cv_idx,:][y_clf_pred == 1,:]
    y_cv_reg = train_loss.values[cv_idx][y_clf_pred == 1]
    
    #selector.fit(X_train_reg,y_train_reg)
    #X_train_reg = selector.transform(X_train_reg)
    #X_cv_reg = selector.transform(X_cv_reg)    
    
    #print 'Important Features ::::', features[selector.get_support()]
    ##ridge.fit(X_train_reg,y_train_reg)
    ##y_reg_pred = ridge.predict(X_cv_reg)
    
    regr.fit(X_train_reg,y_train_reg)
    y_reg_pred = regr.predict(X_cv_reg)
    
    y_reg_pred[y_reg_pred < 0] = 0
    y_reg_pred[y_reg_pred > 100] = 100
    print 'Regg MAE',mean_absolute_error(y_reg_pred,y_cv_reg)
    
    y_clf_pred[y_clf_pred == 1] = y_reg_pred
    mae = mean_absolute_error(y_clf_pred,train_loss.values[cv_idx])
    print 'MAE for ',i,mae
    mae_score.append(mae)
    
print 'Avg AUC score', np.mean(auc_score)
print 'Avg f1 score', np.mean(f1)
print 'F1 scores array',f1
print 'Avg MAE score',np.mean(mae_score)
