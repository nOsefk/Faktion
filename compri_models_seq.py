# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb


df = pd.read_csv('data_clean.csv', index_col = 0)
df.reset_index(drop=True, inplace=True)
y = df['Loser']



            
X = df.drop('award_score', axis=1)
X = X.drop('Loser',axis=1)
X = X.drop('Bronze',axis=1)
X = X.drop('Silver',axis=1)
X = X.drop('Shortlist',axis=1)
X = X.drop('Gold',axis=1)
X = X.drop('Grand Prix',axis=1)

smote = SMOTE()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=33, stratify=y)

X_sm, y_sm = smote.fit_sample(X_train ,y_train)








#gsc = GridSearchCV(
#        estimator=RandomForestClassifier(),
#        param_grid={
#            'max_depth': [9],
#            "min_samples_leaf": [2],
#            'n_estimators': [100],
#        },
#        cv=5, verbose=10, n_jobs=10)
#    
#grid_result = gsc.fit(X_sm, y_sm)
#best_params = grid_result.best_params_
#rfc = grid_result.best_estimator_
#print(best_params)  
#scores_rfc = cross_val_score(rfc, X_sm, y_sm, cv=10, verbose=5)
#y_pred = rfc.predict(X_test)
#print('Random Forest : ', scores_rfc.mean())
#print('Test Score : ', rfc.score(X_test,y_test))


#importances_rfr = pd.Series(rfc.feature_importances_, index = X_train.columns)
#sorted_importances_rfr = importances_rfr.sort_values()
#si_rfr = sorted_importances_rfr[sorted_importances_rfr < 0.0005]
##print(si_rfr)
#
#gsc = GridSearchCV(
#        estimator=GradientBoostingClassifier(),
#        param_grid={
#            'max_depth': [2],
#            "learning_rate" : [0.05],
#            "min_samples_leaf": [ 2 ],
#            'n_estimators': [1000],
#        },
#        cv=5, verbose=10, n_jobs=-1)
#    
#grid_result = gsc.fit(X_sm, y_sm)
#best_params = grid_result.best_params_
#gbc = grid_result.best_estimator_
#print(best_params)  
#scores_gbc = cross_val_score(gbc, X_sm, y_sm, cv=10, verbose=5)
#y_pred = gbc.predict(X_test)
#print('Gradient Boosting : ', scores_gbc.mean())
#print('Test Score : ', gbc.score(X_test,y_test))


#importances_gbc = pd.Series(gbc.feature_importances_, index = X_train.columns)
#sorted_importances_gbc = importances_gbc.sort_values()
#si_gbc = sorted_importances_gbc[sorted_importances_gbc < 0.0005]
#print(si_gbc)

params = {
    'objective' :'multiclass',
    'num_class': 6,
    'learning_rate' : 0.02,
    'num_leaves' : 76,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
}

X_train2, X_valid, y_train2, y_valid = train_test_split(X_sm, y_sm, random_state=7, test_size=0.33)

d_train = lgb.Dataset(X_train2, y_train2)
d_valid = lgb.Dataset(X_valid, y_valid)

bst = lgb.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=100)

preds = bst.predict(X_test)
y_pred = []

for x in preds:
    y_pred.append(np.argmax(x))

#params = {
#    'objective' :'multiclass',
#    'num_class': 6,
#    'learning_rate' : 0.02,
#    'num_leaves' : 76,
#    'feature_fraction': 0.64, 
#    'bagging_fraction': 0.8, 
#    'bagging_freq':1,
#    'boosting_type' : 'gbdt',
#}
#
#X_train2, X_valid, y_train2, y_valid = train_test_split(X_sm, y_sm, random_state=7, test_size=0.33)
#
#d_train = lgb.Dataset(X_train2, y_train2)
#d_valid = lgb.Dataset(X_valid, y_valid)
#
#bst = lgb.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=100)
#
#preds = bst.predict(X_test)
#y_pred = []
#
#for x in preds:
#    y_pred.append(np.argmax(x))





def plot_confusion_matrix(y_true, y_pred, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=20)
    fig.tight_layout()
    fig.set_size_inches(15,10)
    return ax





np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred,  
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

