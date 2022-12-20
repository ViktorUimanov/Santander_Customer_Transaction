from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np




class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None, bayes = False):
        # params['random_state'] = seed
        self.clf = clf(**params)
        self.bayes = bayes

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        if self.bayes:
            return self.clf.predict_proba(x)[:, 1].reshape(1,-1)
        return self.clf.predict_proba(x)[:, 1].reshape(1,-1)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

        
def get_oof(clf, x_train, y_train, x_test, save_results = True, model_name = 'model'):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        print('Starting {} validation'.format(i))
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]
        print(x_te.shape)
        clf.fit(x_tr, y_tr)
        
        y_test_pred = clf.predict(x_te)
        oof_train[test_index] = y_test_pred
        oof_test_skf[i, :] = clf.predict(x_test)
        
    


    oof_test[:] = oof_test_skf.mean(axis=0)

    if save_results:
        oof_test_pd = pd.DataFrame(oof_test)
        oof_test_pd.to_csv('test_{}.csv'.format(model_name))
        oof_train_pd = pd.DataFrame(oof_train)
        oof_train_pd.to_csv('train_{}.csv'.format(model_name))
    return oof_train, oof_test
        
class Gboosting:
    
    def __init__(self, train_df,test_df,params):
        
        self.NFOLDS = 5
        
        self.skf = StratifiedKFold(n_splits = self.NFOLDS, shuffle = True, random_state = 102)

        self.res_val = np.zeros(len(train_df))
        self.predictions = np.zeros(len(test_df))
        self.results = pd.DataFrame()
        self.features = X.columns.to_list()
    
    
    def make_predictions(self):
        for fold_, (train_idx, val_idx) in enumerate(skf.split(X.values, y.values)):
            print("Fold {}".format(fold_))
            trn_data = lgb.Dataset(self.X.iloc[train_idx], label=self.y.iloc[train_idx])
            val_data = lgb.Dataset(self.X.iloc[val_idx], label=self.y.iloc[val_idx])

            lgbt = lgb.train(params, trn_data, num_boost_round = 500_000,
                            valid_sets = [trn_data, val_data], early_stopping_rounds = 3000, 
                             verbose_eval=1000)
            
            predictions_fold = lgbt.predict(X.iloc[val_idx])

            res_val[val_idx] = predictions_fold

            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = self.features
            fold_importance_df["importance"] = lgbt.feature_importance()
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([results, fold_importance_df], axis=0)

            self.predictions += lgbt.predict(test_df[features], num_iteration=lgbt.best_iteration)


        predictions_df = pd.DataFrame({'ID_code' : test_df['ID_code'].values})
        predictions_df['predictions'] = self.predictions
        
        return predictions_df

