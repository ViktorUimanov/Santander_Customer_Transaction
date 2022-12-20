import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(train_df, test_df):
    
    
    X = train_df.drop(['ID_code', 'target'], axis = 1)
    y_train = train_df['target']
    test_df.drop(['ID_code'], axis = 1, inplace = True)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_train = scaler.transform(X)
    scaler.fit(test_df)
    X_test = scaler.transform(test_df)
    
    return X_train, y_train, X_test