# libraries
import os
import pandas as pd
import time
import torch
from model_utils import Simple_NN, make_test_prediction


# data init
df = pd.read_csv(f'{os.getcwd()}/data/test.csv', index_col=0)
df.reset_index(drop=True, inplace=True)

# feature engineering
df['sum'] = df.sum(axis=1)  
df['min'] = df.min(axis=1)
df['max'] = df.max(axis=1)
df['mean'] = df.mean(axis=1)
df['std'] = df.std(axis=1)
df['skew'] = df.skew(axis=1)
df['kurt'] = df.kurtosis(axis=1)
df['med'] = df.median(axis=1)

# df to torch
X = df.values
X = torch.tensor(X, dtype=torch.float32)
print(f'Data shape: {X.shape}\n')

# model init
model = Simple_NN(X.shape[1], 16)
model.load_state_dict(torch.load(f'{os.getcwd()}/models/model.pt'))
model.eval().to('cpu')
print(f'Model:\n{model}')

# make predictions
predict_start_time = time.time()
print('\nMaking predictions ...')
df_predictions = make_test_prediction(X, 64, model)
print(f'Predicted! Took {time.time() - predict_start_time} sec\n')

# save submission csv file
df_predictions = df_predictions.rename(columns={'prediction': 'target'}, index=lambda s: 'test_' + str(s))
df_predictions.index.name = 'ID_code'
df_predictions.to_csv(f'{os.getcwd()}/predictions/predictions.csv')

# check predicted class ratio
predictions = df_predictions.values
ratio = predictions[predictions > 0.5].shape[0] / predictions[predictions < 0.5].shape[0]
print(f'First to zero class ratio: {ratio}')
