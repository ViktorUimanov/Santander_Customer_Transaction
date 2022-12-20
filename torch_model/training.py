# libraries
import os
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from dataset_utils import create_dataset
from model_utils import Simple_NN, reset_weights, WeightedFocalLoss, train_model, test_model


# use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create torch dataset
dataset = create_dataset(f'{os.getcwd()}/data/train.csv')

# model init
model = Simple_NN(dataset.X.shape[1], 16).to(device)
model.apply(reset_weights)

# hyper parameters
n_splits = 5 # Number of K-fold Splits
batch_size = 128 # Num of batches for model training
num_epochs = 20 

# cross-validation 
sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4590)

# model's training parameters
loss_func = WeightedFocalLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, 
                                                    base_lr=0.001, 
                                                    max_lr=0.005, 
                                                    step_size_up=2000, 
                                                    mode='triangular', 
                                                    scale_mode='exp_range', 
                                                    gamma=0.99994, 
                                                    cycle_momentum=False)


train_start_time = time.time()
print('='*100)
print('Starting the training ...\n')
# training loop
predictions_train = np.zeros(dataset.X.shape[0])
for fold, (train_idx,test_idx) in enumerate(sk.split(dataset.X, dataset.y)):
    # history
    train_loss_fold = []
    test_loss_fold = []
    train_acc_fold = []
    test_acc_fold = []
    
    print('----------------------Fold № {}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    model.apply(reset_weights) # each cross-validation fold starts from some random initial state
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # train and validation
        train_loss, train_acc = train_model(model, train_loader, device, loss_func, optimizer, my_lr_scheduler)
        test_loss, test_acc, _ = test_model(model, test_loader, device, loss_func, test = False)

        # print the results for this epoch:
        print(f'Epoch {epoch} of {num_epochs} took {time.time() - start_time:.3f}s')
        
        # loss
        train_loss_fold.append(np.mean(train_loss))
        test_loss_fold.append(np.mean(test_loss))
        # accuracy
        train_acc_fold.append(np.mean(train_acc))
        test_acc_fold.append(np.mean(test_acc))
        
        print(f"\tTraining loss: {train_loss_fold[-1]}")
        print(f"\tTraining accuracy: {train_acc_fold[-1]}")
        print(' ')
        print(f"\tTesting loss: {test_loss_fold[-1]}")
        print(f"\tTesting accuracy: {test_acc_fold[-1]}")
        
    __, _, pred_idx, prediction = test_model(model, fold, test_loader, device, loss_func, test = True)
    
    pred_lst = list(batch.cpu().data.numpy() for batch in prediction)
    pred_ar = np.array(pred_lst).reshape(-1)

    idx_lst = list(batch.cpu().data.numpy() for batch in pred_idx)
    idx_ar = np.array(idx_lst).reshape(-1)
    
    predictions_train[idx_ar] = pred_ar


print(f'\nThe training is over, total time is {time.time() - train_start_time} sec')
print('='*100)

model_save_start_time = time.time()
print('Saving model ...')
torch.save(model.state_dict(), f'{os.getcwd()}/models/model.pt')
print(f'Model saved! Took {time.time() - model_save_start_time} sec')
