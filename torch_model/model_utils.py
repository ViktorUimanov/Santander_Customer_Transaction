# libraries
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC


# fully-connected simple torch model
class Simple_NN(nn.Module):
    def __init__(self ,input_dim ,hidden_dim, dropout = 0.75):
        super(Simple_NN, self).__init__()
        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(int(hidden_dim*input_dim), 1)

    def forward(self, x):
        b_size = x.size(0)
        x = x.reshape(-1, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.reshape(b_size, -1)
        
        out= self.fc2(x)
        return out


# Focal loss
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=0.15, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        #self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean()


# ........
def augment_counts(x, y, t_pos, t_neg):
    xs,xn = [],[]
    for i in range(t_pos):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(208):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
              
        xs.append(x1)

    for i in range(t_neg):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(208):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
                
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys, yn])

    return x,y


# reseting model before each fold
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


# training loop
def train_model(model, train_loader, device, loss_fn, opt, my_lr_scheduler):
    train_loss = []
    train_acc = []
    auroc = AUROC(task='binary')
    
    model.train(True) # enable dropout / batch_norm training behavior
    global ys
    for (X_batch, y_batch, _) in train_loader:

        X_batch, y_batch = augment_counts(X_batch.numpy(), y_batch.numpy(), 2, 1)
        X_batch = torch.tensor(X_batch, dtype = torch.float32)
        y_batch = torch.tensor(y_batch, dtype = torch.float32)

        opt.zero_grad()

        # move data to target device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
    
        # make f/w and b/w pass
        predictions = model(X_batch).view(-1)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        opt.step()
        
        # step for lr
        my_lr_scheduler.step()
        
        train_loss.append(loss.item())
        accuracy = auroc(predictions, torch.tensor(y_batch, dtype=torch.int32).to(device))
        train_acc.append(accuracy)
    
    return train_loss, train_acc


# validation loop
def test_model(model, test_loader, device, loss_fn, test = False):
    pred_idx = []
    predictions_list = []
    test_loss = []
    test_acc = []
    auroc = AUROC(task='binary')
    
    model.eval() # disable dropout / use averages for batch_norm
    with torch.no_grad():
        for X_batch, y_batch, idx in test_loader:
            # move data to target device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # compute predictions
            predictions = model(X_batch).view(-1)
            loss = loss_fn(predictions, y_batch)
            test_loss.append(loss.item())

            accuracy = auroc(predictions, torch.tensor(y_batch, dtype=torch.int32).to(device))
            test_acc.append(accuracy)

            pred_idx.append(idx)
            predictions_list.append(predictions)

    if test:
        return test_loss, test_acc, pred_idx, predictions_list 
    else:
        return test_loss, test_acc, pred_idx


# model inference
def make_test_prediction(X, batch_size, model):
    df_predictions = {'prediction': []}

    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i + batch_size]
        test_preds = model(X_batch).view(-1)
        df_predictions['prediction'].extend(test_preds.detach().numpy())

    return pd.DataFrame(df_predictions)
