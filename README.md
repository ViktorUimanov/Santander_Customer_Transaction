# Santander Customer Transaction
This repository is devoted to Kaggle competition named "Santander Customer Prediction"

## General Idea
We found out that there are no solutions proposing stacked models. Although Gradient Boosting appeared superior to other models, its performance can be enhanced when stacked with other models such as Neural Nets, SVM, and Naive Bayes. 

## Results

After running stacking algorithm with Logistic regression we obtained following results:
<img width="1119" alt="image" src="https://user-images.githubusercontent.com/100875246/208742329-08d1080d-d1b9-40fe-81d1-cc2f05e37ccf.png">


|                | Lgbm_weight | NN_Focal_Loss_weight |   SVM_weight  | Bias_weight | Privat_score (AUC) |
|:--------------:|:-----------:|:--------------------:|:-------------:|:-----------:|:------------------:|
| Logistic_Stack |    0.6945   |        0.2010        |     0.0506    |    0.0539   |      0.90103       |
|    manual      |    0.8000   |        0.1000        |     0.0000    |    0.1000   |      0.89819       |
|    manual      |    1.0000   |        0.0000        |     0.0000    |    0.0000   |      0.89225       |
|    manual      |    0.0000   |        1.0000        |     0.0000    |    0.0000   |      0.89020       |
|    manual      |    0.3000   |        0.3000        |     0.2000    |    0.2000   |      0.88959       |
|    manual      |    0.2500   |        0.2500        |     0.2500    |    0.2500   |      0.86031       |
