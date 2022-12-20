# Santander Customer Transaction
This repository is devoted to Kaggle competition named "Santander Customer Prediction"

## General Idea
We found out that there are no solutions proposing stacked models. Although Gradient Boosting appeared superior to other models, its performance can be enhanced when stacked with other models such as Neural Nets, SVM, and Naive Bayes. 

## Results

After running stacking algorithm with Logistic regression we obtained following results:
<img width="1119" alt="image" src="https://user-images.githubusercontent.com/100875246/208742329-08d1080d-d1b9-40fe-81d1-cc2f05e37ccf.png">


| Architecture | Train accuracy | Train loss | Test accuracy | Test loss |           |
|:------------:|:--------------:|:----------:|:-------------:|:---------:|:---------:|
|    1-32-1    |                |            |               |           |           |
|    2-32-1    |                |            |               |           |           |
|    2-64-1    |                |            |               |           |           |
|    3-64-1    |                |            |               |           |           |
|    2-32-2    |                |            |               |           |           |
|    1-64-2    |                |            |               |           |           |
|    2-64-2    |                |            |               |           |           |
|    3-64-2    |                |            |               |           |           |
|    1-32-3    |                |            |               |           |           |
|    2-64-3    |                |            |               |           |           |
