---
title: Performance matrix in Machine Learning
tags: [Machine Learning, Performace matrix]
style: fill
color: primary
description: Performance matrix used in machine learning for evaluation of different ML algorithms.
---


In the context of machine learning the performance matrix is used after implementation of the ML algorithm to find out the effectiveness of the model. There are various performance matrices out there relevant for the evaluation of different ML algorithm.

Testing data is used to evaluate the performance matrix as training data is already used for tuning the model to predict the output.

**Confusion matrix**

The confusion matrix is mostly used matrix in Machine Learning. confusion matrix is used for classification problems. We can calculate lots of other performance matrix based on the numbers inside the confusion matrix.

After training and testing the model on train and test data respectively we can create confusion matrix based on testing results.

Here for confusion matrix, we will consider the predicted labels (y’) and actual labels (y) of the test data.

![](https://cdn-images-1.medium.com/max/2000/0*ASM0L272NhfPfXet.png)

**True Positive(TP):** It is a number of labels whose actual value and predicted value both are positive(1).

**True Negative(TN):** It is a number of labels whose actual value and predicted value both are negative(0).

**False Positive(FP):** It is the number of labels whose actual value is negative(0) but predicted as positive(1).

**False Negative(FN):** It is the number of labels whose actual value is positive(1) but predicted as negative(0).

Here the main aim is to reduce FP and FN which will eventually increase TP and TN.

Now we will see some of the other performance matrices which can be calculated using TP, FP, TN, and FN.

**Accuracy**

Accuracy calculates how many points are correctly classified amongst all the points.

![](https://cdn-images-1.medium.com/max/2000/0*f7aY1a2yW7neVmwb.png)

It is preferred to use accuracy only when you have balanced dataset, else it will mislead the performance.

**Precision**

![](https://cdn-images-1.medium.com/max/2000/0*O8C0OapEckRMdVFL.png)

When you want to measure what proportion of the prediction is correct precision is useful.

Precision gives the information about the model performance with respect to False Positive.

**Recall**

Recall measures what proportion of the positive points are predicted positive. In other words, the proportion of actual positives is identified correctly.

![](https://cdn-images-1.medium.com/max/2000/0*DRAaj7_W6dCKpoBT.png)

Recall gives the information about the model performance with respect to False Negative.

**F1- score**

F1 score is a measure of the test’s accuracy. It gives balanced value between precision and recall. To calculate F1 score the harmonic mean of precision and recall is used.

![](https://cdn-images-1.medium.com/max/2000/0*JVkx-yF4GDuReQxq.png)

Here let’s say,

A = Precision

H = Half of the Harmonic mean

F1-Score is a good matrix when data is imbalanced.

**Log — loss**

Log loss is a probability-based classification matrix. Log loss uses the prediction where prediction is a probability value between 0 to 1. For any problem lower log loss is preferred as it indicates the better performance of the model.

The log loss extensively used in logistic regression and neural network. For comparing models log loss is a good option.

![](https://cdn-images-1.medium.com/max/2000/0*VDIBGrXorNZ1pdR2.png)

**Hamming Loss**

Hamming loss is the fraction of the wrong label to the total number of labels. Hamming loss is equal to (1 -accuracy) for the binary case. Hamming loss is mostly used in multi-label classification problems.

![](https://cdn-images-1.medium.com/max/2000/0*t6vFj82AVR9c6E3f.png)

N = no. of data points

L = size of label

**ROC — AUC curve**

Here ROC stands for receiver operating characteristic curve and AUC stands for Area Under the ROC Curve.

ROC-AUC curve is a measurement of the model performance at various threshold settings.ROC curve plots two parameters **TPR** and **FPR.**

**TPR **(True Positive Rate): TPR is nothing but recall.

**FPR **(False Positive Rate): FPR is (1- specificity).

TPR and FPR can be defined as follows.

![](https://cdn-images-1.medium.com/max/2000/0*1hydQdFwPxsU5Squ.png)

AUC measures the area underneath the entire ROC curve. AUC provides a measure of the performance across all possible thresholds.

AUC ranges from 0 to 1. Value of AUC greater than 0.5 is preferred.

Here 0 means predictions are 100% wrong and 1 means predictions are 100% right.

![](https://cdn-images-1.medium.com/max/2000/0*-5_S-YVKeYHqITN7.png)

*Originally published at [http://csvankhede.wordpress.com](https://csvankhede.wordpress.com/2019/07/01/performance-matrix-in-machine-learning/) on July 1, 2019.*
