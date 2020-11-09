---
title: Predict Bitcoin price using Machine Learning Model
tags: [Machine Learning, LSTM, Keras, Cryptocurrency]
style: fill
color: warning
description: Prediction of the Bitcoin price using LSTM by using the numerical historical data of the Bitcoin.
---


Bitcoin has set the trend for investors to put their money and trust in the cryptocurrency over the past few years and no doubt that it is for the long run. So wouldn’t it be great if we can predict tomorrow’s price of Bitcoin? The value of the Bitcoin climbed its pick on Dec 2018.

Although there are many cryptocurrencies out there here in this blog we will be predicting the Bitcoin price using LSTM by using the numerical historical data of Bitcoin. There are several resources and API’s which provides up-to-date coin price.

LSTM is artificial recurrent neural network architecture with feedback connection. LSTM memory cell is composed of an input gate, output gate and forget gate

Here the data set is quite simple as it contains only timestamp and Bitcoin price.

![](https://cdn-images-1.medium.com/max/2000/0*yvrbe7367nJMSCHT.png)

Below is the visualization of the data set.

![](https://cdn-images-1.medium.com/max/2000/0*iM-HFcBZYN_WH2za.png)

Here we will use the past 9 days data for prediction and normalize it. Below are the functions to perform this task.

![](https://cdn-images-1.medium.com/max/2000/0*Got-ZEPzucPnVK2n.png)

Split the data into train and test set.

![](https://cdn-images-1.medium.com/max/2000/0*pyVelKTnKQmFaW1g.png)

## Create a Model

The model will consist of two LSTM layers and two dense layers. Below is the model architecture, you can try a different combination of the parameter and different model architectures. Usually, LSTM models consist of a few layers. As it is a regression problem so we will use linear as an activation function.

![](https://cdn-images-1.medium.com/max/2000/0*YjPZzM-csuLOdYGM.png)

Now it is time to train the model, you can train model over different parameters.

![](https://cdn-images-1.medium.com/max/2000/0*hUea3uhxFSu9-15q.png)

After 25 epochs model is trained and now we can use it to predict next day’s Bitcoin price. Do not forget to save the model so you can use it later.

To predict for tomorrow’s price we need past 9 days data. Below new_sample is past 9 days data. We will normalize it and then we will predict. As it will give the value between 0 to 1 we need to de-normalize it to get the real value

![](https://cdn-images-1.medium.com/max/2000/0*8fAdOo7xUk8NzD87.png)

Although it gives the pretty good results they are quite different than actual values. Therefore, the model needs to be improved.

**Note**: *It is not advisable to invest based on this prediction, one should always take the assistance of a qualified advisor for investment.*

Thanks for reading this article.

![](https://cdn-images-1.medium.com/max/2000/0*9KMQ6SuIDSX6uEN3)

AI and machine learning enthusiast. [View all posts by csvankhede](https://csvankhede.wordpress.com/author/csvankhede/)

**Published**

*Originally published at [http://csvankhede.wordpress.com](https://csvankhede.wordpress.com/2019/08/24/bitcoin-price-prediction/) on August 24, 2019.*
