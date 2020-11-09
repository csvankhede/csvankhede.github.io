---
title: Convolutional Neural Network to diagnose Malaria.
tags: [CNN, Deep Learning, Keras, Malaria]
style: border
color: danger
description: Convolutional neural network to detect malaria parasites in microscopic images of the blood samples.
---
![](https://miro.medium.com/max/875/0*ppl28a2aO-aG5ROg)
Malaria causes death for many people, most affected age group by malaria is below 5 years. It is caused by Plasmodium parasites transmitted by the infected female Anopheles mosquitoes. There are several techniques used for a blood test of malaria, but they require an expert to inspect the blood sample of the patient. Here we will create a deep learning model to do this task performed by an expert.

In this article, you will see how well deep learning performs on image data. The task mentioned in this article can be performed by any beginner, as this is a simple application of CNN, nothing fancy

### **Data set**

Here we have a data set of microscopy images of the stained blood sample and annotation files for each image. We need to extract patches from this given data so that we can use it to train our model. Below is the sample image.

![](https://cdn-images-1.medium.com/max/2000/1*KfRVRMjX9LZYRufkOkHqTQ.jpeg)

The annotation file is nothing but XML files containing the x and y coordinates of plasmodium parasites in the image. We have extracted the patches from the image of two types using the annotation file.

Below are some extracted patches. The size of the patches here is 50X50.

![](https://cdn-images-1.medium.com/max/2000/0*4GjHv4YtCBBBQiVJ)

Now we have extracted the patches from the image we need to convert them to a data frame. The RGB image is a collection of pixel values which is a numeric value between 0 to 255 and three channels. So as we have patches of 50×50 with three channels, the total number of columns in the data frame will be 50*50*3 = 7500, starting with 0 to 7499. Below is the pandas data frame of the pixel value and its label. Here label 1 shows parasite present and 0 shows parasite absent.

![](https://cdn-images-1.medium.com/max/2048/0*D5f3oyNtER8qkzHE)

As we don’t want our model to get biased towards any particular label, we selected equal no. of data points from both the classes.

### **Split data into train and test set**

Split the data into train and test set, reshape data to the input shape of the model.

![](https://cdn-images-1.medium.com/max/2048/0*KjfsMn9aInDy8N4a)

### **Model**

As we have our data ready lets create a CNN model. I have created a model of 10 layers which consist of Conv2D, MaxPooling2D, Flatten and Dropout. One should feel free to experiment with the architecture of the model and other parameters to see how the performance of the model changes.

* Conv2D: convolution mixes one function with another to reduce data space while preserving the information. Generates feature map using activation function.

* MaxPooling2D: It down-sample the image by applying a max filter.

* Flatten: It converts input from the previous layer to 1D vector and feeds into Dense layer

* Dropout: It is a regularization technique to prevent CNN from overfitting

Below is the model architecture.

![](https://cdn-images-1.medium.com/max/2000/0*_mFXP4Ymsi-HzB27)

![](https://cdn-images-1.medium.com/max/2048/0*HEwjJijVZXrRFTA2)

So now as we have created our model it’s time to train the model. Save the model after training so that we can use it later.

![](https://cdn-images-1.medium.com/max/2048/0*mqUE6NYIwZWOCmKC)

Now let’s plot the change in the model’s loss with epochs.

![](https://cdn-images-1.medium.com/max/2048/0*xKq3yr91k4R6gLwY)

![](https://cdn-images-1.medium.com/max/2000/0*jnEUZvVWAxSIwhyx)

The graph above shows the difference between train and validation loss is not much, we can say that the model if neither overfitting nor underfitting. After training it’s time to evaluate our model and check its performance on the test data.

![](https://cdn-images-1.medium.com/max/2048/0*BSYaiX3LHwM_-QTt)

Here model.evaluate() returns a list of loss value and accuracy of the model. The model’s loss on the test dataset is 0.1959 and accuracy is 94%. This trained model can be used to predict unseen images in the future.

Thanks for reading.

![](https://cdn-images-1.medium.com/max/2000/0*njMJCdsdgzTaR2lZ)

AI and machine learning enthusiast. [View all posts by csvankhede](https://csvankhede.wordpress.com/author/csvankhede/)

**Published**

*Originally published at [http://csvankhede.wordpress.com](https://csvankhede.wordpress.com/2019/09/04/convolutional-neural-network-to-diagnose-malaria/) on September 4, 2019.*
