---
title: ESRGAN using keras
tags: [Deep Learning, GAN, Keras]
style: fill
color: warning
description: ESRGAN implemetation using keras. An enhanced version of the SRGAN by modifying the model architecture and loss functions.
---


ESRGAN is the enhanced version of the SRGAN. Authors of the ESRGAN tried to enhance the SRGAN by modifying the model architecture and loss functions.

## GAN

Before diving into the ESRGAN first let’s get a high-level understanding of the GAN. GANs are capable of generating Fake data that looks realistic. Some of the GAN applications are to enhance the quality of the image. The high-level architecture of the GAN contains two main networks namely the** generator network** and the **discriminator network**. The generator network tries to generate the fake data and the discriminator network tries to distinguish between real and fake data, hence helping the generator to generate more realistic data.

![](https://cdn-images-1.medium.com/max/2000/1*NqNDgEmAZHhFOyNLp4sJ3w.png)

## ESRGAN

The main architecture of the ESRGAN is the same as the SRGAN with some modifications. ESRGAN has Residual in Residual Dense Block(RRDB) which combines multi-level residual network and dense connection without Batch Normalization.

**Network architecture**

![](https://cdn-images-1.medium.com/max/2048/1*Mr7NA-EEcdYvdlQMKIpr3w.png)

**Residual in Residual Dense Block(RRDB)**

![](https://cdn-images-1.medium.com/max/2048/1*mqxL9uUy2RHmXiYCsnPKjw.png)

    from keras.layers import Add, Concatenate, LeakyReLU, Conv2D, Lambda

    def dense_block(inpt):
        """
        Dense block containes total 4 conv blocks with leakyRelu 
        activation, followed by post conv layer

        Params: tensorflow layer
        Returns: tensorflow layer
        """
        b1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(inpt)
        b1 = LeakyReLU(0.2)(b1)
        b1 = Concatenate()([inpt,b1])

        b2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b1)
        b2 = LeakyReLU(0.2)(b2)
        b2 = Concatenate()([inpt,b1,b2]) 

        b3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b2)
        b3 = LeakyReLU(0.2)(b3)
        b3 = Concatenate()([inpt,b1,b2,b3])

        b4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b3)
        b4 = LeakyReLU(0.2)(b4)
        b4 = Concatenate()([inpt,b1,b2,b3,b4])

        b5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b4)
        b5 = Lambda(lambda x:x*0.2)(b5)
        b5 = Add()([b5, inpt])
        
        return b5

    def RRDB(inpt):
        """
        RRDB(residual in residual dense block) contained three dense  
        block, each block followed by beta contant multiplication(0.2) 
        and addition with dense block input layer.

        Params: tensorflow layer
        Returns: tensorflow layer
        """
        x = dense_block(inpt)
        x = dense_block(x)
        x = dense_block(x)
        x = Lambda(lambda x:x*0.2)(x)
        out = Add()([x,inpt])

        return out

**Relativistic Discriminator**

Besides using standard discriminator ESRGAN uses the relativistic GAN, which tries to predict the probability that the real image is relatively more realistic than a fake image.

![](https://cdn-images-1.medium.com/max/2018/1*jq3iXHKxy0boSM60DK_zAw.png)

    from keras import backend as K

    def relativistic_loss(x):
        real, fake = x
        fake_logits = K.sigmoid(fake - K.mean(real))
        real_logits = K.sigmoid(real - K.mean(fake))
                
        return [fake_logits, real_logits]

The discriminator loss and adversarial loss is then defined as below.

![](https://cdn-images-1.medium.com/max/2918/1*cdyehCB3VHGcA8IoUdLm1g.png)

![](https://cdn-images-1.medium.com/max/2876/1*XU9dPRswc2w8GWvELCAXvQ.png)

    dis_loss =
    K.mean(K.binary_crossentropy(K.zeros_like(fake_logits),fake_logits)+                    K.binary_crossentropy(K.ones_like(real_logits),real_logits))

    gen_loss = K.mean(K.binary_crossentropy(K.zeros_like(real_logit),real_logit)+K.binary_crossentropy(K.ones_like(fake_logit),fake_logit))

**Perceptual loss**

A more effective perceptual loss is introduced by constraining features before the activation function.

    from keras.applications.vgg19 import preprocess_input

    generated_feature = vgg(preprocess_vgg(img_hr))
    original_fearure = vgg(preprocess_vgg(gen_hr))

    percept_loss = tf.losses.mean_squared_error(generated_feature,original_fearure)

![Representative feature maps before and after activation for image ‘baboon’. With the network going deeper, most of the features after activation become inactive while features before activation contains more information.](https://cdn-images-1.medium.com/max/2446/1*UW8ekH0MX73tg1wkjlHEyw.png)*Representative feature maps before and after activation for image ‘baboon’. With the network going deeper, most of the features after activation become inactive while features before activation contains more information.*

**Training details**

ESRGAN scales the Low Resolution(LR) image to a High-Resolution image with an upscaling factor of 4.

For optimization, Adam optimizer is used with default values.

**References**

[**ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks**
*The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic…*arxiv.org](https://arxiv.org/abs/1809.00219)
[**Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**
*Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional…*arxiv.org](https://arxiv.org/abs/1609.04802)
