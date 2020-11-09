---
title: Blazeface Face detection
tags: [Deep Learning, Face Detection, MobileNet]
style: border
color: success
description: Implemetation of Blaze Face model for face detection. 
---


In computer vision face detection is very first building block in many applications. There are many techniques available for face detection, here is a nice blog for more reading.

Some of the practical application which uses face detection as building block are as below

* Face lock in phones

* Identify people on social media

* Lip reading

* Emotion detection

* Face recognition

* Face augmentation

There exists some face detection techniques. Select the right one based on your requirements, have a look at this [blog](https://medium.com/@Intellica.AI/a-guide-for-building-your-own-face-detection-recognition-system-910560fe3eb7) to find more info.

![Face detection techniques](https://cdn-images-1.medium.com/max/2000/1*ZLYUrNgoPyvhU1aORGPxzg.png)*Face detection techniques*

As face detection is widely used in mobile phone applications. So which one will be right choice to work well with mobile devices? A recent Research paper by google introduced a light weight and well-performing face detector which works well with mobile GPU, giving real-time performance.

![BlazeFace](https://cdn-images-1.medium.com/max/2000/1*PaM_aVmzVNB_IVfcKsqVmg.png)*BlazeFace*

The paper provided details of the main building blocks of the model. We can recreate model using the architecture provided in the paper. There are two main building blocks described in paper named single BlazeBlock and double BlazeBlock.

![](https://cdn-images-1.medium.com/max/2000/1*SNhogwDamDUEXqXV0I1G-g.png)

In these blocks Depthwise convolutional layers are used which boosts the speed of the model. The 5x5 kernel is used for DW conv and 1x1 kernel for conv layers. The model uses 5 single BlazeBlocks and 6 double BlazeBlocks.

![Model architecture](https://cdn-images-1.medium.com/max/2000/1*Y3iOn1CZtn1KU4almFcwOA.png)*Model architecture*

For anchor computation two Conv layers with 16x16 kernel size and 8x8 kernel size is used with number of 2 and 8 filters respectively. Output of the model requires post processing, as the number of anchors overlaping increases significantly with the size of the object. This prediction is first croped to fit in the borders of the image dimensions, then NMS is applied to get the final boxes. These boxes can be resized to original size of the image as the input image was resized to 128x128. The final bounding boxes can be randered on the image using cv2.rectangle.

The final detected face can be further used as input to another model for specific task.

A tflite model of the blazeface can be found [here](https://github.com/google/mediapipe/blob/master/mediapipe/models/face_detection_front.tflite). We can extract layer details and model architecture as below.

    tflite_path = 'face_detection_front.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()

    for i in interpreter.get_tensor_details():
        print("name|shape ",i["name"],"|",i["shape"])

Here we need the weights of the kernels and bias. We can restore the weight for our model by assigning it to layers of the model.

    for var in model.variables:
        weight = tflite_map[var.name]
        var.assign(weight)

Predict the face bounding box and render it using cv2.

    for bx in final_boxes:
        cv2.rectangle(orig_image, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 10)

    plt.figure(figsize=(15,15))
    plt.imshow(orig_frame)
    plt.show()

![edited emoji image](https://cdn-images-1.medium.com/max/2000/1*1rJYvPgApZ_8rcaVmUdg7w.png)*edited emoji image*

Save the model so it can be used for further task such as face recognition, face augmentation, etc. The paper BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs is available [here](https://arxiv.org/pdf/1907.05047.pdf).

Thanks for reading…..
