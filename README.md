# DL-project-3
This is the project 2 for MSBD6000B Deep Learning Course

Author:
CHANG, YAJIE    20459996           
XU, CHAOYI      20461901      
WANG, WEIXIAO   20476516      
QIU, FENG       20398324      

Breast cancer is the most common cancer in women worldwide. In this project, you are required to classify the x-ray images into normal and abnormal to help detect the cancer.
There are two data sets for the project on the Dropbox: Dataset_A.zip  Dataset_B.zip

Please refer to the Readme file in the dataset for the information you need.

Multi-instance classification
  Each medical image has much higher resolution than natural images, so it is hard to store such big feature mappings in the GPU. Since resizing the image may lose some important details, you need to devide the image into many patches. By regarding each patch as a instance, you can use image-level labels to conduct deep multi-instance learning.
  Only the Dataset_A is used in this task.
  Except downsampling the images, any additional pre- or post-processing on the training set is allowed.
  Suggested reference: Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification, CVPR 2016.
  Coupon will be given if you improve the mentioned method or develop your own solution.
  
Grading will be based on the testing accuracy, please upload the predicted results of both tasks in csv format.
The source codes should be uploaded to github.
You need to write a report to describe the details of your implementation and the report should be put in github.
