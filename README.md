# har-in-dark

This project was mainly tested on the dark environment data set, tested different frame sampling effects, and used image enhancement technology. The images were first classified using traditional svm. (We performed post-fusion of frames in our experiments, so the actual action detection problem became an image classification problem). Then use the pre-trained resnet50 that comes with pytorch for training, and also verify its effect.

## File and Folder Structure

```
├──  Low-light-Image-Enhancement-master - The file primarily utilizes low-light image enhancement techniques.
│    └── exposure_enhancement.py
│    └── utils.py  -These two files contain the core code for low-light image enhancement techniques.
│    └── exposure_enhancement.py
│    └── demo.py
│    └── demo_test.py  -These two files mainly perform image enhancement on the test set and training set images, with the only difference being the respective paths.
│
│
├──  HAR_sampling.py  -Here, the video is processed using uniform sampling techniques to save frames from the original video at every two frames' interval.
│ 
│
├──  feature_extraction.py  -a pre-trained large model is used to extract features from images, resulting in a feature dimension of 512
│
│
├──  svm.py   -SVM (Support Vector Machine) was employed to classify the extracted features.
│
│
├── svm_enhance.py  - SVM (Support Vector Machine) was employed to classify the extracted features(after enhance).
│
│
├── svm.ipynb  - - SVM (Support Vector Machine) was employed to classify the extracted features(after enhance).Jupyter format was primarily used for ease of debugging.
│  
│
├── resnet50.py  - this folder contains end to end learning methond resnet50 to classify
![]https://github.com/Songyu8/har-in-dark/blob/master/images/frame_1.png
![]https://github.com/Songyu8/har-in-dark/blob/master/images/output.png
![]https://github.com/Songyu8/har-in-dark/blob/master/images/frame_0_DUAL_g0.6_l0.15.png
![]https://github.com/Songyu8/har-in-dark/blob/master/images/sssss1.png
```
