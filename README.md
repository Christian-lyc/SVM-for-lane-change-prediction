# SVM-for-lane-change-prediction
# C++ SVM（opencv）

This repository is meant to provide an easy-to-use implementation of the SVM classifier using the opencv. 
<br />
opencv2.4.13 is needed in this project. There's a parameter tuning program in the main.cpp. You can use it to find the best C, gamma and p.

There is also a one against all SVM train method above, you can either choose this or choose one against one SVM to train your data.

Original PR: https://github.com/Itseez/opencv/pull/5291

Refactored PR: https://github.com/Itseez/opencv/pull/6096


## Results

acc is 89.9% based on NGSIM dataset. I only use the 5 frames information to predict the lane change intention after 3 seconds 

## Questions and suggestions

You can contact me by email: 13772290@163.com
