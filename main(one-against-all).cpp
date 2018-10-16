#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "SvmSgd.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

using namespace std;
using namespace cv;
int read_csv(cv::Mat &csvm, string csv_name, char dlm = ','){
    CvMLData tper;
    tper.set_delimiter(dlm);
    int r = tper.read_csv(csv_name.c_str());    
    if (r != 0)
        return r;
    csvm = Mat(tper.get_values(), true);
    return r;
}


int main (){
    
    // Matrices with the features vectors used in the training phase and the correspondent labels
    //cv::Mat trainFeatures; // Matrix with size NxM, with N = number of training examples and M = number of features
    //cv::Mat labels; // Matrix with size Nx1

    // Matrix with all features we want to predict (N2xM, with N2 = number of feature vectors to predict)

    cv::Mat trainFeatures;
    cv::Mat trainLabels;
    cv::Mat testFeatures;
    cv::Mat testLabels;
    read_csv(trainFeatures, "traindata.csv");
    read_csv(trainLabels, "trainLabel.csv");
    read_csv(testFeatures,"testdata.csv");
    read_csv(testLabels,"testLabel.csv");

    trainLabels.convertTo(trainLabels,CV_32FC1);
    Mat train_classes = Mat::zeros(trainFeatures.rows, 3, CV_32FC1);
    
    for(int i=0; i<train_classes.rows; i++)
    {
        train_classes.at<float>(i, trainLabels.at<float>(i)-1) = 1.f;
        
    }

    testLabels=testLabels.t();
    train_classes=train_classes.t();
    

    normalize(trainFeatures,trainFeatures,1.1,-1,NORM_MINMAX);
    normalize(testFeatures,testFeatures,1.1,-1,NORM_MINMAX);


    
    CvSVM svm;
    CvSVMParams param;
    CvSVM *stor_svms=new CvSVM[train_classes.rows];
    vector<string> category_name;
    
    param.svm_type= CvSVM::C_SVC;
    param.kernel_type=CvSVM::POLY;
    param.degree=3;
    param.gamma=0.033;//0.033
    param.C=312.5;//312.5
    param.coef0=0;
    param.nu=0.5;
    param.p=0;

    for(int i=0; i<train_classes.rows; i++){

        stor_svms[i].train(trainFeatures, train_classes.row(i),Mat(),Mat(),param);
        
        //存储svm
        // string svm_filename=string("M")+category_name[i] + string("SVM.xml");
        // stor_svms[i].save(svm_filename.c_str());
    }


    std::cout << "SVM trained." << std::endl;
    
    int sum=0;
    Mat predict_classes = Mat::zeros(testFeatures.rows, 3, CV_32FC1);
    for(int j=0; j< 3; j++){
        for (int i = 0; i < testFeatures.rows; ++i){

            float Label =stor_svms[j].predict(testFeatures.row(i));

            predict_classes.at<float>(i, j) = Label;

            
        }
        
    }
    
    
    float truth_label;
    for(int i=0; i<testFeatures.rows; i++)
    {
        for(int j=0; j< 3; j++){
            if(predict_classes.at<float>(i, j) ==1){
                truth_label=j+1;
                
                if (testLabels.at<float>(i)==(int)truth_label){
                    sum++;
                }
            }
        }
        cout << "Predicted label"<<i<<": "<<truth_label<<endl;
        cout << "Ture label"<<i<<": "<<testLabels.at<float>(i) <<endl;

    }

    cout << "预测准确率为："<<(double)sum/218<<endl;
    return 0;
}

