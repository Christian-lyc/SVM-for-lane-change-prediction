#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

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

    // Predicted label for the new feature vector
    int Label;

    // Create simple matrices for demonstration (dummy examples)

    cv::Mat trainFeatures;
    cv::Mat trainLabels;
    cv::Mat testFeatures;
    cv::Mat testLabels;
    read_csv(trainFeatures, "traindata.csv");
    read_csv(trainLabels, "trainLabel.csv");
    read_csv(testFeatures,"testdata.csv");
    read_csv(testLabels,"testLabel.csv");
    trainFeatures.convertTo(trainFeatures, CV_32FC1); 
    testFeatures.convertTo(testFeatures,CV_32FC1);
    testLabels.convertTo(testLabels,CV_32FC1); 
    testLabels=testLabels.t();
    //cout << trainFeatures << endl;
    normalize(trainFeatures,trainFeatures,1.1,-1,NORM_MINMAX);
    normalize(testFeatures,testFeatures,1.1,-1,NORM_MINMAX);
    
    CvSVM svm;
    CvSVMParams param;

    param.svm_type= CvSVM::C_SVC;
    param.kernel_type=CvSVM::POLY;
    param.degree=3;
    param.gamma=0.033;
    param.C=312.5;
    param.coef0=0;
    param.nu=0.5;
    param.p=0;
    svm.train(trainFeatures, trainLabels,Mat(),Mat(),param);

    //调参程序

    // CvParamGrid nuGrid = CvParamGrid(1,1,0.0);  
    // CvParamGrid coeffGrid = CvParamGrid(1,1,0.0);  
    // CvParamGrid degreeGrid = CvParamGrid(1,1,0.0); 

    // svm.train_auto(trainFeatures,trainLabels,Mat(),Mat(),param,10,
    // svm.get_default_grid(CvSVM::C),
    // svm.get_default_grid(CvSVM::GAMMA),
    // svm.get_default_grid(CvSVM::P),
    // nuGrid,
    // coeffGrid,
    // degreeGrid);

    // CvSVMParams params_re = svm.get_params();  
    // svm.save("training_srv.xml");  
    // float C = params_re.C;  
    // float P = params_re.p;  
    // float gamma = params_re.gamma;  
    // printf("\nParms: C = %f, P = %f,gamma = %f \n",C,P,gamma); 

    std::cout << "SVM trained." << std::endl;
    
    int sum=0;
    for (int i = 0; i < testFeatures.rows; ++i){
        // Predict label for the new feature vector (newFeature should be a 1xM matrix)
        Label =svm.predict(testFeatures.row(i));
        float truth_label=testLabels.at<float>(i);
        if (Label==(int)truth_label){
            sum++;
        }
        std::cout << "Predicted label" << i << ": " << Label << std::endl;
        std::cout << "Ture label" << i <<": " << truth_label << std::endl;
    }
    cout << "预测准确率为："<<(double)sum/218<<endl; //218 is testFeatures.rows
    return 0;
}

