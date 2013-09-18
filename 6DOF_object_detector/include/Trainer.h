// Author: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#ifndef CRFORESTTRAINING_H
#define CRFORESTTRAINING_H

#include "utils.h"
#include "Parameters.h"
#include "Pixel.h"


struct rawData {

    vector<vector<string> > vFilenames;
    vector<vector<CvRect> > vBBox;
    vector<vector<CvPoint> > vCenter;
    vector<vector<float> > vPoseAngle; // in degrees
    vector<vector<float> > vPitchAngle; // in degrees
    vector<string> internal_files;
    std::vector<cv::Point3f> cg;
    cv::Point3f bbSize;

};

class CRForestTraining
{
public:
    CRForestTraining();

    static void generateNewImage(cv::Mat &mask, cv::Mat &img, cv::Mat &newImg );
    static void generateTrainingImage(cv::Mat &rgbImage, cv::Mat &depthImage );

    // Extract patches from training data
    static void extract_Pixels( rawData& data , const Parameters &p, CRPixel& Train, cv::RNG* pRNG );
    static void generate3DModel( Parameters& param, vector< vector<string> >& vFilenames, vector<vector<CvPoint> >& vCenter, vector< vector<  CvRect > > &vBBox , vector<vector<float> > &vPoseAngle, vector<vector<float> > &vPitchAngle, vector<cv::Point3f> &cg, cv::Point3f &bbSize );
    static void getObjectSize(Parameters &p, vector< vector<string> >& vFilenames, vector<vector<CvPoint> >& vCenter, vector< vector<  CvRect > > &vBBox );
};

#endif // CRFORESTTRAINING_H
