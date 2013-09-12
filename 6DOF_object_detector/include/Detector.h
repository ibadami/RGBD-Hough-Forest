/*
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

// Modified by: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#pragma once

#include <stdio.h>
#include <vector>

#include "Forest.h"
#include "Candidate.h"
#include "Parameters.h"
#include "utils.h"

// Auxilary structure
struct XYZIndex {
    int val;
    unsigned int index;
    bool operator<(const XYZIndex& a) const { return val<a.val; }
};

class CRForestDetector {
public:
    // Constructor
    CRForestDetector(const CRForest* pRF, int w, int h, double s_points=-1.0 ,double s_forest=-1.0, bool bpr = true) : crForest(pRF), width(w), height(h), sample_points(s_points),do_bpr(bpr){
        crForest->GetClassID(Class_id);
    }

    // Detection functions
public:
    void detectObject(const cv::Mat& img, const cv::Mat& depthImg, const vector<cv::Mat>& vImg,  const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::vector< cv::Mat >& vImgAssign, const std::vector<cv::Mat>& classProbs, const Parameters& p, int this_class, std::vector<Candidate >& candidates);

    void voteForCandidate( std::vector< cv::Mat> vimgAssign, Candidate& new_cand, int kernel_width, float max_width, float max_height  );

    void getClassConfidence(const std::vector<cv::Mat>& vImgAssign,std::vector<cv::Mat>& classConfidence);

    void fullAssignCluster(const cv::Mat &img, const cv::Mat &depthImg, vector< cv::Mat > &vImgAssign, const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals);

    void trainStat(IplImage* img, CvRect bbox, std::vector< std::vector<float> >& stat, float inv_set_size = 1.0f);

    void transposeMatrix(cv::Mat src, cv::Mat &dst, int order, int nScales);


private:
    void assignCluster(const cv::Mat &img, const cv::Mat &depthImg, vector<cv::Mat> &vImgAssign, const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals);

    void voteForCenter(const std::vector<cv::Mat>& vImgAssign, std::vector< std::vector<cv::Mat> >& vImgDetect, const  cv::Mat& depthImg, std::vector< std::vector< std::vector< std::vector< std::vector< std::pair< cv::Point, int > > > > > >& voterImages, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::vector<float>& scales, int& this_class, cv::Rect* focus, const float& prob_threshold, const std::vector<cv::Mat>& classProbs, const Parameters& param, bool addPoseInformation = false,  bool addScaleInformation = false  );

    void detectCenterPeaks(std::vector<Candidate >& candidates, const std::vector<std::vector<cv::Mat> >& imgDetect, const std::vector<cv::Mat>& vImgAssign, const std::vector< std::vector< std::vector< std::vector<std::vector< std::pair< cv::Point, int > > > > > >& voterImages, const  cv::Mat& depthImg, const cv::Mat& img, const Parameters& param, int this_class);

    void voteForPose(const cv::Mat img, const std::vector< std::vector< std::vector< std::vector<std::vector< std::pair< cv::Point, int > > > > > >& voterImages, const std::vector<cv::Mat>& vImgAssign, const std::vector<std::vector<cv::Mat> >& vImgDetect, std::vector<Candidate>& candidates, const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const int kernel_width, const std::vector<float>&scales, const float thresh, const bool DEBUG, const bool addPoseScore);

    void detectPosePeaks(vector< cv::Mat > &positiveAcc, vector< cv::Mat> &negativeAcc, Eigen::Matrix3f &positiveFinalOC, Eigen::Matrix3f &negativeFinalOC);

    void detectPosePeaks_meanShift(std::vector<std::vector<float> >&Qx, std::vector<std::vector<float> > &Qy,std::vector<std::vector<float> > &Qz, Eigen::Matrix3f &positiveFinalOC, Eigen::Matrix3f & negativeFinalOC);

    void detectPosePeaks_meanShift_common(std::vector<float> &Qx, std::vector<float> &Qy, std::vector<float> &Qz, std::vector<float> &Qw, Eigen::Matrix3f & finalOC);

    void detectPosePeaks_slerp(std::vector<Eigen::Quaternionf>& qMean,Eigen::Matrix3f &finalOC);

    void detectMaxima(const vector<vector<cv::Mat> >& poseHoughSpace,  Eigen::Quaternionf& finalOC, int& step, float& score);
//    void detectMaxima(const vector<cv::Mat> & poseHoughSpace,  Eigen::Quaternionf& finalOC, int& step, float& score);

    void detectMaximaK_means(std::vector<Eigen::Quaternionf>& qMean, Eigen::Matrix3f &finalOC);

    void axisOfSymmetry(std::vector<Eigen::Quaternionf>& qMean, Eigen::Quaternionf &qfinalOC );


    /*************************************************************************************************************************************************/

public:
    // Get/Set functions
    unsigned int GetNumLabels() const { return crForest->GetNumLabels(); }
    void GetClassID(std::vector<std::vector<int> >& v_class_ids){ crForest->GetClassID( v_class_ids ); }
    const CRForest* GetCRForest(){return crForest;}

    //private variables
private:
    const CRForest* crForest;
    std::vector<std::vector<int> > Class_id;
    int width;
    int height;
    double sample_points;
    bool do_bpr;
};
