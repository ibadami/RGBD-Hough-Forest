// Author: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Surfel.h"



struct Line {

    Eigen::Vector3f direction;
    Eigen::Vector3f point;
};

struct Plane {
    Eigen::Vector4f coefficients;
    Eigen::Vector3f point;
    Eigen::Vector3f getNormal(){
        Eigen::Vector3f normal = Eigen::Vector3f(coefficients[0], coefficients[1], coefficients[2]);
        return normal;
    }

};

struct BoundingBoxXYZ{

    float x,y,z;
    float height, width, depth;

    cv::Point3f getCenter(){ cv::Point3f center(x + width/2.f, y + height/2.f, z + depth/2.f); return center;}
    cv::Point3f getDimention(){return cv::Point3f(width, height, depth);}
};


struct MouseEvent {

    MouseEvent() {
        event = -1;
        buttonState = 0;
    }
    cv::Point pt;
    int event;
    int buttonState;

};

void onMouse( int event, int x, int y, int flags, void* userdata ) ;

// save floating point images
void saveFloatImage( char* buffer , IplImage* img);

bool isInsideRect(cv::Rect* rect, int x, int y);


bool isInsideKernel2D(float x, float y, float cx, float cy , float radius);


// Calculate PCA over mask
void calcPCA(cv::Mat &img_mask, cv::Point2f &meanPoint, cv::Size2f &dimension, float &rotAngle);

Eigen::Matrix3f quaternionToMatrix(Eigen::Quaternionf &q);

// Eigen::Quaternionf logQuaternion(Eigen::Quaternionf q);

//Eigen::Quaternionf expQuaternion(Eigen::Quaternionf q);

// generalized quaternion interpolation
Eigen::Quaternionf quatInterp(const std::vector<Eigen::Quaternionf>& rotation);

void selectConvexHull( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > >& convexHull_ ) ;

void selectPlane( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > &convexHull_, Plane &table_plane ) ;

void getObjectPointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, float minHeight, float maxHeight,
                          std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull, Plane &table_plane, Eigen::Vector3f turnTable_center, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &objectCloud  ) ;

Eigen::Vector3f getTurnTableCenter( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, Plane &table_plane ) ;

void printScore(cv::Mat &img, string &objectName, float score, cv::Point2f &pt, bool print_score );

void get3DBoundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::vector< Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& transformationMatrixOC, std::vector< cv::Point3f> &cg, cv::Point3f &bbSize);

void create3DBB(cv::Point3f &bbSize, Eigen::Matrix4f &transformationMatrixOC , cv::Size2f &img_size, std::vector< cv::Point2f > &imagePoints);

void createWireFrame (cv::Mat &img, std::vector<cv::Point2f> &vertices);

void  getLine( pcl::PointXYZRGB  &p,  Line&line);

void getLinePlaneIntersection(Line &line, Plane &plane, Eigen::Vector3f &ptIntersection);
