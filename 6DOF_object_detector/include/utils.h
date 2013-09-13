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

    Eigen::Vector3d direction;
    Eigen::Vector3d point;
};

struct Plane {
    Eigen::Vector4d coefficients;
    Eigen::Vector3d point;
    Eigen::Vector3d getNormal(){
        Eigen::Vector3d normal = Eigen::Vector3d(coefficients[0], coefficients[1], coefficients[2]);
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

Eigen::Matrix3d quaternionToMatrix(Eigen::Quaterniond &q);

// Eigen::Quaterniond logQuaternion(Eigen::Quaterniond q);

//Eigen::Quaterniond expQuaternion(Eigen::Quaterniond q);

// generalized quaternion interpolation
Eigen::Quaterniond quatInterp(const std::vector<Eigen::Quaterniond>& rotation);

void selectConvexHull( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > >& convexHull_ ) ;

void selectPlane( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > &convexHull_, Plane &table_plane ) ;

void getObjectPointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, float minHeight, float maxHeight,
                          std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > convexHull, Plane &table_plane, Eigen::Vector3d turnTable_center, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &objectCloud  ) ;

Eigen::Vector3d getTurnTableCenter( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, Plane &table_plane ) ;

void printScore(cv::Mat &img, string &objectName, float score, cv::Point2f &pt, bool print_score );

void get3DBoundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::vector< Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >& transformationMatrixOC, std::vector< cv::Point3f> &cg, cv::Point3f &bbSize);

void create3DBB(cv::Point3f &bbSize, Eigen::Matrix4d &transformationMatrixOC , cv::Size2f &img_size, std::vector< cv::Point2f > &imagePoints);

void createWireFrame (cv::Mat &img, std::vector<cv::Point2f> &vertices);

void  getLine( pcl::PointXYZRGB  &p,  Line&line);

void getLinePlaneIntersection(Line &line, Plane &plane, Eigen::Vector3d &ptIntersection);
