// Author: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#ifndef SURFEL_H_
#define SURFEL_H_

#include <omp.h>
#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "utils.h"
//#include <pcl/console/parse.h>

struct SurfelFeature{
    SurfelFeature() {}

    float fVector[4];
    void print() const {
        std::cout << "feature vector = " << fVector[0]<< " "<< fVector[1]<< " "<< fVector[2]<< " "<< fVector[3]<< std::endl;
    }

};

class Surfel {

public:
    static void imagesToPointCloud(const cv::Mat& depthImg, const cv::Mat& colorImg, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);
    static void imagesToPointCloud_( cv::Mat& depthImg, cv::Mat& colorImg, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, cv::Mat &mask );
    static void houghPointCloud( std::vector<cv::Mat>& houghImg, std::vector<float> &scales,  pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, float max_val );
    static void computeSurfel(pcl::PointCloud<pcl::Normal>::Ptr normals, cv::Point2f pt1, cv::Point2f pt2, cv::Point2f center, SurfelFeature &sf, float depth1, float depth2);
    static void calcSurfel2CameraTransformation(cv::Point3f &s1, cv::Point3f &s2, pcl::Normal &n1, pcl::Normal &n2, Eigen::Matrix4f &TransformationSC1, Eigen::Matrix4f &TransformationSC2);
    static void calcQueryPoint2CameraTransformation(cv::Point3f &s1, cv::Point3f &s2, cv::Point3f &query_point, const pcl::Normal &qn1, Eigen::Matrix4f &TransformationQueryC1, Eigen::Matrix4f &TransformationQueryC2);
    static void addCoordinateSystem( Eigen::Matrix4f &transformationMatrixOC, boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, string id);
    static void addCoordinateSystem(const Eigen::Matrix4f &transformationMatrixOC, boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, string id);
};


#endif /* SURFEL_H_ */
