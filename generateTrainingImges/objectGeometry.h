#ifndef OBJECTGEOMETRY_H
#define OBJECTGEOMETRY_H

#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>


#define PI 3.14159265

struct positions{

    std::vector< cv::Point3f > objPose;

    void getpositions( int num, bool trainNeg ){

        objPose.reserve( num );
        if( !trainNeg ){

            float step = 180.f / ( float(num) );

            for( int i = 0; i <= num; i++ ){

                float angle = ( step * ( i ) - step ) ;

                cv::Point3f op;
                op.x = 0.22f * std::cos( angle* PI / 180.f );
                op.y = 0.22f * std::sin( angle* PI / 180.f );
                op.z = 0;
                objPose.push_back( op );

            }
        }else{

            cv::Point3f op;

            ///////////////////////// row -1 ///////////////////////

            op.x = -0.30f; op.y = -0.20f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.15f; op.y = -0.20f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.f; op.y = -0.20f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.15f; op.y = -0.20f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.30f; op.y = -0.20f; op.z = 0.f;
            objPose.push_back( op );


            ///////////////////////////// row 0 ///////////////////

            op.x = -0.37f; op.y = 0.0f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.225f; op.y = 0.0f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.075f; op.y = 0.0f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.075f; op.y = 0.0f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.225f; op.y = 0.0f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.37f; op.y = 0.0f; op.z = 0.f;
            objPose.push_back( op );


            ///////////////////////////  row 1 ////////////////////////

            op.x = -0.45f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.30f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.15f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.15f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.30f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.45f; op.y = 0.2f; op.z = 0.f;
            objPose.push_back( op );


            ///////////////////////////  row 2 ////////////////////////

            op.x = -0.55f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.37f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.225f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.075f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.075f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.225f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.37f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.55f; op.y = 0.45f; op.z = 0.f;
            objPose.push_back( op );

            ////////////////////////   row 3   ///////////////////////

            op.x = -0.50f; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.30f; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.10; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.f; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.10f; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.30f; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.50f; op.y = 0.7f; op.z = 0.f;
            objPose.push_back( op );

            // ////////////////// row 4 /////////////////////////////

            op.x = -0.77f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.57f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.32f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = -0.075f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.075f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.32f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.57f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            op.x = 0.77f; op.y = 1.f; op.z = 0.f;
            objPose.push_back( op );

            ////////////////////////   row 5   ///////////////////////

            //            op.x = -0.80f; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );

            //            op.x = -0.65f; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );

            //            op.x = -0.35f; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );

            //            op.x = -0.05; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );

            //            op.x = 0.35f; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );

            //            op.x = 0.65f; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );

            //            op.x = 0.80f; op.y = 1.2f; op.z = 0.f;
            //            objPose.push_back( op );



        }
    }
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

void onMouse( int event, int x, int y, int flags, void* userdata ) {
    if( userdata ) {
        MouseEvent* data = (MouseEvent*) userdata;
        data->event = event;
        data->pt = cv::Point( x, y );
        data->buttonState = flags;
    }
}



inline cv::Point3f P3toR3(cv::Point2f &pixelCoordinates, cv::Point2f &center, float depth){

    float focal_length = 525.f; // in pixel
    cv::Point3f realCoordinates;
    realCoordinates.z = depth; // in meter
    realCoordinates.x = (pixelCoordinates.x - center.x)* depth / focal_length;
    realCoordinates.y = (pixelCoordinates.y - center.y)* depth / focal_length;
    return realCoordinates;
}

inline void R3toP3(cv::Point3f &realCoordinates, cv::Point2f &center, cv::Point2f &pixelCoordinates, float &depth){

    float focal_length = 525.f;
    depth = realCoordinates.z;
    if (depth == 0){
        pixelCoordinates.x = 0;
        pixelCoordinates.y = 0;
    }else{
        pixelCoordinates.x = realCoordinates.x * focal_length / depth + center.x;
        pixelCoordinates.y = realCoordinates.y * focal_length / depth + center.y;
    }
}



inline void debugFunction( pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud){

    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_debug (new pcl::PointCloud< pcl::PointXYZRGB >);

    cloud_debug->is_dense = true;
    cloud_debug->height = cloud->height;// height;
    cloud_debug->width = cloud->width; // width;
    cloud_debug->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 0.f );
    cloud_debug->sensor_orientation_ = Eigen::Quaternionf::Identity();
    cloud_debug->points.reserve( cloud->size() ) ;

    for( int c = 0;  c < cloud->size(); c++ ){

        pcl::PointXYZRGB p = cloud->at(c);
        p.z = 0;
        cloud_debug->points.push_back(p);
    }

    // show cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("3D Viewer") );
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_debug_rgb( cloud_debug );

    viewer->setBackgroundColor( 1, 1, 1 );
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud_debug, cloud_debug_rgb, "changed cloud");
    viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );

    while (!viewer->wasStopped ()){

        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

    }
}

inline void pointCloud2Images( pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, cv::Mat& depthImg, cv::Mat& colorImg, cv::Size2f& imgSize ) {

    float depthMax = 2.f;
    cv::RNG rng(cv::getCPUTickCount());

    cv::Point2f center( imgSize.width/2.f, imgSize.height/2.f );
    colorImg = cv::Mat::zeros( int(imgSize.height), int(imgSize.width), CV_8UC3 );
    depthImg = cv::Mat( int(imgSize.height), int(imgSize.width), CV_32FC1, -1 );
    cv::Vec3b rand_bgr( rng.operator ()(255), rng.operator ()(255), rng.operator ()(255) );


    for( int p = 0; p< cloud->size(); p++ ){

        pcl::PointXYZRGB cloudPt = cloud->at( p );
        cv::Point3f pt( cloudPt.x, cloudPt.y, cloudPt.z );
        cv::Vec3b bgr( cloudPt.b, cloudPt.g, cloudPt.r );


        cv::Point2f pixel;
        float depth;// in meters
        R3toP3( pt, center, pixel, depth );


        // check if pixel is in the image
        if(int(pixel.x) < 0 || int(pixel.x) >= int(imgSize.width) ||int(pixel.y) < 0 || int(pixel.y) >= int(imgSize.height) )
            continue;

        // check if the pixel is already filled before has higher depth then new pixel
        if( depthImg.at< float >( int(pixel.y), int(pixel.x) ) < 0.f ){

            if( depth > depthMax ){
                depth = depthMax;
                bgr = rand_bgr;
            }

            depthImg.at< float >( int(pixel.y), int(pixel.x) ) = depth;
            colorImg.at< cv::Vec3b >( int(pixel.y), int(pixel.x) ) = bgr;

        }
        if( depthImg.at<float >( int(pixel.y), int(pixel.x) ) > depth ){

            if( depth > depthMax ){
                depth = depthMax;
            }

            depthImg.at< float >( int(pixel.y), int(pixel.x)) = depth;
            colorImg.at< cv::Vec3b >( int(pixel.y), int(pixel.x) ) = bgr;
            bgr = rand_bgr;
        }


    }

    //        cv::imshow("colorimg", colorImg);
    //        cv::imshow("depthImg", depthImg);
    //        cv::waitKey(0);

}


inline void images2PointCloud( cv::Mat& depthImg, cv::Mat& colorImg, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, cv::Mat &mask ) {


    //    cv::imshow("img", colorImg);
    //    cv::waitKey(0);
    //    cv::destroyWindow("img");

    //cloud->header = depthImg.header;//
    cloud->is_dense = true;
    cloud->height = colorImg.rows;// height;
    cloud->width = colorImg.cols; // width;
    cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 0.f );
    cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    //  cloud->sensor_orientation_ = Eigen::Vector4f( std::cos(PI/16), 0.f, 0.f, -std::sin(PI/16) );
    cloud->points.resize( colorImg.rows * colorImg.cols ) ;

    const float invfocalLength = 1.f / 525.f;
    const float centerX = colorImg.cols / 2.f;
    const float centerY = colorImg.rows / 2.f;

    // check if mask is null data pointer
    if (mask.empty())
        mask = cv::Mat::ones(depthImg.rows, depthImg.cols, CV_8UC1);

    cv::Point pt;
    int idx = 0;
    for ( int y = 0; y < colorImg.rows; y++ ) {
        for ( int x = 0; x < colorImg.cols; x++ ) {

            if( mask.at<unsigned char>(y,x) > 0 ){

                pcl::PointXYZRGB& p = cloud->points[ idx ];

                pt.x = x;
                pt.y = y;
                float dist = depthImg.at< unsigned short >( pt ) / 1000.0f; // convert it into meter

                //      if ( dist == 0 ) {
                //        p.x = std::numeric_limits< float >::quiet_NaN();
                //        p.y = std::numeric_limits< float >::quiet_NaN();
                //        p.z = std::numeric_limits< float >::quiet_NaN();
                //      }
                //      else {

                float xf = x;
                float yf = y;
                p.x = ( xf - centerX ) * dist * invfocalLength;
                p.y = ( yf - centerY ) * dist * invfocalLength;
                p.z = dist;
                //      }
                //        float x_old = ( xf - centerX ) * dist * invfocalLength;
                //        float y_old = ( yf - centerY ) * dist * invfocalLength;
                //        float z_old = dist;
                //
                //        // rotate with angle theta
                //        float theta = -0.38f;
                //        p.x = x_old;
                //        p.y = y_old * cos(theta) - z_old * sin(theta) - .26;
                //        p.z = y_old * sin(theta) + z_old * cos(theta);

                //      }

                //depthdata++;

                float r = colorImg.at<cv::Vec3b>( pt )[ 2 ];
                float g = colorImg.at<cv::Vec3b>( pt )[ 1 ];
                float b = colorImg.at<cv::Vec3b>( pt )[ 0 ];

                pcl::RGB rgb;
                rgb.r = (uint8_t)r;
                rgb.g = (uint8_t)g;
                rgb.b = (uint8_t)b;
                rgb.a = 1;

                p.rgba = rgb.rgba;
                idx++;

            }
        }
    }
}

inline void selectPlane( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > &convexHull_, Plane &table_plane ) {

    // let user select convex hull points in the images
    std::cout << "select convex hull in the image\n";
    std::cout << "left click: add point, right click: finish selection\n";

    cv::Mat img_cv;
    cv::cvtColor( img_rgb, img_cv, cv::COLOR_BGR2GRAY );

    cv::Mat cameraMatrix, distortionCoeffs;
    //    getCameraCalibration( cameraMatrix, distortionCoeffs ); // how to get these??
    cameraMatrix = cv::Mat::zeros( 3, 3, CV_32FC1);
    cameraMatrix.at<float>(0,0) = 525.f; //fx
    cameraMatrix.at<float>(1,1) = 525.f; //fy
    cameraMatrix.at<float>(0,2) = img_rgb.cols / 2.f; // cx
    cameraMatrix.at<float>(1,2) = img_rgb.rows / 2.f; // cy
    cameraMatrix.at<float>(2,2) = 1;

    distortionCoeffs = cv::Mat::zeros( 1, 4, CV_32FC1);

    // wait for user input
    // left click: save clicked point in reference frame
    // right/middle click: stop selection

    bool stopSelection = false;
    while( !stopSelection ) {

        // project selected points into image
        cv::Mat img_viz = img_rgb.clone();
        if( convexHull_.size() > 0 ) {

            std::vector< cv::Point3f > convexHullCamera( convexHull_.size() );
            for( unsigned int j = 0; j < convexHull_.size(); j++ ) {

                // transform point from reference frame to camera frame
                Eigen::Vector4d p;
                p[ 0 ] = convexHull_[ j ]( 0 );
                p[ 1 ] = convexHull_[ j ]( 1 );
                p[ 2 ] = convexHull_[ j ]( 2 );
                p[ 3 ] = 1;

                p = ( referenceTransform.inverse() * p ).eval();
                convexHullCamera[ j ].x = p[ 0 ];
                convexHullCamera[ j ].y = p[ 1 ];
                convexHullCamera[ j ].z = p[ 2 ];

            }

            std::vector< cv::Point2f > imagePoints( convexHullCamera.size() );
            cv::Mat rot( 3, 1, CV_64FC1, 0.f );
            cv::Mat trans( 3, 1, CV_64FC1, 0.f );
            cv::projectPoints( cv::Mat( convexHullCamera ), rot, trans, cameraMatrix, distortionCoeffs, imagePoints );

            for( unsigned int j = 0; j < imagePoints.size(); j++ ) {

                if( imagePoints[ j ].x < 0 || imagePoints[ j ].x > img_cv.cols - 1 )
                    continue;

                if( imagePoints[ j ].y < 0 || imagePoints[ j ].y > img_cv.rows - 1 )
                    continue;

                cv::Scalar c( 0.f, 0.f, 255.f );
                cv::circle( img_viz, imagePoints[ j ], 10, c, -1 );

            }
        }

        int displayHeight = 240;
        double imgScaleFactor = ( (float) displayHeight ) / ( (float) img_viz.rows );
        if( img_viz.rows != displayHeight ) {
            cv::Mat tmp;
            cv::resize( img_viz, tmp, cv::Size(), imgScaleFactor, imgScaleFactor, cv::INTER_LINEAR );
            tmp.copyTo( img_viz );
        }
        cv::imshow( "Select turn table Plane", img_viz );

        MouseEvent mouse;
        cv::setMouseCallback( "Select turn table Plane", onMouse, &mouse );
        cv::waitKey( 10 );

        if( mouse.event == CV_EVENT_LBUTTONDOWN ) {

            // find corresponding 3D position in point cloud
            float img2cloudScale = ( (float) cloud->height ) / ( (float) displayHeight );
            unsigned int idx = round( img2cloudScale * ( (float) mouse.pt.y ) ) * cloud->width + round( img2cloudScale * ( (float) mouse.pt.x ) );
            if( idx < cloud->points.size() && !isnan( cloud->points[ idx ].x ) ) {

                //  transform point to reference frame
                Eigen::Vector4d p;
                p[ 0 ] = cloud->points[ idx ].x;
                p[ 1 ] = cloud->points[ idx ].y;
                p[ 2 ] = cloud->points[ idx ].z;
                p[ 3 ] = 1;

                p = ( referenceTransform * p ).eval();

                convexHull_.push_back( p.cast< float >().block< 3, 1 >( 0, 0 ) );
            }

        }
        else if( mouse.event == CV_EVENT_RBUTTONDOWN ) {
            stopSelection = true;
        }
        else if( mouse.event == CV_EVENT_MBUTTONDOWN ) {
            stopSelection = true;
        }
    }

    if( convexHull_.size() < 3 ) {
        std::cout << "Plane requires more than 3 points\n";

    }else {

        // find normal
        Eigen::Vector3f p0 = convexHull_[ 0 ];
        Eigen::Vector3f p1 = convexHull_[ 1 ];
        Eigen::Vector3f p2 = convexHull_[ 2 ];

        Eigen::Vector3f v0 = p1 - p0;
        Eigen::Vector3f v2 = p1 - p2;

        Eigen::Vector3f normal = v0.cross(v2);

        Eigen::Vector3f Y = Eigen::Vector3f(0.f,1.f,0.f);

        if( Y.dot(normal) > 0)
            normal = -normal;

        normal.normalize();

        table_plane.coefficients[0] = normal[0];
        table_plane.coefficients[1] = normal[1];
        table_plane.coefficients[2] = normal[2];
        table_plane.coefficients[3] = -normal.dot(p1);

        table_plane.point = p1;

    }


}


inline void getLine( pcl::PointXYZRGB& p, Line& line){

    line.point = Eigen::Vector3f(0.f, 0.f, 0.f);
    line.direction = Eigen::Vector3f(p.x,p.y,p.z) - line.point;
    line.direction.normalize();
}

inline void getLinePlaneIntersection(Line& line, Plane& plane, Eigen::Vector3f& ptIntersection){

    Eigen::Vector3f Po = plane.point;
    Eigen::Vector3f Lo = line.point;
    Eigen::Vector3f n  = plane.getNormal();
    Eigen::Vector3f l = line.direction;
    float d;

    d = std::abs(( (Po - Lo).dot( n ) ) / ( l.dot( n ) ));

    // point of intersection can be found by subtituing d into the eqaution of line
    ptIntersection = d*l + Lo;

}

inline void generateTrainingImage(cv::Mat& rgbImage, cv::Mat& depthImage, Plane& table_plane, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, cv::Vec3b& table_color ){

    cv::Mat mask;
    images2PointCloud(depthImage, rgbImage, cloud, mask);

    Line ray;

    int r = table_color[0];
    int g = table_color[1];
    int b = table_color[2];

    pcl::RGB rgb;
    rgb.r = (uint8_t)r;
    rgb.g = (uint8_t)g;
    rgb.b = (uint8_t)b;
    rgb.a = 1;


    // for each pixel generate ray and find intersection with plane
    for( int x = 0; x < rgbImage.cols; x++ ){
        for( int y = 0; y < rgbImage.rows; y++ ){


            pcl::PointXYZRGB  &p = cloud->at(x,y);
            Eigen::Vector3f ptIntersection;

            // get line
            getLine( p, ray );
            getLinePlaneIntersection(ray, table_plane, ptIntersection);

            // check if point of intersection is closer to camera then the actual point
            if(ptIntersection[2] <= p.z + 0.015){

                p.x = ptIntersection[0];
                p.y = ptIntersection[1];
                p.z = ptIntersection[2];

                p.rgba = rgb.rgba;

            }

        } // end for loop for y
    } // end for loop for x

    if( 0 ){
        // show cloud
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("3D Viewer") );
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_rgb( cloud );

        viewer->setBackgroundColor( 0, 0, 0 );
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud, cloud_rgb, "changed cloud");
        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );

        while (!viewer->wasStopped ()){

            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

        }
    }

} // end function

inline void generatePlane( cv::Size2f& imgSize, Plane& table_plane, pcl::PointCloud< pcl::PointXYZRGB >::Ptr&  plane , cv::Mat& table_wood){


    plane->reserve(imgSize.width * imgSize.height);

    Line ray;
    float centerX = imgSize.width/2.f;
    float centerY = imgSize.height/2.f;
    float invfocalLength = 1.f / 525.f;

    // for each pixel generate ray and find intersection with plane
    for( int x = 0; x < imgSize.width; x++ ){
        for( int y = 0; y < imgSize.height; y++ ){

            pcl::PointXYZRGB  p ;
            p.x = ( x - centerX )  * invfocalLength;
            p.y = ( y - centerY )  * invfocalLength;
            p.z = 1;

            cv::Vec3b bgr = table_wood.at<cv::Vec3b>( y, x );
            pcl::RGB rgb;
            rgb.r = (uint8_t)bgr.val[2];
            rgb.g = (uint8_t)bgr.val[1];
            rgb.b = (uint8_t)bgr.val[0];
            rgb.a = 1;

            Eigen::Vector3f ptIntersection;

            // get line
            getLine( p, ray );
            getLinePlaneIntersection(ray, table_plane, ptIntersection);

            p.x = ptIntersection[0];
            p.y = ptIntersection[1];
            p.z = ptIntersection[2];

            p.rgba = rgb.rgba;

            plane->push_back(p);



        } // end for loop for y
    } // end for loop for x

    if( 0 ){
        // show cloud
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("plane") );
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> plane_rgb( plane );

        viewer->setBackgroundColor( 0, 0, 0 );
        viewer->addPointCloud<pcl::PointXYZRGB> (plane, plane_rgb, "changed cloud");
        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );

        while (!viewer->wasStopped ()){

            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

        }
    }
}


inline Eigen::Vector3f getTurnTableCenter( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, Plane &table_plane ) {

    // let user select convex hull points in the images
    std::cout << "select convex hull in the image\n";
    std::cout << "left click: add point, right click: finish selection\n";

    cv::Mat img_cv;
    cv::cvtColor( img_rgb, img_cv, cv::COLOR_BGR2GRAY );

    std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > turnTable;
    std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > turnTable_proj;

    cv::Mat cameraMatrix, distortionCoeffs;
    //    getCameraCalibration( cameraMatrix, distortionCoeffs ); // how to get these??
    cameraMatrix = cv::Mat::zeros( 3, 3, CV_32FC1);
    cameraMatrix.at<float>(0,0) = 525.f; //fx
    cameraMatrix.at<float>(1,1) = 525.f; //fy
    cameraMatrix.at<float>(0,2) = img_rgb.cols / 2.f; // cx
    cameraMatrix.at<float>(1,2) = img_rgb.rows / 2.f; // cy
    cameraMatrix.at<float>(2,2) = 1;

    distortionCoeffs = cv::Mat::zeros( 1, 4, CV_32FC1);

    // wait for user input
    // left click: save clicked point in reference frame
    // right/middle click: stop selection

    bool stopSelection = false;
    while( !stopSelection ) {

        // project selected points into image
        cv::Mat img_viz = img_rgb.clone();
        if( turnTable.size() > 0 ) {

            std::vector< cv::Point3f > turnTableCamera( turnTable.size() );
            std::vector< cv::Point3f > turnTableCamera_proj( turnTable.size() );
            for( unsigned int j = 0; j < turnTable.size(); j++ ) {

                // transform point from reference frame to camera frame

                turnTableCamera[ j ].x = turnTable[j](0);
                turnTableCamera[ j ].y = turnTable[j](1);
                turnTableCamera[ j ].z = turnTable[j](2);

                turnTableCamera_proj[ j ].x = turnTable_proj[ j ]( 0 );
                turnTableCamera_proj[ j ].y = turnTable_proj[ j ]( 1 );
                turnTableCamera_proj[ j ].z = turnTable_proj[ j ]( 2 );

            }

            std::vector< cv::Point2f > imagePoints( turnTableCamera.size() );
            cv::Mat rot( 3, 1, CV_64FC1, 0.f );
            cv::Mat trans( 3, 1, CV_64FC1, 0.f );
            cv::projectPoints( cv::Mat( turnTableCamera ), rot, trans, cameraMatrix, distortionCoeffs, imagePoints );

            for( unsigned int j = 0; j < imagePoints.size(); j++ ) {

                if( imagePoints[ j ].x < 0 || imagePoints[ j ].x > img_cv.cols - 1 )
                    continue;

                if( imagePoints[ j ].y < 0 || imagePoints[ j ].y > img_cv.rows - 1 )
                    continue;

                cv::Scalar c( 0.f, 0.f, 255.f );
                cv::circle( img_viz, imagePoints[ j ], 10, c, -1 );

            }

            std::vector< cv::Point2f > imagePoints_proj( turnTableCamera_proj.size() );
            //            cv::Mat rot( 3, 1, CV_64FC1, 0.f );
            //            cv::Mat trans( 3, 1, CV_64FC1, 0.f );
            cv::projectPoints( cv::Mat( turnTableCamera_proj ), rot, trans, cameraMatrix, distortionCoeffs, imagePoints_proj );

            for( unsigned int j = 0; j < imagePoints_proj.size(); j++ ) {

                if( imagePoints_proj[ j ].x < 0 || imagePoints_proj[ j ].x > img_cv.cols - 1 )
                    continue;

                if( imagePoints_proj[ j ].y < 0 || imagePoints_proj[ j ].y > img_cv.rows - 1 )
                    continue;

                cv::Scalar c( 100.f, 255.f, 0.f );
                cv::circle( img_viz, imagePoints_proj[ j ], 10, c, -1 );

            }


        }

        int displayHeight = 240;
        double imgScaleFactor = ( (float) displayHeight ) / ( (float) img_viz.rows );
        if( img_viz.rows != displayHeight ) {
            cv::Mat tmp;
            cv::resize( img_viz, tmp, cv::Size(), imgScaleFactor, imgScaleFactor, cv::INTER_LINEAR );
            tmp.copyTo( img_viz );
        }
        cv::imshow( "Select center of turn table ", img_viz );

        MouseEvent mouse;
        cv::setMouseCallback( "Select center of turn table ", onMouse, &mouse );
        cv::waitKey( 10 );

        if( mouse.event == CV_EVENT_LBUTTONDOWN ) {

            // find corresponding 3D position in point cloud
            float img2cloudScale = ( (float) cloud->height ) / ( (float) displayHeight );
            unsigned int idx = round( img2cloudScale * ( (float) mouse.pt.y ) ) * cloud->width + round( img2cloudScale * ( (float) mouse.pt.x ) );
            if( idx < cloud->points.size() && !isnan( cloud->points[ idx ].x ) ) {

                //  transform point to reference frame

                Eigen::Vector3f p, p_proj;
                p[ 0 ] = cloud->points[ idx ].x;
                p[ 1 ] = cloud->points[ idx ].y;
                p[ 2 ] = cloud->points[ idx ].z;

                //                p = ( referenceTransform * p ).eval();

                // project on plane
                Eigen::Vector3f normal = table_plane.coefficients.block< 3, 1 >( 0, 0 );
                p_proj = p - normal.dot( p - table_plane.point ) * normal;

                turnTable_proj.push_back(p_proj);
                turnTable.push_back( p.cast< float >().block< 3, 1 >( 0, 0 ) );
            }

        }
        else if( mouse.event == CV_EVENT_RBUTTONDOWN ) {
            stopSelection = true;
        }
        else if( mouse.event == CV_EVENT_MBUTTONDOWN ) {
            stopSelection = true;
        }
    }

    Eigen::Vector3f center;

    if( turnTable.size() < 3 ) {
        std::cout << "circle requires more than 3 points\n";
        return center;
    }
    else {

        // calcualte center of peri circle
        Eigen::Vector3f normal = table_plane.coefficients.block< 3, 1 >( 0, 0 );

        // mid points
        Eigen::Vector3f m1,m2;
        m1 = ( turnTable_proj[0] + turnTable_proj[1] ) / 2.f ;
        m2 = ( turnTable_proj[1] + turnTable_proj[2] ) / 2.f;

        // bisectors
        Eigen::Vector3f b1, b2;
        b1 = normal.cross( turnTable_proj[0] - turnTable_proj[1] );
        b2 = normal.cross( turnTable_proj[1] - turnTable_proj[2] );

        b1.normalize();
        b2.normalize();

        Eigen::Vector2f v = (m1-m2).block<2,1>(0,0);
        Eigen::Matrix2f A;
        A.block<2,1>(0,0) = b2.block<2,1>(0,0);
        A.block<2,1>(0,1) = -b1.block<2,1>(0,0);

        Eigen::Vector2f lambda;
        lambda = A.inverse()*v;

        center = m2 + lambda[1]*b2;

        table_plane.point = center;

    }

    return center;

}


inline void drawTransformation(cv::Mat &img, cv::Mat &depthImg , Eigen::Matrix4f &m_transformationMatrixOC){

    Eigen::Affine3f transformationMatrixOC;
    transformationMatrixOC.matrix() = m_transformationMatrixOC;

    // Initialize the cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudTC(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbTC(cloudTC);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudTO(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbTO(cloudTO);

    // Populate the cloud
    cv::Mat mask;
    images2PointCloud( depthImg, img, cloud, mask);

    /* Transform point cloud */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    pcl::transformPointCloud(*cloud, *cloudTO, transformationMatrixOC.inverse());
    pcl::transformPointCloud(*cloudTO, *cloudTC, transformationMatrixOC);

    //        cout << "\n"<< transformationMatrixOC.matrix() << endl;

    //visualize coordinates in object frame
    pcl::Normal OX(transformationMatrixOC.matrix()(0,0),transformationMatrixOC.matrix()(1,0),transformationMatrixOC.matrix()(2,0));
    pcl::Normal OY(transformationMatrixOC.matrix()(0,1),transformationMatrixOC.matrix()(1,1),transformationMatrixOC.matrix()(2,1));
    pcl::Normal OZ(transformationMatrixOC.matrix()(0,2),transformationMatrixOC.matrix()(1,2),transformationMatrixOC.matrix()(2,2));
    pcl::Normal Origin(transformationMatrixOC.matrix()(0,3),transformationMatrixOC.matrix()(1,3),transformationMatrixOC.matrix()(2,3));

    pcl::PointXYZ O_(Origin.normal_x, Origin.normal_y, Origin.normal_z);
    pcl::PointXYZ X_(0.1 * OX.normal_x+Origin.normal_x, 0.1 * OX.normal_y+Origin.normal_y, 0.1 * OX.normal_z+Origin.normal_z);
    pcl::PointXYZ Y_(0.1 * OY.normal_x+Origin.normal_x, 0.1 * OY.normal_y+Origin.normal_y, 0.1 * OY.normal_z+Origin.normal_z);
    pcl::PointXYZ Z_(0.1 * OZ.normal_x+Origin.normal_x, 0.1 * OZ.normal_y+Origin.normal_y, 0.1 * OZ.normal_z+Origin.normal_z);

    viewer->addLine(O_, X_, 255, 0, 0, "lineX");
    viewer->addLine(O_, Y_, 0, 255, 0, "lineY");
    viewer->addLine(O_, Z_, 0, 0, 255, "lineZ");
    viewer->addCoordinateSystem	(0.1f,0.f,0.f,0.f, 0 );

    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "Original cloud");
    //        viewer->addPointCloud<pcl::PointXYZRGB> (cloudTO, rgbTO, "object frame cloud");
    viewer->addPointCloud<pcl::PointXYZRGB> (cloudTC, rgbTC, "camera frame cloud");

    while (!viewer->wasStopped ()){
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}


inline void calcObject2CameraTransformation( float &pose, float &pitch, cv::Point3f &rObjCenter, Eigen::Matrix4f &transformationMatrixOC){

    Eigen::Affine3f Rx,T, Rz;

    pcl::getTransformation(rObjCenter.x , rObjCenter.y, rObjCenter.z, 0.f, 0.f, 0.f, T);
    pcl::getTransformation(0.f,0.f,0.f, pitch, 0.f, 0.f, Rx);
    pcl::getTransformation(0.f, 0.f, 0.f, 0.f, 0.f, pcl::deg2rad(-pose), Rz);


    transformationMatrixOC = T.matrix() * Rx.matrix() * Rz.matrix();
}


inline void changeGamma(cv::Mat& img, cv::Mat& changed_img , cv::Mat& intensity_gradient_img){

    img.copyTo(changed_img);
    cv::Mat color, color_, colorInv_, rand_gradient1, rand_gradient2;

    if(intensity_gradient_img.channels() == 1)
        cv::cvtColor(intensity_gradient_img,  color, CV_GRAY2BGR);
    else
        intensity_gradient_img.copyTo(color);

    // create random strngth of gradient
    cv::RNG rng(cv::getTickCount());
    int rand_num = rng.operator ()(50) + 200; // a num between [200 250)
    double minVal, maxVal;
    cv::minMaxIdx(color, &minVal, &maxVal);
    cv::convertScaleAbs(color, color_, rand_num/maxVal, -minVal); // scale it to [0,rand_num
    cv::Mat constVal = cv::Mat(color_.rows, color_.cols, CV_8UC3, cv::Scalar(rand_num, rand_num, rand_num) );
    cv::Mat constValby2 = cv::Mat(color_.rows, color_.cols, CV_8UC3, cv::Scalar(rand_num/2, rand_num/2, rand_num/2) );
    cv::subtract(constVal, color_, colorInv_ );

    // insert gradient into image
    img.copyTo(changed_img);
    cv::subtract(color_, constValby2, rand_gradient1);
    cv::add(rand_gradient1, img, changed_img);

    cv::subtract(colorInv_, constValby2, rand_gradient2);
    cv::subtract( img, rand_gradient2, changed_img);

    //    cv::imshow("original gradient", color);
    //    cv::imshow("scaled gradient", color_);
    //    cv::imshow("inv scale gradient", colorInv_);
    //    cv::imshow("gradient1", rand_gradient1);
    //    cv::imshow("gradient2", rand_gradient2);
    //    cv::imshow("img", img);
    //    cv::imshow("changed", changed_img);
    //    cv::waitKey(0);


}


inline void moveOverPlane(pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4f& transformationOC, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud_moved, cv::Point3f& position, float& pose, Eigen::Vector3f& table_point){

    // downsampling using voxel grid
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.001f, 0.001f, 0.001f); // 5mm x 5mm x 5mm voxels
    sor.filter (*cloud_filtered);

    // remove noise
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_cluster_rgb(cloud_cluster);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
    sor1.setInputCloud (cloud_filtered);
    sor1.setMeanK (50);
    sor1.setStddevMulThresh (1.0);
    sor1.filter (*cloud_cluster);

    Eigen::Matrix4f inverseTransform = transformationOC.inverse();
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr temp (new  pcl::PointCloud< pcl::PointXYZRGB >);
    pcl::transformPointCloud(*cloud_cluster, *temp, inverseTransform);

    // Create the filtering object
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_filtered1 (new  pcl::PointCloud< pcl::PointXYZRGB >);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (temp);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 2.0);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_filtered1);

    // find bounding box surrounded
    pcl::PointXYZRGB  min_pt, max_pt;
    pcl::getMinMax3D (*cloud_filtered1, min_pt, max_pt);

    pcl::PointCloud< pcl::PointXYZRGB >::Ptr temp1 (new  pcl::PointCloud< pcl::PointXYZRGB >);
    pcl::transformPointCloud(*cloud, *temp1, inverseTransform);

    //    Eigen::Vector4f transformed_table_plane_point = transformationOC.inverse() * Eigen::Vector4f(table_point[0], table_point[1], table_point[2], 1);

    Eigen::Affine3f Rz;
    pcl::getTransformation(0.f, 0.f, 0.f, 0.f, 0.f, pcl::deg2rad(pose), Rz);

    Eigen::Vector4f vpose( position.x, position.y, position.z, 1 );
    Eigen::Vector4f vpose_ = Rz*vpose;

    float x = vpose_[0]; //sign*(rng.operator ()( 100 ) / 1000.f + 0.1f);
    float y = vpose_[1]; //sign*(rng.operator ()( 100 ) / 1000.f + 0.1f);

    //    float z = transformed_table_plane_point[2];
    float z = -min_pt.z;
//    cout<< z<< endl;

    // translate object on xy plane
    Eigen::Vector3f translation = Eigen::Vector3f(x, y, z);
    Eigen::Quaternionf rot = Eigen::Quaternionf::Identity();
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr temp2 (new  pcl::PointCloud< pcl::PointXYZRGB >);
    pcl::transformPointCloud( *temp1, *temp2, translation, rot );

    pcl::transformPointCloud(*temp2, *cloud_moved, transformationOC );


    if( 0 ){
        // show cloud
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("translation over plane") );
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_rgb( cloud );
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_moved_rgb( cloud_moved );

        viewer->setBackgroundColor( 1, 1, 1 );
        //        viewer->addPointCloud<pcl::PointXYZRGB> (cloud, cloud_rgb, "changed1 cloud");
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud_moved, cloud_moved_rgb, "changed 2cloud");
        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );

        while (!viewer->wasStopped ()){

            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

        }
    }
}

inline void getPointOnLineWithMinDepth(const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Line& l, float thresh, pcl::PointXYZRGB& pt){

    std::vector<pcl::PointXYZRGB> linePoints;
    linePoints.reserve(10);
    for( int p = 0; p < cloud->size(); p++ ){ // for each point in cloud

        pcl::PointXYZRGB cloud_pt = cloud->at(p);

        if(cloud_pt.x == 0.f && cloud_pt.y == 0.f &&cloud_pt.z == 0.f  )
            continue;

        Eigen::Vector3f dist = l.point - cloud_pt.getVector3fMap();

        float d = ( dist - ( dist.dot( l.direction ) ) * l.direction ).norm();

        if( d >= 0.f && d < thresh )
            linePoints.push_back(cloud_pt);
    }

    // find point with least depth
    float depth = FLT_MAX;
    int index = -1;
    for(int l = 0;  l < linePoints.size(); l++){

        if(linePoints[l].z < depth){

            depth = linePoints[l].z;
            index = l;
        }

    }

    pcl::PointXYZRGB minDepthPt = linePoints[index];

    // project point on line
    Eigen::Vector3f projection = (l.direction.dot(minDepthPt.getVector3fMap())) * l.direction;

    pt.x = projection[0];
    pt.y = projection[1];
    pt.z = projection[2];
    pt.rgba = minDepthPt.rgba;

}


inline void pointCloud2Images_byPointInflation( pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, cv::Mat& depthImg, cv::Mat& colorImg, cv::Size2f &imgSize ) {

    cv::Point2f center( imgSize.width/2.f, imgSize.height/2.f );
    colorImg = cv::Mat::zeros( imgSize.height, imgSize.width, CV_8UC3 );
    depthImg = cv::Mat( imgSize.height, imgSize.width, CV_32FC1, -1 );

    for( int p = 0; p< cloud->size(); p++ ){

        pcl::PointXYZRGB &samplePt = cloud->at( p );
        if(samplePt.x == 0.f && samplePt.y == 0.f &&samplePt.z == 0.f || samplePt.z < 0 )
            continue;
        Line l;
        getLine(samplePt, l);
        pcl::PointXYZRGB cloudPt;
        getPointOnLineWithMinDepth(cloud, l, 0.01, cloudPt);

        cv::Point3f pt( cloudPt.x, cloudPt.y, cloudPt.z );
        cv::Vec3b bgr( cloudPt.b, cloudPt.g, cloudPt.r );

        cv::Point2f pixel;
        float depth;// in meters
        R3toP3( pt, center, pixel, depth );

        // check if pixel is in the image
        if(int(pixel.x) < 0 || int(pixel.x) > imgSize.width ||int(pixel.y) < 0 || int(pixel.y) > imgSize.height )
            continue;

        // check if the pixel is already filled before

        if( depthImg.at< float >( pixel ) < 0 ){ // then fill the color and depth image

            colorImg.at< cv::Vec3b >( pixel ) = bgr;
            depthImg.at< float >( pixel ) = depth;

        }else{

            if(depthImg.at< float >( pixel ) > depth){

                depthImg.at< float >( pixel ) = depth;
                colorImg.at< cv::Vec3b >( pixel ) = bgr;
            }
        }

    }
}


void fillHoles(const cv::Mat& img, const cv::Mat& depthImg, cv::Mat& filledImg, cv::Mat& filledDepthImg, float radius){

    cv::Mat copy_img = cv::Mat(img.rows, img.cols, img.type());
    cv::Mat copy_depthImg = cv::Mat(depthImg.rows, depthImg.cols, depthImg.type());
    img.copyTo(copy_img);
    depthImg.copyTo(copy_depthImg);

    for(int it = 0; it < 3; it++){


        ////////////////////////////////////////////////  find mask ///////////////////////////////////////////
        // convert color image to gray
        cv::Mat gray;
        cv::cvtColor( copy_img, gray, CV_BGR2GRAY );

        //threshold the image
        cv::Mat binary = cv::Mat(copy_img.rows, copy_img.cols, CV_8U, 0);
        cv::threshold( gray, binary, 0, 255, CV_THRESH_BINARY );

        // Apply closing operation
        cv::Mat closed;
        cv::Mat element = cv::Mat(radius, radius, CV_8U, 255);
        cv::morphologyEx( binary, closed, CV_MOP_CLOSE, element );

        // threhold the closed image
        cv::Mat closedThreshold;
        cv::threshold( closed, closedThreshold, 0, 255, CV_THRESH_BINARY );

        // subtract the binary image from closed
        cv::Mat sub;
        cv::subtract( closedThreshold, binary, sub );

        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Apply closing operation on original image
        cv::Mat closed_original, closed_depth_original;
        //        copy_img.copyTo(closed_original);
        //        copy_depthImg.copyTo(closed_depth_original);

        cv::Mat element_ = cv::Mat(radius, radius, CV_8U, 255);
        cv::morphologyEx( copy_img, closed_original, CV_MOP_CLOSE, element_ );
        cv::morphologyEx( copy_depthImg, closed_depth_original, CV_MOP_CLOSE, element_ );

        // only add the two image where there is a hole
        copy_img.copyTo(filledImg);
        copy_depthImg.copyTo(filledDepthImg);
        cv::add( copy_img, closed_original, filledImg, sub, img.depth() );
        cv::add( copy_depthImg, closed_depth_original, filledDepthImg, sub, depthImg.depth()  );

        // this is a hack
        closed_depth_original.copyTo(filledDepthImg);

        //        cv::imshow( "img", copy_img );
        //        cv::imshow( "closed depth img", closed_depth_original );
        ////        cv::imshow("gray", gray);
        //        cv::imshow("subtraction", sub);
        //        cv::imshow( "filledimg", filledImg );
        //        cv::imshow("depthImg", depthImg);
        //        cv::imshow(" depth filled image", filledDepthImg);
        //        cv::waitKey(0);

        filledImg.copyTo(copy_img);
        filledDepthImg.copyTo(copy_depthImg);

    }

}

void CombineImages(std::vector< cv::Mat >& vRgbImg, std::vector< cv::Mat>& vDepthImg, cv::Mat& rgbImg, cv::Mat& depthImg ){

    rgbImg = cv::Mat(vRgbImg[0].rows, vRgbImg[0].cols, CV_8UC3);
    depthImg = cv::Mat(vRgbImg[0].rows, vRgbImg[0].cols, CV_16UC1);

    for( unsigned int r = 0; r < vRgbImg[0].rows; r++  ){
        for(unsigned int c = 0; c< vRgbImg[0].cols; c++){

            // find minimum depth
            float depth = FLT_MAX;
            cv::Vec3b bgr(0,0,0);
            for(int s = 0; s < vRgbImg.size(); s++  ){

                if( vDepthImg[s].at<float>(r,c) > 0 && vDepthImg[s].at<float>(r,c) < depth){

                    depth = vDepthImg[s].at<float>(r,c);
                    bgr =  vRgbImg[s].at< cv::Vec3b >( r,c );
                }

            }

            rgbImg.at<cv::Vec3b>(r,c) = bgr;
            depthImg.at< unsigned short>(r,c) = int(depth*1000); // convert it to milimeter

        }
    }

}

#endif // OBJECTGEOMETRY_H

