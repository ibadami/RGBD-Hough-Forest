
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;
string objName;
string testClassFiles = "./test_all.txt";
string testClassPath = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/test";
string poseGroundTruth = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/groundTruth/pose/";
string locationGroundTruth = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/groundTruth/location/";

bool DEBUG = 0;


struct MouseEvent {

    MouseEvent() {
        event = -1;
        buttonState = 0;
    }
    cv::Point pt;
    int event;
    int buttonState;

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
        else if( mouse.event == CV_EVENT_RBUTTONDOWN && convexHull_.size() >=3) {
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


// load testing image filenames
void loadTestClassFile(string& testimagefiles, std::vector<std::vector<string> >& vFilenames ) {

    ifstream in_class(testimagefiles.c_str());
    if(in_class.is_open()) {
        int n_test_classes;
        in_class >> n_test_classes;
        vFilenames.resize(n_test_classes);
        //test_classes.resize(n_test_classes);

        cout << "number Classes: " << vFilenames.size() << endl;
        string labelfile;
        for(int l=0; l < n_test_classes;++l) {
            in_class >> labelfile;
            ifstream in(labelfile.c_str());
            if(in.is_open()) {
                unsigned int size;
                in >> size;
                cout << "Load Test Examples: " << l << " - " << size << endl;
                vFilenames[l].resize(size);
                for(unsigned int i=0; i<size; ++i) {
                    // Read filename
                    in >> vFilenames[l][i];
                }
            } else {
                cerr << "File not found " << labelfile.c_str() << endl;
                exit(-1);
            }
            in.close();
        }
    } else {
        cerr << "File not found " << testimagefiles.c_str() << endl;
        exit(-1);
    }
    in_class.close();
}


void readGroundTruth( std::string& filename, std::vector< cv::Rect >& groundTruth ){

    // read groundtruth files
    unsigned int size;
    ifstream in( filename.c_str() );
    if( in.is_open() ){

        in >> size;
        groundTruth.resize(size);

        for( unsigned int i = 0; i < size; i++ ){

            in >> groundTruth[ i ].x; in >> groundTruth[ i ].y;
            in >> groundTruth[ i ].width; in >> groundTruth[ i ].height;

        }
    }

    in.close();
}


cv::Point3f P3toR3(cv::Point2f &pixelCoordinates, cv::Point2f &center, float depth){

    float focal_length = 525.f; // in pixel
    cv::Point3f realCoordinates;
    realCoordinates.z = depth; // in meter
    realCoordinates.x = (pixelCoordinates.x - center.x)* depth / focal_length;
    realCoordinates.y = (pixelCoordinates.y - center.y)* depth / focal_length;
    return realCoordinates;
}



int main(int argc, char* argv[]) {

    if( argc < 2 ){
        cout<< " Please provide objname " << endl;

    }else{

        objName = argv[1];

        // read test images
        vector< vector< string > > vFilenames;
        loadTestClassFile( testClassFiles, vFilenames);

        for(unsigned int i = 2;  i <  vFilenames.size();  i++ ){

            // create test set folder
            string foldername = vFilenames[i][0];
            foldername.erase(foldername.find_first_of("/"));
            string exec = "mkdir " + poseGroundTruth + foldername;
            system( (exec).c_str());
            for (unsigned int j = 0; j < vFilenames[i].size(); j++ ) {

                // Load images
                cv::Mat rgbImg = cv::imread( (testClassPath + "/" + vFilenames[i][j]).c_str(), CV_LOAD_IMAGE_COLOR );

                string filename = vFilenames[ i ][ j ];
                int size_of_string = vFilenames[ i ][ j].size();
                filename.replace( size_of_string - 4, 15, "_filleddepth.png" );
                cv::Mat depthImg = cv::imread( ( testClassPath + "/" + filename ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );

                std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > planePoints ;
                Eigen::Matrix4d referenceTransform = Eigen::Matrix4d::Identity();
                pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud ( new pcl::PointCloud< pcl::PointXYZRGB > );
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_rgb( cloud);

                cv::Mat mask;
                images2PointCloud( depthImg, rgbImg, cloud, mask );

                // get plane
                Plane table_plane;
                selectPlane( rgbImg, cloud, referenceTransform, planePoints, table_plane) ;
                Eigen::Vector3f normal = table_plane.getNormal();

                // check if object is present
                filename = vFilenames[ i ][ j ];
                filename.replace( size_of_string - 4, 4, ".txt" );
                string groundTruthFilename  = locationGroundTruth + objName + "/" + filename;
                // read ground truth
                std::vector< cv::Rect > groundTruths(0);
                readGroundTruth( groundTruthFilename, groundTruths );

                string poseFile = poseGroundTruth + "/" + filename;

                // save normal in the .txt file
                ofstream out(poseFile.c_str());
                if(out.is_open()){
//                    if(groundTruths.size() > 0){
                        cout << normal[0] << " " << normal[1] << " " << normal[2] << endl;
                        out << normal[0] << " " << normal[1] << " " << normal[2] << endl;
                        cout  <<  poseFile << endl;
//                    }
//                    else{
//                        cout  <<  vFilenames[ i ][ j ]<< endl;
//                        out << 0;
//                    }

                }

                if(DEBUG){

                    if(groundTruths.size() > 0){

                        cv::Point2f imgcenter(rgbImg.cols/2.f, rgbImg.rows/2.f);

                        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_debug ( new pcl::visualization::PCLVisualizer ("debug pose ground truth") );
                        viewer_debug->setBackgroundColor( 1, 1, 1 );
                        viewer_debug->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );
                        viewer_debug->addPointCloud<pcl::PointXYZRGB> (cloud,cloud_rgb, "cloud");

                        std::vector< pcl::PointXYZ > objCenter(groundTruths.size());
                        std::vector< pcl::PointXYZ >  anotherPoint(groundTruths.size());


                        for(unsigned int n = 0;  n < groundTruths.size(); n++ ){

                            cv::Point2f center2D(groundTruths[n].x + groundTruths[n].width/2.f,  groundTruths[n].y + groundTruths[n].height/2.f );
                            float depth = depthImg.at< unsigned short> (center2D)/1000.f;
                            cv::Point3f center3D  = P3toR3(center2D, imgcenter, depth);

                            objCenter[n] = pcl::PointXYZ(center3D.x, center3D.y, center3D.z);


                            anotherPoint[n].x = center3D.x + normal[0]/10.f;
                            anotherPoint[n].y = center3D.y + normal[1]/10.f;
                            anotherPoint[n].z = center3D.z + normal[2]/10.f;

                            stringstream ss; ss << n;
                            viewer_debug->addLine( objCenter[n], anotherPoint[n], 255, 0, 0, ss.str() );

                        }

                        while (!viewer_debug->wasStopped ()){

                            viewer_debug->spinOnce (100);
                            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

                        }
                        //  TODO: project it in the image
                    }
                }
            }
        }
    }

    return 0;

}

