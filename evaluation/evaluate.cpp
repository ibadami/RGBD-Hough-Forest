/*
// Author: Ishrat Badami University of Bonn
// Email: badami@informatik.uni-bonn.des
//
*/

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PATH_SEP "/"

using namespace std;

// set some gloabal parameters keep changing for different configurations

string testimagepath = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/test";

string testimagefiles = "test_all.txt";

string boundingboxFolder = "boundingBox_original";

string groundTruthFolder = "../multiclassOD/rgbd-dataset/groundTruth/location";

string groundTPoseFolder = "../multiclassOD/rgbd-dataset/groundTruth/pose";

string suffix = "";

string candidatepath = "../multiclassOD/rgbd-dataset/output/detection";

bool pose_present = 0;

string objname;

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

void readGroundPose( std::string& filename, Eigen::Vector3f& groundPoses ){

    // read groundpose files
    ifstream in( filename.c_str() );
    if( in.is_open() ){

        for( unsigned int i = 0; i < 3; i++ ){

            in >> groundPoses[i];

        }
    }

    in.close();
}


void readBoundingBox( std::string& filenameBB, std::string& filenameCand, cv::Mat& img, std::vector< cv::Rect >& boundingBoxes, std::vector< Eigen::Vector3f >& pose_z, std::vector < float >& scores, bool pose_present ){

    //read and convert the 3D boundingbox to 2D rectangle which escribed all the 8 points
    unsigned int size;
    ifstream in( filenameBB.c_str() );

    if( in.is_open() ){

        in >> size;
        boundingBoxes.resize( size );
        pose_z.resize(size);
        for( unsigned int i = 0; i < size; i++ ){ // for each 3D bounding box
            vector< cv::Point2f > vertices( 8 );
            cv::Point2f temp;
            for( unsigned int j = 0; j < 8; j++ ){

                in >> temp.x; vertices[ j ].x = max( min(int(temp.x), img.cols) , 0);
                in >> temp.y; vertices[ j ].y = max( min(int(temp.y), img.rows) , 0);
            }

            boundingBoxes[i] = ( cv::boundingRect( vertices ) );

            if(pose_present){

                for( unsigned int j = 0; j < 3; j++ )

                    in >> pose_z[i][j];

            }
        }

    }
    in.close();

    unsigned int size_cand = 0;
    float useless = 0.f;
    ifstream in_cand( filenameCand.c_str() );
    if( in_cand.is_open() ){

        in_cand >> size_cand;
        scores.resize( size_cand );
        for( unsigned int i = 0; i < size; i++ ){ // for each 3D bounding box
            in_cand >> scores[ i ];

            for( unsigned int j = 0; j < 5; j++ )
                in_cand >> useless;
        }

    }
    in_cand.close();

}



void findOverlap( cv::Rect& groundTruth, cv::Rect& candidate, float& strength ){


    float Xgt1 = groundTruth.x; float Xgt2 = groundTruth.x + groundTruth.width;
    float Ygt1 = groundTruth.y; float Ygt2 = groundTruth.y + groundTruth.height;

    float Xcd1 = candidate.x; float Xcd2 = candidate.x + candidate.width;
    float Ycd1 = candidate.y; float Ycd2 = candidate.y + candidate.height;


    if ( Xgt1 < Xcd2 && Xgt2 > Xcd1 && Ygt1 < Ycd2 && Ygt2 > Ycd1){ // if there is any overlap

        float AI = max( 0, int( min( Xcd2, Xgt2 ) - max( Xcd1, Xgt1 ) ) ) * max( 0, int( min( Ycd2, Ygt2 ) - max( Ycd1, Ygt1 ) ) );

        if( AI > 0 ){

            float Agt = groundTruth.area();
            float Acd = candidate.area();
            //            float maxArea = max(Agt, Acd);
            float minArea = min(Agt, Acd);

            //            if( minArea / maxArea < 0.4f )
            //                matched = 0;

            if( AI / minArea > 0.5f )
                strength = AI/minArea;


        }

    }

}


void readtestImages( string &testimagefiles, vector< vector< string> >& vFilenames){

    ifstream in_class(testimagefiles.c_str());
    if(in_class.is_open()) {

        int n_test_classes;
        in_class >> n_test_classes;
        vFilenames.resize(n_test_classes);

        cout << "number Classes: " << vFilenames.size() << endl;
        string labelfile;

        for( int l = 0; l < n_test_classes; ++l ) {

            in_class >> labelfile;
            ifstream in( labelfile.c_str() );
            if( in.is_open() ){

                unsigned int size;
                in >> size;
                cout << "Load Test Examples: " << l << " - " << size << endl;
                vFilenames[ l ].resize(size);

                for( unsigned int i = 0; i < size; ++i )
                    in >> vFilenames[ l ][ i ]; // Read filename

            } else {
                cerr << "File not found " << labelfile.c_str() << endl;
                exit( -1 );
            }

            in.close();
        }

    }else {

        cerr << "File not found " << testimagefiles.c_str() << endl;
        exit( -1 );

    }

    in_class.close();

}

void calcPRA(vector< vector< float> >& tpfptnfn, vector<vector< float> >& PRA){

    PRA.resize(tpfptnfn.size());

    for( unsigned int i = 0; i < tpfptnfn.size(); i++ ){

        PRA[i].resize(3);
        PRA[i][0] = tpfptnfn[i][0]/ (tpfptnfn[i][0] + tpfptnfn[i][1]); // precision
        PRA[i][1] = tpfptnfn[i][0]/ (tpfptnfn[i][0] + tpfptnfn[i][3]); // recall
        // accuracy
        PRA[i][2] = ( tpfptnfn[i][0] + tpfptnfn[i][2] ) / (tpfptnfn[i][0] + tpfptnfn[i][1]+ tpfptnfn[i][2]+ tpfptnfn[i][3]);

    }

}

void calcPRA_all(vector< float>& tp, vector< float>& fp, vector< float>& fn, vector<vector< float> >& PRA){

    PRA.resize(tp.size());

    for( unsigned int i = 0; i < tp.size(); i++ ){

        PRA[i].resize(3);
        PRA[i][0] = tp[i]/ (tp[i] + fp[i]); // precision
        PRA[i][1] = tp[i]/ (tp[i] + fn[i]); // recall

        PRA[i][2] = ( tp[i] ) / (tp[i] + fp[i]+ fn[i] ); // accuracy

    }

}

cv::Point3f P3toR3(cv::Point2f &pixelCoordinates, cv::Point2f &center, float depth){

    float focal_length = 525.f; // in pixel
    cv::Point3f realCoordinates;
    realCoordinates.z = depth; // in meter
    realCoordinates.x = (pixelCoordinates.x - center.x)* depth / focal_length;
    realCoordinates.y = (pixelCoordinates.y - center.y)* depth / focal_length;
    return realCoordinates;
}

void imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud) {

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

    //   const float* depthdata = reinterpret_cast<const float*>(&depthImg.data[0]);
    //   const unsigned char* colordata = &colorImg.data[0];

    cv::Point pt;
    int idx = 0;
    for ( int y = 0; y < colorImg.rows; y++ ) {
        for ( int x = 0; x < colorImg.cols; x++ ) {

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

            //    p.r = ( unsigned int )colorImg.at<cv::Vec3b>( pt )[ 0 ];
            //    p.g = ( unsigned int )colorImg.at<cv::Vec3b>( pt )[ 1 ];
            //    p.b = ( unsigned int )colorImg.at<cv::Vec3b>( pt )[ 2 ];

            //      int rgb = (r << 16) + (g << 8) + b;
            //      p.rgb = *(reinterpret_cast<float*>(&rgb));


            idx++;

        }
    }

}

int main( int argc, char* argv[ ] ){

    if( argc < 2 ){

        cout << "provide name of the object you want to do evaluation for" << endl;
        return(-1);

    }else{

        objname = argv[ 1 ];

        // start evaluation for objname
        cout<< "starting evaluation for "<< objname <<endl;

        // read test images
        vector< vector< string > > vFilenames;
        readtestImages( testimagefiles, vFilenames );

        // save statistics in a vector of vector
        vector< vector< float > > tftf;

        float minThresh = 0.f, maxThresh = 3.f, step = 0.1f;
        int size_of_vector = int((maxThresh-minThresh)/step);

        vector<float>falsePositive_all(size_of_vector, 0);
        vector<float>falseNegative_all(size_of_vector, 0);
        vector<float>truePositive_all(size_of_vector, 0);
        vector<float>trueNegative_all(size_of_vector, 0);

        string filename_all_dataset = objname + "_" + suffix + "_all_dataset.txt";
        string filename_individual_dataset = objname + "_" + suffix + "_individual_dataset.txt";
        string filename_orientations = objname + "_" + suffix + "_orientations.txt";

        ofstream out_individual(filename_individual_dataset.c_str());
        ofstream out_orientations(filename_orientations.c_str());

        if(out_individual.is_open() && out_orientations.is_open()){

            out_individual << minThresh << " " << maxThresh << " " << step << endl;
            out_individual << endl;

            // for each test dataset check for detection
            for( unsigned int l = 0; l < vFilenames.size(); l++ ){
                tftf.clear();
                tftf.reserve(500);

//                out << l << endl;

                for( float threshold =  minThresh; threshold < maxThresh; threshold += step ){

                    float falsePositive = 0, falseNegative = 0, truePositive = 0, trueNegative = 0;


                    for( unsigned int i = 0; i < vFilenames[ l ].size(); i++ ){

                        cout<< "threshold.... " << threshold <<"  set " << l << " .... image: " << i << " out of " << vFilenames[ l ].size() << endl;

                        string rawname = vFilenames[ l ][ i ];
                        rawname.erase( rawname.find_first_of( "." ), rawname.length() );

                        string filename_img = testimagepath + PATH_SEP + vFilenames[ l ][ i ];

                        // read img and depth image for debugging
                        cv::Mat img = cv::imread( filename_img.c_str(), CV_LOAD_IMAGE_COLOR );


                        cout << "loaded image file: " << ( testimagepath + PATH_SEP + vFilenames[ l ][ i ] ).c_str() << endl;

                        // Load Depth Image
                        string filename_depthImg = filename_img;
                        int size_of_string = filename_depthImg .size();
                        filename_depthImg.replace( size_of_string - 4, 15, "_filleddepth.png" );
                        cv::Mat depthImg = cv::imread( ( filename_depthImg ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );
                        if( depthImg.empty() ) {

                            cout << "Could not load image file: " << ( filename_depthImg ).c_str() << endl;
                            exit( -1 );
                        }

                        string filename_groundtruth = groundTruthFolder+ PATH_SEP + objname + PATH_SEP + rawname + ".txt";
                        string filename_boundingbox = candidatepath + PATH_SEP + objname + suffix + PATH_SEP + vFilenames[ l ][ i ] + PATH_SEP + "boundingboxes.txt";
                        string filename_candidate = candidatepath + PATH_SEP + objname + suffix + PATH_SEP + vFilenames[ l ][ i ] + PATH_SEP + "candidates.txt";
                        string filename_groundpose = groundTPoseFolder+ PATH_SEP + objname + PATH_SEP + rawname + ".txt";

                        // read ground truth location and pose
                        std::vector< cv::Rect > groundTruths;
                        readGroundTruth( filename_groundtruth, groundTruths );

                        Eigen::Vector3f  groundPose;
                        if(pose_present)
                            readGroundPose( filename_groundpose, groundPose );


                        // draw all ground truths

                        for( unsigned int gt = 0; gt < groundTruths.size(); gt++ )
                            cv::rectangle( img,groundTruths[ gt ], CV_RGB( 0, 0, 255 ), 3 );

                        // read candidates bounding box + poses
                        std::vector< cv::Rect > boundingboxes;
                        std::vector< Eigen::Vector3f > pose_z;
                        std::vector< float > scores;
                        readBoundingBox( filename_boundingbox, filename_candidate, img, boundingboxes, pose_z, scores, pose_present );

                        if( boundingboxes.size() == 0 && groundTruths.size() > 0 ){
                            falseNegative += groundTruths.size();

                        }else if( boundingboxes.size() > 0 && groundTruths.size() == 0 ){

                            for (unsigned int b = 0; b < boundingboxes.size(); b++)

                                if(scores[ b ] > threshold){
                                    cv::rectangle( img, boundingboxes[ b ], CV_RGB( 255, 0, 0 ), 3 );
                                    falsePositive ++;
                                }
                        }

                        else{

                            std::vector<bool> bbCounter;
                            std::vector<bool> gtCounter( groundTruths.size(), 0 );

                            // match candidate with ground truth
                            for( unsigned int b = 0;  b < boundingboxes.size(); b++ ){

                                if( scores[ b ] < threshold )
                                    continue;

                                bbCounter.push_back( 0 );
                                std::vector< float > gtStrengths( groundTruths.size(), 0.f );
                                // check if the candidate has any matching groundtruth
                                for( unsigned int g = 0; g < groundTruths.size(); g++ ){

                                    // check if the groundtruth is already matched with some other boundingbox already
                                    // if not find the match for this bounding box

                                    if( !gtCounter[ g ] ) // matched with the highest strength
                                        findOverlap( groundTruths[ g ], boundingboxes[ b ], gtStrengths[g] );

                                }

                                std::vector< float >::iterator it;
                                it = std::max_element( gtStrengths.begin(), gtStrengths.end() );
                                int max_index = std::distance( gtStrengths.begin(), it );
                                double max_value = gtStrengths[ max_index ];

                                if( max_value > 0.5 ){

                                    bbCounter.back() = 1;
                                    gtCounter[ max_index ] = 1;
                                }

                                if(bbCounter.back() && pose_present){

                                    // show cloud
                                    if(0){

                                        // create normal lines out of groundtruth and detection
                                        // project both the vectors into point cloud
                                        cv::Point2f img_center(img.cols/2.f, img.rows/2.f);
                                        cv::Point2f objcenter(groundTruths[ max_index ].x +groundTruths[ max_index ].width/2.f, groundTruths[ max_index ].y + groundTruths[ max_index ].height/2.f);
                                        float depth = depthImg.at<unsigned short>(int(objcenter.y), int(objcenter.x))/1000.f; // convert into meters
                                        cv::Point3f center3D = P3toR3(objcenter, img_center, depth);

                                        pcl::PointXYZ center;
                                        center.x = center3D.x;
                                        center.y = center3D.y;
                                        center.z = center3D.z;

                                        pcl::PointXYZ normal_gt;
                                        normal_gt.x = center.x + groundPose[0];
                                        normal_gt.y = center.y + groundPose[1];
                                        normal_gt.z = center.z + groundPose[2];

                                        pcl::PointXYZ normal_dt;
                                        normal_dt.x = center.x + pose_z[b][0];
                                        normal_dt.y = center.y + pose_z[b][1];
                                        normal_dt.z = center.z + pose_z[b][2];

                                        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("3D Viewer") );
                                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

                                        cv::imshow("img", img); cv::waitKey(0);
                                        cv::imshow("depthimg", depthImg); cv::waitKey(0);
                                        imagesToPointCloud(depthImg, img,  cloud);
                                        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_rgb( cloud );

                                        viewer->setBackgroundColor( 1, 1, 1 );
                                        viewer->addPointCloud<pcl::PointXYZRGB> (cloud, cloud_rgb, "changed cloud");
                                        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );

                                        viewer->addLine( center, normal_gt, 0, 0, 255, "ground truth" );
                                        viewer->addLine( center, normal_dt, 0, 255,0, "detection" );

                                        while (!viewer->wasStopped ()){

                                            viewer->spinOnce (100);
                                            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

                                        }
                                    }

                                    // find angle and push back to histogram

                                    float diff_angle = acos(groundPose.dot(pose_z[b]));
                                    out_orientations << diff_angle << "\n";

                                }

                                if( bbCounter.back() )
                                    cv::rectangle( img, boundingboxes[ b ], CV_RGB( 0, 255, 0 ), 3 );

                                else
                                    cv::rectangle( img, boundingboxes[ b ], CV_RGB( 255, 0, 0), 3 );

                            }

                            // count tp, fp, fn
                            for( unsigned int g = 0; g < gtCounter.size(); g++ ){

                                falseNegative += ( !gtCounter[ g ] );
                                truePositive += ( gtCounter[ g ] );
                            }

                            for( unsigned int b = 0; b < bbCounter.size(); b++ )
                                falsePositive += ( !bbCounter[ b ] );

                        }

//                        cv::imshow( "Test Image", img ); cv::waitKey( 20 );
                    }

                    // save values in vector
                    vector< float > values(4);
                    values[0] = truePositive;
                    values[1] = falsePositive;
                    values[2] = trueNegative;
                    values[3] = falseNegative;
                    tftf.push_back(values);

                    //add fp, fn, tp, tn to fp_all, fn_all, tp_all and so on

                    falseNegative_all[threshold] += falseNegative;
                    truePositive_all[threshold] += truePositive;
                    falsePositive_all[threshold] += falsePositive;

                }

                vector< vector< float > > PRA;
                calcPRA(tftf, PRA);

                for(unsigned int  i = 0;  i < PRA.size(); i++ )
                   out_individual << PRA[i][0] << " " << PRA[i][1] << " "<< PRA[i][2] << endl;

                out_individual << endl;
                out_individual << endl;
            }

            out_individual.close();

        }

        vector< vector< float > > PRA;
        calcPRA_all(truePositive_all, falsePositive_all, falseNegative_all, PRA);

        ofstream out_all(filename_all_dataset.c_str());
        if(out_all.is_open()){
            out_all << minThresh << " " << maxThresh << " " << step << endl;
            out_all << endl;
            for(unsigned int  i = 0;  i < PRA.size(); i++ )
               out_all << PRA[i][0] << " " << PRA[i][1] << " "<< PRA[i][2] << endl;
        }
        out_all.close();
        out_orientations.close();

        return 0;
    }
}
