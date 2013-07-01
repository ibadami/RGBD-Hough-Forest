#include <iostream>
#include "config.h"
#include "objectGeometry.h"

using namespace std;

void generateNewTrainingImages(std::vector<objectConfig>& objects, string& trainclasspath, std::vector<string>& table_wood_image, std::vector<string>& intensity_gradient_images, bool trainNeg){

    //generate table plane for each pitch angle
    std::vector < Plane > table_plane( 4 );
    std::vector < cv::Point3f > table_center( 4 );

    for( int p = 0;  p < 4; p++ ){

        if( p == 2 ) // no pitch for this index in RGBD dataset
            continue;

        // Load images
        cv::Mat rgbImg = cv::imread( (trainclasspath + "/" + objects[ 0 ].filenames[ 0 ][ p ][ 0 ] + ".png").c_str(), CV_LOAD_IMAGE_COLOR );
        cv::Mat depthImg = cv::imread( (trainclasspath + "/" + objects[ 0 ].filenames[ 0 ][ p ][ 0 ] + "_filleddepth.png").c_str(),CV_LOAD_IMAGE_ANYDEPTH );

        std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > planePoints ;
        Eigen::Matrix4d referenceTransform = Eigen::Matrix4d::Identity();
        pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud ( new pcl::PointCloud< pcl::PointXYZRGB > );
        cv::Mat mask;
        images2PointCloud( depthImg, rgbImg, cloud, mask );

        // get plane
        selectPlane( rgbImg, cloud, referenceTransform, planePoints, table_plane[p] ) ;
        // get center of the table
        Eigen::Vector3f tempCenter = getTurnTableCenter( rgbImg, cloud, referenceTransform, table_plane[p] ) ;
        table_center[ p ] =  cv::Point3f( tempCenter[ 0 ], tempCenter[ 1 ], tempCenter[ 2 ] ) ;

    }

    cv::destroyWindow("Select turn table Plane");
    cv::destroyWindow("Select center of turn table ");

    cv::RNG rng(cv::getTickCount());

    unsigned int num_extra_objects = 8;
    if(trainNeg)
        num_extra_objects = 46;
    positions pObj;
    pObj.getpositions( num_extra_objects, trainNeg );
    num_extra_objects = pObj.objPose.size();

    float pitch[ 4 ];
    pitch[ 0 ] = std::acos(table_plane[0].getNormal().dot(Eigen::Vector3f(0.f,0.f,1.f)));
    pitch[ 1 ] = std::acos(table_plane[1].getNormal().dot(Eigen::Vector3f(0.f,0.f,1.f)));
    pitch[ 2 ] = 0;
    pitch[ 3 ] = std::acos(table_plane[3].getNormal().dot(Eigen::Vector3f(0.f,0.f,1.f)));
    // for each object
    for( int i = 0; i < objects.size(); i++ ){

        for( int j = 0; j < objects[ i ].filenames.size(); j++ ){// for each instance

            for( int k = 0; k <  objects[ i ].filenames[ j ].size(); k++ ){ // for every kth pitch angle

                if(k == 2)
                    continue;

                for( int m = 0; m < objects[ i ].filenames[ j ][ k ].size(); m += 5 ){ // for every mth pose
                    string imgname, dimgname;
                    string objname = objects[i].filenames[j][k][m];
                    objname.erase(objname.find_first_of("/"));
                    string filename = objects[i].filenames[j][k][m];
                    filename.erase(0,filename.find_last_of("/")+1);
		    
                    if(trainNeg){
                        imgname = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/neg_/" + objname + "/" + filename + ".png");
                        dimgname = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/neg_/" + objname + "/" + filename + "_filleddepth.png");
                    }else{

                        imgname = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + ".png");
                        dimgname = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + "_filleddepth.png");
                    }

                    ifstream ifile(imgname.c_str());
                    ifstream difile(dimgname.c_str());
                    if (ifile && difile){ // The file exists, and is open for input
                        cout<<  imgname << " Exists!"<< endl;
			 cout<<  dimgname << " Exists!"<< endl;
                        continue;
                    }
                    
                    // if pose or mask or location is not available
                    string posefile, maskfile, locationfile;
		    
                    posefile = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train/" + objects[i].filenames[j][k][m] + "_pose.txt");
                    maskfile = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train/" + objects[i].filenames[j][k][m] + "_mask.png");
		    locationfile = ("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train/" + objects[i].filenames[j][k][m] + "_loc.txt");
		    ifstream pfile(posefile.c_str());
                    ifstream mfile(maskfile.c_str());
		    ifstream lfile(locationfile.c_str());
                    
		      if((!pfile) || (!mfile) || (!lfile)){
			cout<<"pfile = "<< pfile <<endl;
cout<<"mfile = " << mfile <<endl;
cout<< "lfile = " <<lfile << endl;

			continue;
}
                    pcl::PointCloud< pcl::PointXYZRGB >::Ptr all_objects_cloud ( new pcl::PointCloud< pcl::PointXYZRGB > );
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> all_objects_cloud_rgb( all_objects_cloud );

                    std::vector< cv::Mat >vRgbImg; vRgbImg.reserve( num_extra_objects + 2 );
                    std::vector< cv::Mat >vDepthImg; vDepthImg.reserve( num_extra_objects + 2 );
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    // Load main images for object in interest
                    cv::Mat rgbImg = cv::imread( (trainclasspath + "/" + objects[i].filenames[j][k][m] + ".png").c_str(), CV_LOAD_IMAGE_COLOR );
                    cv::Mat depthImg = cv::imread((trainclasspath + "/" + objects[i].filenames[j][k][m] + "_filleddepth.png").c_str(),CV_LOAD_IMAGE_ANYDEPTH);
                    cv::Mat maskImg = cv::imread((trainclasspath + "/" + objects[i].filenames[j][k][m] + "_mask.png").c_str(),CV_LOAD_IMAGE_ANYDEPTH);
                    cv::Size2f imgSize( rgbImg.cols, rgbImg.rows );


                    if ( !trainNeg ) {

                        if(maskImg.empty())
                            continue;

                        pcl::PointCloud< pcl::PointXYZRGB >::Ptr mainCloud (new pcl::PointCloud< pcl::PointXYZRGB >);
                        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> mainCloud_rgb( mainCloud );
                        images2PointCloud(depthImg, rgbImg, mainCloud, maskImg);

                        cv::Mat depthImg_main, rgbImg_main;
                        pointCloud2Images(mainCloud, depthImg_main, rgbImg_main, imgSize);
                        fillHoles( rgbImg_main, depthImg_main, rgbImg_main, depthImg_main, 3.f );


                        vRgbImg.push_back(rgbImg_main);
                        vDepthImg.push_back(depthImg_main);

                        all_objects_cloud->resize( all_objects_cloud->size() + mainCloud->size() );
                        *all_objects_cloud += *mainCloud;

                        //                        // show cloud
                        //                        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("with extra objects") );

                        //                        viewer->setBackgroundColor( 1, 1, 1 );
                        //                        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );
                        //                        viewer->addPointCloud<pcl::PointXYZRGB> (mainCloud, mainCloud_rgb, "all objects");

                        //                        while (!viewer->wasStopped ()){

                        //                            viewer->spinOnce (100);
                        //                            boost::this_thread::sleep (boost::posix_time::microseconds (100000));

                        //                        }
                    }
                    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    // generate random color for plane
                    int color_index =  rng.operator ()(table_wood_image.size());
                    cv::Mat table_wood_img = cv::imread( table_wood_image[color_index], CV_LOAD_IMAGE_COLOR);
                    pcl::PointCloud< pcl::PointXYZRGB >::Ptr plane ( new pcl::PointCloud< pcl::PointXYZRGB >);
                    generatePlane( imgSize, table_plane[ k ], plane , table_wood_img);

                    cv::Mat depthImg_plane, rgbImg_plane;
                    pointCloud2Images(plane, depthImg_plane, rgbImg_plane, imgSize);
                    fillHoles( rgbImg_plane, depthImg_plane, rgbImg_plane, depthImg_plane, 3.f );

                    vRgbImg.push_back(rgbImg_plane);
                    vDepthImg.push_back(depthImg_plane);

                    all_objects_cloud->resize( all_objects_cloud->size() + plane->size() );
                    *all_objects_cloud += *plane;

                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    // generate random objects and load their point cloud

                    int count = 0;
                    int indx[ 2 ]; indx[ 0 ] = 0; indx[ 1 ] = 5;

                    std::vector< string> rand_filename;
                    std::vector< float> rand_poseAngle;
                    std::vector< Eigen::Matrix4f, Eigen::aligned_allocator< Eigen::Matrix4f > > rand_transformationOC;
                    rand_transformationOC.reserve(num_extra_objects);
                    rand_filename.reserve(num_extra_objects);
                    rand_poseAngle.reserve(num_extra_objects);

                    while( count < num_extra_objects ){

                        int rand_object = rng.operator ()( objects.size() );
                        if( rand_object == i )
                            continue;

                        int rand_instance = rng.operator ()( objects[ rand_object ].filenames.size() );

                        int num1 =  rng.operator ()( 2 ) % 2;
                        int num2 = rng.operator ()( objects[ rand_object ].filenames[ rand_instance ][ k ].size());
                        int rand_pose = num2 - ( num2 % 10 ) + indx[num1];

                        if( rand_pose < 0 || rand_pose >= objects[ rand_object ].filenames[ rand_instance ][ k ].size())
                            continue;

                        rand_filename.push_back( objects[ rand_object ].filenames[ rand_instance ][ k ][ rand_pose ]) ;
                        float poseAngle = readPose(trainclasspath + "/" + rand_filename.back() + "_pose.txt");
                        float pitchAngle = pitch[k];

                        Eigen::Matrix4f tempTransformation;
                        calcObject2CameraTransformation( poseAngle, pitchAngle, table_center[ k ], tempTransformation );
                        rand_poseAngle.push_back(poseAngle);
                        rand_transformationOC.push_back(tempTransformation);
                        //                        drawTransformation(rgbImg, depthImg, tempTransformation);
                        count++;
                    }


                    for( int obj = 0;  obj < num_extra_objects; obj++ ){

                        // Load images
                        cv::Mat rgbImg_ = cv::imread( (trainclasspath + "/" + rand_filename[ obj ]  + ".png").c_str(), CV_LOAD_IMAGE_COLOR );
                        cv::Mat depthImg_ = cv::imread( (trainclasspath + "/" + rand_filename[ obj ]  + "_filleddepth.png").c_str(),CV_LOAD_IMAGE_ANYDEPTH );
                        cv::Mat maskImg_ = cv::imread( (trainclasspath + "/" + rand_filename[ obj ] + "_mask.png").c_str(),CV_LOAD_IMAGE_ANYDEPTH );
                        if(maskImg.empty())
                            continue;

                        cv::Mat maskImg_erode; cv::erode( maskImg_,maskImg_erode,cv::Mat( 7, 7, CV_8U, 1 ),cv::Point( -1, -1 ), 2 );
                        pcl::PointCloud< pcl::PointXYZRGB >::Ptr tempCloud( new pcl::PointCloud< pcl::PointXYZRGB >() );
                        images2PointCloud( depthImg_, rgbImg_, tempCloud, maskImg_erode );

                        pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_moved( new pcl::PointCloud< pcl::PointXYZRGB > );
                        moveOverPlane( tempCloud, rand_transformationOC[ obj ], cloud_moved, pObj.objPose[ obj ], rand_poseAngle[ obj ] , table_plane[k].point);

                        cv::Mat depthImg_moved, colorImg_moved, filledRgbImg_moved, filledDepthImg_moved;
                        pointCloud2Images( cloud_moved, depthImg_moved, colorImg_moved, imgSize );

                        fillHoles( colorImg_moved, depthImg_moved, filledRgbImg_moved, filledDepthImg_moved, 3.f );
                        vRgbImg.push_back( filledRgbImg_moved );
                        vDepthImg.push_back( filledDepthImg_moved );

                        all_objects_cloud->resize( all_objects_cloud->size() + cloud_moved->size() );
                        *all_objects_cloud += *cloud_moved;

                    }

                    //                    // show cloud
                    //                    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("with extra objects") );

                    //                    viewer->setBackgroundColor( 0, 0, 0 );
                    //                    viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );
                    //                    viewer->addPointCloud<pcl::PointXYZRGB> (all_objects_cloud, all_objects_cloud_rgb, "all objects");

                    //                    while (!viewer->wasStopped ()){

                    //                        viewer->spinOnce (100);
                    //                        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

                    //                    }

                    cv::Mat finalImg, finalDepthImg;
                    CombineImages(vRgbImg, vDepthImg, finalImg, finalDepthImg );

                    // change gamma
                    int gamma_index = rng.operator ()(intensity_gradient_images.size());
                    cv::Mat intensity_gradient_img  = cv::imread(intensity_gradient_images[gamma_index], CV_LOAD_IMAGE_ANYDEPTH);
                    cv::Mat changedGammaImg;
                    changeGamma(finalImg, changedGammaImg, intensity_gradient_img );



                    if(trainNeg){

                        cv::imwrite(("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/neg_/" + objname + "/" + filename + ".png").c_str(), changedGammaImg);
                        cv::imwrite(("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/neg_/" + objname + "/" + filename + "_filleddepth.png").c_str(), finalDepthImg);

                    }else{
                        string src = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train/" + objects[i].filenames[j][k][m] + "_pose.txt";
                        string dst = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + "_pose.txt";
                        system(("cp " + src + " " + dst).c_str());

                        src = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train/" + objects[i].filenames[j][k][m] + "_loc.txt";
                        dst = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + "_loc.txt";
                        system(("cp " + src + " " + dst).c_str());


                        src = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train/" + objects[i].filenames[j][k][m] + "_mask.png";
                        dst = "/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + "_mask.png";
                        system(("cp " + src + " " + dst).c_str());


                        cv::imwrite(("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + ".png").c_str(), changedGammaImg);
                        cv::imwrite(("/home/local/stud/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/newTrain2/" + objects[i].filenames[j][k][m] + "_filleddepth.png").c_str(), finalDepthImg);


                    }
                    cv::imshow("img", changedGammaImg);

                    if( 1 ){

                        cv::Mat temp( finalDepthImg.rows, finalDepthImg.cols,CV_32F, 0.f );
                        double minVal, maxVal;
                        cv::minMaxLoc(finalDepthImg, &minVal, &maxVal, 0, 0);
                        for( int y = 0; y < finalDepthImg.rows; y++ )
                            for( int x = 0; x < finalDepthImg.cols; x++ )
                                temp.at< float >( y, x ) = finalDepthImg.at< unsigned short >( y,x ) /  maxVal;

                        cv::imshow( "depthimage", temp );

                    }

                    cv::waitKey( 20 );


                    //                    // convert back to cloud to debug
                    //                    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_debug(new pcl::PointCloud< pcl::PointXYZRGB >);
                    //                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_debug_rgb( cloud_debug );
                    //                    cv::Mat mask;
                    //                    images2PointCloud(finalDepthImg, finalImg, cloud_debug, mask);

                    //                    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_debug ( new pcl::visualization::PCLVisualizer ("debug") );
                    //                    viewer_debug->setBackgroundColor( 0, 0, 0 );
                    //                    viewer_debug->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );
                    //                    viewer_debug->addPointCloud<pcl::PointXYZRGB> (cloud_debug,cloud_debug_rgb, "debug images");

                    //                    while (!viewer_debug->wasStopped ()){

                    //                        viewer_debug->spinOnce (100);
                    //                        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

                    //                    }

                }
            }
        }
    }
}



int main( int argc, char* argv[] ){

    if( argc < 2 ){
        cout<< " Please provide configuration files for objects " << endl;

    }

    std::vector< objectConfig > objects;
    if( argc > 1 ){

        // read all config files
        loadConfig( argv[ 1 ], objects );
    }

    bool trainNeg = 0;
    if( argc > 2 )
        trainNeg = argv[ 2 ];

    string trainclasspath = "/home/local/badami/Uni_Bonn/Master_thesis/datasets/rgbd-dataset/train";
    string table_wood_file = "../table_wood.txt";
    string intensity_gradient_file = "../intensity_gradient.txt";

    std::vector< string > table_wood_images;
    std::vector< string > intensity_gradient_images;

    readIntensity_gradient( intensity_gradient_file, intensity_gradient_images );
    readTable_wood( table_wood_file, table_wood_images );

    generateNewTrainingImages( objects, trainclasspath, table_wood_images, intensity_gradient_images, trainNeg);

    return 0;
}
