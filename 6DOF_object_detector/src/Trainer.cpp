
// Author: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#include "Trainer.h"

CRForestTraining::CRForestTraining()
{}

void CRForestTraining::getObjectSize(Parameters &p, vector< vector<string> >& vFilenames, vector<vector<CvPoint> >& vCenter, vector< vector<  CvRect > > &vBBox ){

    for( int l = 0; l < p.nlabels; ++l ) {

        if( p.class_structure[ l ]!= 0 ){
            float  bbWidth = 0;
            float  bbHeight = 0;
            for( int i = 0; i < ( int )vFilenames[ l ].size(); ++i) {

                // Load depth image
                string filename = vFilenames[ l ][ i ];
                int size_of_string = vFilenames[ l ][ i ].size();

                filename.replace( size_of_string - 4, 15, "_filleddepth.png" );
                cv::Mat depthImg;
                depthImg = cv::imread( ( p.trainclasspath + "/" + filename ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );

                float scale = 1000.f / depthImg.at< unsigned short >( vCenter[ l ][ i ] );

                bbWidth += vBBox[ l ][ i ].width / scale;
                bbHeight += vBBox[ l ][ i ].height / scale;

            }
            // take average of bounding box size
            p.objectSize.first = bbWidth / vFilenames[l].size();
            p.objectSize.second = bbHeight / vFilenames[l].size();
        }

    }
}
void CRForestTraining::generate3DModel( Parameters &param, vector< vector<string> >& vFilenames, vector<vector<CvPoint> >& vCenter, vector< vector<  CvRect > > &vBBox , vector<vector<float> > &vPoseAngle, vector<vector<float> > &vPitchAngle, vector<cv::Point3f> &cg, cv::Point3f &bbSize ){

    // check if model is already generated
    string filePath = param.object_models_path +"/" + param.objectName + ".txt";
    ifstream in(filePath.c_str());
    if(in.is_open()){
        in >> param.objectSize.first; in >> param.objectSize.second;
        cg.resize(3);
        for(int i = 0; i < 3; i++ ){
            in >> cg[i].x; in >> cg[i].y; in >> cg[i].z;
        }
        in >> bbSize.x; in >> bbSize.y; in >> bbSize.z;
        in.close();

    }else{

        for(int l = 0; l< param.nlabels; l++){

            if( param.class_structure[l] == 0 )
                continue;

            float minHeight = 0, maxHeight = 100;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

            std::vector< Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > transformationMatrixOC(3);

            for( int pNr = 0; pNr < 3; pNr++ ){

                std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > convexHull;
                std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > planePoints;
                Plane table_plane;
                Eigen::Matrix4d referenceTransform = Eigen::Matrix4d::Identity();

                float pose = vPoseAngle[ l ][ 2 * pNr ];
                float pitch = vPitchAngle[ l ][ 2 * pNr];

                /////////////////////////////////////////////////////  1st image  ////////////////////////////////////////////////////

                // Load image
                string trainclasspath_ = param.trainclasspath;
                trainclasspath_.replace(param.trainclasspath.size() - 9, 9 , "train");
                cv::Mat img = cv::imread( (trainclasspath_ + "/" + vFilenames[ l ][ 2 * pNr ]).c_str(), CV_LOAD_IMAGE_COLOR );
                int size_of_string = vFilenames[ l ][ 2 * pNr ].size();

                if( img.empty() ) {
                    cout << "Could not load image file: " << (trainclasspath_ + "/" + vFilenames[l][2 * pNr]).c_str() << endl;
                    exit(-1);
                }

                // Load depth image
                string filename = vFilenames[l][2 * pNr];
                if( param.class_structure[l] != 0 ){
                    filename.replace( size_of_string - 4, 9, "_depth.png" );
                }else{
                    filename.replace( size_of_string - 4, 9, "_depth.png" );
                }
                cv::Mat depthImg = cv::imread( ( trainclasspath_ + "/" + filename ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );

                if( depthImg.empty() ) {
                    cout << "Could not load image file: " << (( trainclasspath_ + "/" + filename ).c_str()) << endl;
                    exit(-1);
                }

                // create point cloud
                pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud0 ( new pcl::PointCloud< pcl::PointXYZRGB > );
                Surfel::imagesToPointCloud( depthImg, img, cloud0 );

                // segment object cloud
                selectConvexHull( img, cloud0, referenceTransform, convexHull );
                selectPlane( img, cloud0, referenceTransform, planePoints,table_plane );
                Eigen::Vector3d turnTable_center = getTurnTableCenter( img, cloud0, referenceTransform, table_plane );

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud0 ( new pcl::PointCloud<pcl::PointXYZRGB> );
                getObjectPointCloud( cloud0, minHeight, maxHeight, convexHull, table_plane , turnTable_center, objectCloud0 );

                // get object coordinate system
                cv::Point3f center(turnTable_center[0], turnTable_center[1], turnTable_center[ 2 ] - 0.01f); // added correction term of 0.01f in center
                CRPixel::calcObject2CameraTransformation( pose, pitch, center, transformationMatrixOC[ pNr ] );
                Eigen::Matrix4d transformationMatrixOC_inverse = transformationMatrixOC[pNr].inverse();

                // transform cloud to object coordinate system
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud0_transformed ( new pcl::PointCloud<pcl::PointXYZRGB> );
                pcl::transformPointCloud (*objectCloud0, *objectCloud0_transformed, transformationMatrixOC_inverse.cast<float>());

                // add cloud to common object cloud
                objectCloud->resize(objectCloud->size() + objectCloud0_transformed->size() );
                *objectCloud +=  *objectCloud0_transformed;

                //////////////////////////////////////////////////  2nd image  ////////////////////////////////////////////////////

                // Load image
                img = cv::imread( ( trainclasspath_ + "/" + vFilenames[l][ 2 * pNr + 1 ] ).c_str(), CV_LOAD_IMAGE_COLOR );
                size_of_string = vFilenames[l][ 2 * pNr + 1 ].size();

                if( img.empty() ) {
                    cout << "Could not load image file: " << ( trainclasspath_ + "/" + vFilenames[l][ 2 * pNr + 1 ] ).c_str() << endl;
                    exit(-1);
                }

                // Load depth image
                filename = vFilenames[l][ 2 * pNr + 1 ];
                if( param.class_structure[l] != 0 ){
                    filename.replace( size_of_string - 4, 9, "_filleddepth.png" );
                }else{
                    filename.replace( size_of_string - 4, 9, "_depth.png" );
                }
                depthImg = cv::imread( ( trainclasspath_ + "/" + filename ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );

                if( depthImg.empty() ) {
                    cout << "Could not load image file: " << (( trainclasspath_ + "/" + filename ).c_str()) << endl;
                    exit(-1);
                }

                // create object cloud
                pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud1 ( new pcl::PointCloud< pcl::PointXYZRGB >);
                Surfel::imagesToPointCloud(depthImg, img, cloud1);

                // segment point cloud using already calculated convex hull and turn table plane
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud1 ( new pcl::PointCloud<pcl::PointXYZRGB> );
                getObjectPointCloud( cloud1, minHeight, maxHeight, convexHull, table_plane , turnTable_center, objectCloud1 );

                // transform camera to object coordinates
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud_temp ( new pcl::PointCloud<pcl::PointXYZRGB> );
                pcl::transformPointCloud (*objectCloud1, *objectCloud_temp, transformationMatrixOC_inverse.cast<float>());

                // rotate around z axis by approx 180 degree
                float rotateAngle = vPoseAngle[l][2*pNr +1] - vPoseAngle[l][2*pNr];

                Eigen::AngleAxisf rotate;
                rotate.angle() = rotateAngle/180.f * PI;
                rotate.axis() = Eigen::Vector3f(0.f, 0.f, 1.f); // rotate around z axis
                Eigen::Matrix3d rotationMatrix(rotate.cast<double>());

                Eigen::Matrix4d transformMatrix180 = Eigen::Matrix4d::Identity();
                transformMatrix180.block<3,3>(0,0) = rotationMatrix;

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud1_transformed ( new pcl::PointCloud<pcl::PointXYZRGB> );
                pcl::transformPointCloud (*objectCloud_temp, *objectCloud1_transformed, transformMatrix180.cast<float>());

                // add point cloud to common object cloud
                objectCloud->resize(objectCloud->size() + objectCloud1_transformed->size() );
                *objectCloud +=  *objectCloud1_transformed;

            }// pitch angle for loops end here

//            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D object model"));
//            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> objectCloud_rgb(objectCloud);
//            viewer->setBackgroundColor(1,1,1);
//            viewer->addPointCloud<pcl::PointXYZRGB> (objectCloud, objectCloud_rgb, "all clouds");
//            viewer->addCoordinateSystem(0.1, 0.f, 0.f, 0.f, 0.f);

//            while (!viewer->wasStopped ()){
//                viewer->spinOnce (100);
//                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//            }

            cg.resize(3);
            get3DBoundingBox( objectCloud, transformationMatrixOC, cg, bbSize );

            getObjectSize(param, vFilenames, vCenter, vBBox);

            // write model parameters to .text file

            string filePath = param.object_models_path +"/" + param.objectName + ".txt";
            string execstr = "mkdir "+ param.outpath + "/object_models/";
            system( execstr.c_str() );

            ofstream out(filePath.c_str());
            if(out.is_open()){

                out << param.objectSize.first << " " <<  param.objectSize.second << endl;
                cg.resize( 3 );
                for( int i = 0; i < 3; i++ ){

                    out << cg[i].x << " " << cg[i].y<< " " << cg[i].z << endl;
                }
                out << bbSize.x << " " << bbSize.y<< " " << bbSize.z << endl;
            }
            out.close();


        }// label's for loop end here
    }
}

void CRForestTraining::extract_Pixels( rawData& data , const Parameters &p, CRPixel& Train, cv::RNG* pRNG ) {

    vector<vector<string> >& vFilenames = data.vFilenames;
    vector<vector<CvRect> >& vBBox = data.vBBox;
    vector<vector<CvPoint> >& vCenter = data.vCenter;
    vector<vector<float> >& vPoseAngle = data.vPoseAngle; // in degrees
    vector<vector<float> >& vPitchAngle = data.vPitchAngle; // in degrees

    std::vector<cv::Point3f>& cg  = data.cg;
    cv::Point3f& bbSize = data.bbSize;


    // for each class/label
    for( int l = 0; l < p.nlabels; ++l ) {

        cout << "Label: " << l << " " << p.class_structure[ l ] << " ";

        int subsamples = 0;
        if ( p.class_structure[ l ] == 0 )
            subsamples = p.subsample_images_pos;
        else
            subsamples = p.subsample_images_neg;

        // load postive images and extract patches
        for( int i = 0; i < ( int )vFilenames[ l ].size(); i++) {

            if( i % 100 == 0 )
                cout << i << " " << flush;
            if( subsamples <= 0 || ( int )vFilenames[ l ].size() <= subsamples || ( pRNG->operator float()*double( vFilenames[ l ].size()) < double( subsamples ) ) ) {

                // Load image
                cv::Mat img = cv::imread(( p.trainclasspath + "/" + vFilenames[ l ][ i ]).c_str(), CV_LOAD_IMAGE_COLOR );
                int size_of_string = vFilenames[ l ][ i ].size();

                if(img.empty()) {
                    cout << "Could not load image file: " << ( p.trainclasspath + "/" + vFilenames[ l ][ i ]).c_str() << endl;
                    exit(-1);
                }

                // Load depth image
                string filename = vFilenames[ l ][ i ];
                //filename.replace( size_of_string - 4, 15, "_filleddepth.png" );
                filename.replace( size_of_string - 4, 15, "_filleddepth.png" );

                cv::Mat depthImg( img.rows, img.cols, CV_16UC1 );
                depthImg = cv::imread( ( p.trainclasspath + "/" + filename ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );
                if(depthImg.empty()) {
                    cout << "Could not load depth image file: " << p.trainclasspath << "/" << filename << endl;
                    exit(-1);
                }

//                cout << "depth img " <<( p.trainclasspath + "/" + filename ).c_str()<<endl;

                // Load mask image
                filename = vFilenames[ l ][ i ];
                //filename.replace( size_of_string - 4, 15, "_filleddepth.png" );
                filename.replace( size_of_string - 4, 15, "_mask.png" );

                cv::Mat maskImg = cv::Mat::ones( img.rows, img.cols, CV_8UC1 );

                if( p.class_structure[ l ] != 0 ){

                    maskImg = cv::imread( ( p.trainclasspath + "/" + filename ).c_str(),CV_LOAD_IMAGE_ANYDEPTH );

                    // Extract positive training patches
                    int pitchIndex;
                    float pose = vPoseAngle[ l ][ i ];
                    float pitch = vPitchAngle[ l ][ i ];

                    // write switch and find pitchIndex

                    switch ( int( pitch ) ){
                    case 23:
                        pitchIndex = 0;
                        break;
                    case 38:
                        pitchIndex = 1;
                        break;
                    case 54:
                        pitchIndex = 2;
                        break;
                    default:
                        std::cout<< "error reading pitch angle"<< endl;
                    }

                    Eigen::Matrix4d transformationMatrixOC;
                    CRPixel::calcObject2CameraTransformation( pose, pitch, cg[ pitchIndex ], transformationMatrixOC );

                    // get object coordinate system

                    Train.extractPixels( p, img, depthImg, maskImg, p.samples_pixel_pos, l, i, &vBBox[ l ][ i ], &vCenter[ l ][ i ], &cg[ pitchIndex ] , &bbSize , &transformationMatrixOC );

                }else{

                    cv::Point3f cg_(0.f,0.f,0.f);
                    cv::Point3f bbSize_(0.f,0.f,0.f);
                    Eigen::Matrix4d transformationMatrixOC = Eigen::Matrix4d::Identity() ;

                    Train.extractPixels( p, img, depthImg, maskImg, p.samples_pixel_neg, l, i , &vBBox[ l ][ i ], &vCenter[ l ][ i ], &cg_ , &bbSize_ , &transformationMatrixOC  );
                }
            }
        }cout << endl;
    }cout << endl;
}

void CRForestTraining::generateNewImage( cv::Mat &mask, cv::Mat &img, cv::Mat &newImg ){

    img.copyTo(newImg);

    cv::RNG rng( cv::getTickCount() );

    for(int r = 0; r < img.rows; r++){
        for(int c = 0; c < img.cols; c++ ){

            if(mask.at<unsigned char>(r,c) > 0)
                continue;
            else{

                int red = rng.operator()(256);
                int green = rng.operator()(256);
                int blue = rng.operator()(256);

                // in BGR sequence
                cv::Vec3b rgb(blue, green, red);
                newImg.at<cv::Vec3b>(r,c) = rgb;

            }

        }
    }
}

void CRForestTraining::generateTrainingImage( cv::Mat &rgbImage, cv::Mat &depthImage ){

    std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > planePoints ;
    Plane table_plane ;
    Line ray;
    Eigen::Matrix4d referenceTransform = Eigen::Matrix4d::Identity();
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud (new pcl::PointCloud< pcl::PointXYZRGB >);
    Surfel::imagesToPointCloud(depthImage, rgbImage, cloud);

    int r = 255;
    int g = 0;
    int b = 0;


    pcl::RGB rgb;
    rgb.r = (uint8_t)r;
    rgb.g = (uint8_t)g;
    rgb.b = (uint8_t)b;
    rgb.a = 1;

    // get plane
    selectPlane( rgbImage, cloud, referenceTransform, planePoints, table_plane ) ;

    // for each pixel generate ray and find intersection with plane
    for( int x = 0; x < rgbImage.cols; x++ ){
        for( int y = 0; y < rgbImage.rows; y++ ){


            pcl::PointXYZRGB  &p = cloud->at(x,y);
            Eigen::Vector3d ptIntersection;

            // get line
            getLine( p, ray );
            getLinePlaneIntersection(ray, table_plane, ptIntersection);

            // check if point of intersection is closer to camera then the actual point
            if(ptIntersection[2] <= p.z){

                p.x = ptIntersection[0];
                p.y = ptIntersection[1];
                p.z = ptIntersection[2];

                p.rgba = rgb.rgba;

            }

        } // end for loop for y
    } // end for loop for x


    // show cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ("3D Viewer") );
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_rgb( cloud );

    viewer->setBackgroundColor( 1, 1, 1 );
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, cloud_rgb, "changed cloud");
    viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );

    while (!viewer->wasStopped ()){

        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

    }

} // end function
