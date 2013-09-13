// Author: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de
#include "utils.h"

void onMouse( int event, int x, int y, int flags, void* userdata ) {
    if( userdata ) {
        MouseEvent* data = (MouseEvent*) userdata;
        data->event = event;
        data->pt = cv::Point( x, y );
        data->buttonState = flags;
    }
}

// save floating point images
void saveFloatImage( char* buffer , IplImage* img){
    std::ofstream fp_out;
    fp_out.open(buffer);

    float x;
    x = float(img->width);
    fp_out.write(reinterpret_cast<char *>(&x),sizeof(float));
    x = float(img->height);
    fp_out.write(reinterpret_cast<char *>(&x),sizeof(float));

    int stepData;
    float* rawData;
    cvGetRawData( img, (uchar**)&(rawData), &stepData);
    stepData /= sizeof(rawData[0]);

    for (int cy = 0; cy < img->height; cy++){
        for (int cx = 0; cx < img->width; cx++){
            x = *(rawData + cx + cy*stepData);
            fp_out.write(reinterpret_cast<char *>(&x),sizeof(float));
        }
    }

    fp_out.close();
}

bool isInsideRect(cv::Rect* rect, int x, int y){

    if (x > rect->x && x < rect->x + rect->width && y > rect->y && y < rect->y + rect->height){
        return true;
    }else{
        return false;
    }
}

bool isInsideKernel2D(float x, float y, float cx, float cy , float radius){

    float sum = (x-cx)*(x-cx) + (y-cy)*(y-cy);

    if (sum > radius*radius)
        return false;

    return true;
}

// Calculate PCA over mask
void calcPCA(cv::Mat &img_mask, cv::Point2f &meanPoint, cv::Size2f &dimension, float &rotAngle){
    //generate 2 x 1 data matrix containing (x,y) coordinates of mask image
    std::vector<cv::Point2f> dataVector;
    dataVector.reserve(30000);
    for(int y = 0; y < img_mask.rows; y++){
        for(int x = 0; x< img_mask.cols;x++){
            if(img_mask.at<unsigned char>(y,x) > 0){
                dataVector.push_back(cv::Point2f(x,y));
            }
        }
    }

    cv::Mat data = cv::Mat(2, dataVector.size(), CV_32FC1);
    for(int k = 0; k < dataVector.size();k++){
        data.at<float>(0,k) = dataVector[k].x;
        data.at<float>(1,k) = dataVector[k].y;
    }


    cv::PCA pca;
    pca(data,cv::Mat(),CV_PCA_DATA_AS_COL);

    // check for mirroring
    float detEV = (pca.eigenvectors.at<float>(0,0) * pca.eigenvectors.at<float>(1,1) - pca.eigenvectors.at<float>(0,1) * pca.eigenvectors.at<float>(1,0) );
    if( detEV < 0 ) {
        pca.eigenvectors.at<float>(1,0) = -pca.eigenvectors.at<float>(1,0);
        pca.eigenvectors.at<float>(1,1) = -pca.eigenvectors.at<float>(1,1);
    }

    // project data on principal axis
    cv::Mat projectedData = pca.project(data);


    // get length of both principal direction;
    double minX, minY, maxX, maxY;
    cv::Point2i minXIdx, minYIdx, maxXIdx, maxYIdx;
    cv::minMaxLoc(projectedData.row(0), &minX, &maxX, &minXIdx, &maxXIdx);
    cv::minMaxLoc(projectedData.row(1), &minY, &maxY, &minYIdx, &maxYIdx);

    cv::Mat mean = pca.mean;

    float lx = (maxX-minX);
    float ly = (maxY-minY);
    float cx = (maxX+minX)/2.f + mean.at<float>(0);
    float cy = (maxY+minY)/2.f + mean.at<float>(1);

    meanPoint = cv::Point2f(cx,cy);
    dimension = cv::Size2f(lx,ly);


    // get rotation angle
    rotAngle = atan2(pca.eigenvectors.at<float>(0,1), pca.eigenvectors.at<float>(0,0)); // in Radians

    cv::Mat projectedMask = cv::Mat::zeros(img_mask.rows, img_mask.cols, CV_8UC1);

    for(int s = 0; s< projectedData.cols; s++){
        projectedMask.at<unsigned char>( projectedData.at<float>(1,s) + mean.at<float>(1), projectedData.at<float>(0,s) + mean.at<float>(0) ) = 255;

    }
    //    cv::imshow("projected", projectedMask);
    //    cv::imshow("mask", img_mask);
    //    cv::waitKey(0);

    //    cv::RotatedRect PCARect(cv::Point2f(cx,cy),cv::Size2f(lx,ly), rotAngle*180/M_PI);
}

Eigen::Matrix3d quaternionToMatrix(Eigen::Quaterniond &q){

    Eigen::Matrix3d rotationMatrix;
    rotationMatrix.setIdentity();
    float x,y,z,w;
    x = q.coeffs()[0]; y = q.coeffs()[1]; z = q.coeffs()[2]; w = q.coeffs()[3];

    /*rotationMatrix[0][0] = w*w+x*x-y*y-z*z;
    rotationMatrix.*/
    rotationMatrix(0,0) = w*w+x*x-y*y-z*z;

    rotationMatrix(0,1) = 2*x*y-2*w*z;

    rotationMatrix(0,2) = 2*x*z+2*w*y;

    rotationMatrix(1,0) = 2*x*y+2*w*z;

    rotationMatrix(1,1) = w*w-x*x+y*y-z*z;

    rotationMatrix(1,2) = 2*y*z+2*w*x;

    rotationMatrix(2,0) = 2*x*z-2*w*y;

    rotationMatrix(2,1) = 2*y*z-2*w*x;

    rotationMatrix(2,2) = w*w-x*x-y*y+z*z;

    //    std::cout << rotationMatrix << std::endl;

    return rotationMatrix;

}

// generalized quaternion interpolation
Eigen::Quaterniond quatInterp(const std::vector<Eigen::Quaterniond>& rotation){

    const double invSamples = 1.0 / ((double)rotation.size());
    const int maxIterations = 1;
    Eigen::Quaterniond meanRotation;


    meanRotation.setIdentity();

    for( int k = 0; k < maxIterations; k++ ) {

        Eigen::Vector3d errorU;
        errorU.setConstant( 0.0 );

        for( int i = 0; i < rotation.size(); i++ ) {

            //            Eigen::Quaterniond rotation;
            //            rotation = Eigen::Matrix3d( transformations[i].block( 0, 0, 3, 3 ) ).cast<double>();

            if( k == 0 && i == 0)
                meanRotation = rotation[i];

            Eigen::Quaterniond diffQ = meanRotation.inverse()* rotation[i];
            Eigen::Vector3d u = diffQ.vec();
            float sinTheta = u.norm();
            if( u.norm() > 1e-10 )
                u.normalize();
            else {
                sinTheta = 0.f;
                u *= 0.f;
            }
            float cosTheta = diffQ.w();
            float theta = atan2( sinTheta, cosTheta );
            if( theta < 0 )
                theta += 2.f*M_PI;

            errorU += invSamples * theta * u;

        }

        float theta = errorU.norm();
        if( theta > 1e-10 ) {
            errorU.normalize();
        }
        else
            errorU *= 0.0;

        Eigen::Quaterniond errorQ;
        errorQ.w() = cos( theta );
        errorQ.x() = sin( theta ) * errorU(0);
        errorQ.y() = sin( theta ) * errorU(1);
        errorQ.z() = sin( theta ) * errorU(2);

        meanRotation = meanRotation * errorQ;

    }
    return meanRotation;
}


void selectConvexHull( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > >& convexHull_ ) {

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
        cv::imshow( "Select Convex Hull", img_viz );

        MouseEvent mouse;
        cv::setMouseCallback( "Select Convex Hull", onMouse, &mouse );
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

                convexHull_.push_back( p.block< 3, 1 >( 0, 0 ) );
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
        std::cout << "convex hull requires more than 3 points\n";

    }
    else {

        // project selected points on common plane

        Eigen::Vector4f plane_parameters;

        // Use Least-Squares to fit the plane through all the given sample points and find out its coefficients
        EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
        Eigen::Vector4f xyz_centroid;

        pcl::PointCloud< pcl::PointXYZ >::Ptr selectedPoints( new pcl::PointCloud< pcl::PointXYZ >() );
        for( unsigned int i = 0; i < convexHull_.size(); i++ ) {
            pcl::PointXYZ p;
            p.x = convexHull_[ i ]( 0 );
            p.y = convexHull_[ i ]( 1 );
            p.z = convexHull_[ i ]( 2 );
            selectedPoints->points.push_back( p );
        }

        // Estimate the XYZ centroid
        pcl::compute3DCentroid( *selectedPoints, xyz_centroid );
        xyz_centroid[ 3 ] = 0;

        // Compute the 3x3 covariance matrix
        pcl::computeCovarianceMatrix( *selectedPoints, xyz_centroid, covariance_matrix );

        // Compute the model coefficients
        EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
        EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
        pcl::eigen33( covariance_matrix, eigen_vectors, eigen_values );

        // remove components orthogonal to the plane..
        for( unsigned int i = 0; i < convexHull_.size(); i++ ) {

            Eigen::Vector3d p = convexHull_[ i ];

            float l = p.dot( eigen_vectors.cast<double>().block< 3, 1 >( 0, 0 ) ) - xyz_centroid.cast<double>().cast<double>().block< 3, 1 >( 0, 0 ).dot( eigen_vectors.cast<double>().block< 3, 1 >( 0, 0 ) );

            p -= l * eigen_vectors.cast<double>().block< 3, 1 >( 0, 0 );

            convexHull_[ i ]( 0 ) = p( 0 );
            convexHull_[ i ]( 1 ) = p( 1 );
            convexHull_[ i ]( 2 ) = p( 2 );

        }

    }

}

void selectPlane( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > &convexHull_, Plane &table_plane ) {

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
        cv::imshow( "Select Plane", img_viz );

        MouseEvent mouse;
        cv::setMouseCallback( "Select Plane", onMouse, &mouse );
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

                convexHull_.push_back( p.block< 3, 1 >( 0, 0 ) );
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
        Eigen::Vector3d p0 = convexHull_[ 0 ];
        Eigen::Vector3d p1 = convexHull_[ 1 ];
        Eigen::Vector3d p2 = convexHull_[ 2 ];

        Eigen::Vector3d v0 = p1 - p0;
        Eigen::Vector3d v2 = p1 - p2;

        Eigen::Vector3d normal = v0.cross(v2);

        Eigen::Vector3d Y = Eigen::Vector3d(0.f,1.f,0.f);

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

void getObjectPointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, float minHeight, float maxHeight,
                          std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > convexHull, Plane &table_plane, Eigen::Vector3d turnTable_center, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& objectCloud  ) {


    // extract map and stitched point cloud from selected volume..
    // find convex hull for selected points in reference frame
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_selected_points( new pcl::PointCloud< pcl::PointXYZRGB >() );
    for( unsigned int j = 0; j < convexHull.size(); j++ ) {
        pcl::PointXYZRGB p;
        p.x = convexHull[ j ]( 0 );
        p.y = convexHull[ j ]( 1 );
        p.z = convexHull[ j ]( 2 );
        cloud_selected_points->points.push_back( p );
    }

    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_convex_hull( new pcl::PointCloud< pcl::PointXYZRGB >() );
    pcl::ConvexHull< pcl::PointXYZRGB > chull;
    chull.setInputCloud( cloud_selected_points );
    chull.reconstruct( *cloud_convex_hull );

    pcl::ExtractPolygonalPrismData< pcl::PointXYZRGB > hull_limiter;

    // get indices in convex hull
    pcl::PointIndices::Ptr object_indices( new pcl::PointIndices() );
    hull_limiter.setInputCloud( cloud );
    hull_limiter.setInputPlanarHull( cloud_convex_hull );
    hull_limiter.setHeightLimits( minHeight, maxHeight );
    //    hull_limiter.setViewPoint( transformedCloud->sensor_origin_[ 0 ], transformedCloud->sensor_origin_[ 1 ], transformedCloud->sensor_origin_[ 2 ] );
    hull_limiter.segment( *object_indices );

    std::cout << object_indices->indices.size() << "\n" << std::endl;

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    // Extract the inliers
    extract.setInputCloud(cloud);
    extract.setIndices (object_indices);
    extract.setNegative (false);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr subcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    extract.filter (*subcloud);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> subcloud_rgb(subcloud);


    if(0){
        // visualize
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor(1,1,1);
        //    viewer->addPointCloud<pcl::PointXYZRGB> (subcloud, rgb, "sample cloud");
        viewer->addPointCloud<pcl::PointXYZRGB> (subcloud, subcloud_rgb, "clipped cloud");
        viewer->addCoordinateSystem(0.1, 0.f, 0.f, 0.f, 0.f);
//        viewer->addLine( O, X, 255, 0, 0, "center" );
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
    }


    float sum  = table_plane.coefficients[0] + table_plane.coefficients[1] + table_plane.coefficients[2] + table_plane.coefficients[3];
    //    Eigen::Vector4f plane_normal(coefficients->values[0] ,coefficients->values[1] ,coefficients->values[2],coefficients->values[3]);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clipped_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> clipped_rgb(clipped_cloud);

    clipped_cloud->reserve(subcloud->size());

    for( int i = 0; i< subcloud->size(); i++){

        Eigen::Vector4d p;
        p[ 0 ] = subcloud->at( i ).x;
        p[ 1 ] = subcloud->at( i ).y;
        p[ 2 ] = subcloud->at( i ).z;
        p[ 3 ] = 1;

        float product = p.dot(table_plane.coefficients) * sum;
        if(product < -0.01f)
            clipped_cloud->push_back(subcloud->at( i ));

    }

    Eigen::Vector3d normal = table_plane.coefficients.block< 3, 1 >( 0, 0 );
    Eigen::Vector3d temp = normal + turnTable_center;

    // add line
    pcl::PointXYZRGB O, X;
    O.x = turnTable_center[0];
    O.y = turnTable_center[1];
    O.z = turnTable_center[2];

    X.x = temp[0];
    X.y = temp[1];
    X.z = temp[2];

    objectCloud = clipped_cloud;

    if(0){
        // visualize
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor(1,1,1);
        //    viewer->addPointCloud<pcl::PointXYZRGB> (subcloud, rgb, "sample cloud");
        viewer->addPointCloud<pcl::PointXYZRGB> (clipped_cloud, clipped_rgb, "clipped cloud");
        viewer->addCoordinateSystem(0.1, 0.f, 0.f, 0.f, 0.f);
//        viewer->addLine( O, X, 255, 0, 0, "center" );
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
    }

}

Eigen::Vector3d getTurnTableCenter( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, Eigen::Matrix4d& referenceTransform, Plane &table_plane ) {

    // let user select convex hull points in the images
    std::cout << "select convex hull in the image\n";
    std::cout << "left click: add point, right click: finish selection\n";

    cv::Mat img_cv;
    cv::cvtColor( img_rgb, img_cv, cv::COLOR_BGR2GRAY );

    std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > turnTable;
    std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > turnTable_proj;

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
        cv::imshow( "Select turn table", img_viz );

        MouseEvent mouse;
        cv::setMouseCallback( "Select turn table", onMouse, &mouse );
        cv::waitKey( 10 );

        if( mouse.event == CV_EVENT_LBUTTONDOWN ) {

            // find corresponding 3D position in point cloud
            float img2cloudScale = ( (float) cloud->height ) / ( (float) displayHeight );
            unsigned int idx = round( img2cloudScale * ( (float) mouse.pt.y ) ) * cloud->width + round( img2cloudScale * ( (float) mouse.pt.x ) );
            if( idx < cloud->points.size() && !isnan( cloud->points[ idx ].x ) ) {

                //  transform point to reference frame

                Eigen::Vector3d p, p_proj;
                p[ 0 ] = cloud->points[ idx ].x;
                p[ 1 ] = cloud->points[ idx ].y;
                p[ 2 ] = cloud->points[ idx ].z;

                //                p = ( referenceTransform * p ).eval();

                // project on plane
                Eigen::Vector3d normal = table_plane.coefficients.block< 3, 1 >( 0, 0 );
                p_proj = p - normal.dot( p - table_plane.point ) * normal;

                turnTable_proj.push_back(p_proj);
                turnTable.push_back( p.block< 3, 1 >( 0, 0 ) );
            }

        }
        else if( mouse.event == CV_EVENT_RBUTTONDOWN ) {
            stopSelection = true;
        }
        else if( mouse.event == CV_EVENT_MBUTTONDOWN ) {
            stopSelection = true;
        }
    }

    Eigen::Vector3d center;

    if( turnTable.size() < 3 ) {
        std::cout << "circle requires more than 3 points\n";
        return center;
    }
    else {

        // calcualte center of peri circle
        Eigen::Vector3d normal = table_plane.coefficients.block< 3, 1 >( 0, 0 );

        // mid points
        Eigen::Vector3d m1,m2;
        m1 = ( turnTable_proj[0] + turnTable_proj[1] ) / 2.f ;
        m2 = ( turnTable_proj[1] + turnTable_proj[2] ) / 2.f;

        // bisectors
        Eigen::Vector3d b1, b2;
        b1 = normal.cross( turnTable_proj[0] - turnTable_proj[1] );
        b2 = normal.cross( turnTable_proj[1] - turnTable_proj[2] );

        b1.normalize();
        b2.normalize();

        Eigen::Vector2d v = (m1-m2).block<2,1>(0,0);
        Eigen::Matrix2d A;
        A.block<2,1>(0,0) = b2.block<2,1>(0,0);
        A.block<2,1>(0,1) = -b1.block<2,1>(0,0);

        Eigen::Vector2d lambda;
        lambda = A.inverse()*v;

        center = m2 + lambda[1]*b2;

    }

    return center;

}

void printScore(cv::Mat &img, string &objectName, float score, cv::Point2f &pt, bool print_score ){

    // Text variables

    string ntext = objectName;
    stringstream ss;
    ss << score;
    string stext = ss.str();

    double scale = 1.5f;
    cv::putText(img, ntext.c_str(), pt, 1, scale, CV_RGB(255, 0, 100), 2);
    if(print_score)
        cv::putText(img, stext.c_str(), cv::Point(pt.x, pt.y + 50), 1, scale, CV_RGB(255, 0, 255), 2);

}

void get3DBoundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,  std::vector< Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >& transformationMatrixOC, std::vector< cv::Point3f> &cg, cv::Point3f &bbSize){

    // downsampling using voxel grid
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.001f, 0.001f, 0.001f); // 1mm x 1mm x 1mm voxels
    sor.filter (*cloud_filtered);

    // remove noise
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_cluster_rgb(cloud_cluster);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
    sor1.setInputCloud (cloud_filtered);
    sor1.setMeanK (200);
    sor1.setStddevMulThresh (1.0);
    sor1.filter (*cloud_cluster);


    // find bounding box surrounded
    pcl::PointXYZRGB  min_pt, max_pt;
    pcl::getMinMax3D (*cloud_cluster, min_pt, max_pt);

    // debug
    if(0){

        // draw bounding box
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor( 1, 1, 1 );
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud_cluster, cloud_cluster_rgb, "clipped cloud");
        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );
        viewer->addCube( min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z, 1.f, 0.f,0.f);
        while (!viewer->wasStopped ()){

            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }

    }

    // save it to bounding box in object coordinate system
    BoundingBoxXYZ BB3D;
    BB3D.x = min_pt.x;
    BB3D.y = min_pt.y;
    BB3D.z = min_pt.z;
    BB3D.width = max_pt.x - min_pt.x;
    BB3D.height = max_pt.y - min_pt.y;
    BB3D.depth= max_pt.z - min_pt.z;

    // get center of object in object coordinate system
    cv::Point3f temp = BB3D.getCenter();
    pcl::PointXYZRGB cgO;
    cgO.x =  temp.x;
    cgO.y =  temp.y;
    cgO.z =  temp.z;

    bbSize = BB3D.getDimention(); // bounding box dimension calculated once and for all

    // calculate center in camera coordinate system with 3 different pitch angle
    std::vector< pcl::PointXYZRGB > cg_(3);
    cg_[0] = pcl::transformPoint(cgO, Eigen::Affine3f(transformationMatrixOC[0].cast<float>()) );
    cg_[1] = pcl::transformPoint(cgO, Eigen::Affine3f(transformationMatrixOC[1].cast<float>()) );
    cg_[2] = pcl::transformPoint(cgO, Eigen::Affine3f(transformationMatrixOC[2].cast<float>()) );

    cg[0] = cv::Point3f(cg_[0].x, cg_[0].y, cg_[0].z);
    cg[1] = cv::Point3f(cg_[1].x, cg_[1].y, cg_[1].z);
    cg[2] = cv::Point3f(cg_[2].x, cg_[2].y, cg_[2].z);

    //    Eigen::Vector4f cg_vector0(cg_[0].x, cg_[0].y, cg_[0].z, 1.f);
    //    transformationMatrixOC[0].block< 4,1 >( 0, 3 ) = cg_vector0;

    //    Eigen::Vector4f cg_vector1(cg_[1].x, cg_[1].y, cg_[1].z, 1.f);
    //    transformationMatrixOC[1].block< 4,1 >( 0, 3 ) = cg_vector1;

    //    Eigen::Vector4f cg_vector2(cg_[2].x, cg_[2].y, cg_[2].z, 1.f);
    //    transformationMatrixOC[2].block< 4,1 >( 0, 3 ) = cg_vector2;


    //    // once again
    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
    //    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_cluster_transformed_rgb(cloud_cluster);

    //    Eigen::Matrix4d transformationMatrixOC_inverse = transformationMatrixOC[0].inverse();
    //    pcl::transformPointCloud (*cloud_cluster, *cloud_cluster_transformed, transformationMatrixOC_inverse );


    //    pcl::getMinMax3D (*cloud_cluster_transformed, min_pt, max_pt);

    //    // debug
    //    if(1){

    //        // draw bounding box
    //        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //        viewer->setBackgroundColor( 1, 1, 1 );
    //        viewer->addPointCloud<pcl::PointXYZRGB> (cloud_cluster_transformed, cloud_cluster_transformed_rgb, "clipped cloud");
    //        viewer->addCoordinateSystem( 0.1, 0.f, 0.f, 0.f, 0.f );
    //        viewer->addCube( min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z );
    //        while (!viewer->wasStopped ()){

    //            viewer->spinOnce (100);
    //            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    //        }

    //    }

}

void create3DBB(cv::Point3f &bbSize, Eigen::Matrix4d &transformationMatrixOC , cv::Size2f &img_size, std::vector< cv::Point2f > &imagePoints){

    float w = bbSize.x/2.f;
    float h = bbSize.y/2.f;
    float d = bbSize.z/2.f;

    std::vector<cv::Point3f> vertices(8);

    // in Object coordinates system
    pcl::PointXYZ point_0_min( -w, -h, -d);

    pcl::PointXYZ point_1_w( w, -h, -d);
    pcl::PointXYZ point_1_h( -w, h, -d);
    pcl::PointXYZ point_1_d( -w, -h, d);

    pcl::PointXYZ point_2_wh( w, h, -d);
    pcl::PointXYZ point_2_wd( w, -h, d);
    pcl::PointXYZ point_2_dh( -w, h, d);

    pcl::PointXYZ point_3_max( w, h, d);


    // transformed into camera coordinates
    pcl::PointXYZ point_min_TC   = pcl::transformPoint(point_0_min, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[0] = cv::Point3f(point_min_TC.x, point_min_TC.y, point_min_TC.z);

    pcl::PointXYZ point_1_w_TC   = pcl::transformPoint(point_1_w, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[1] = cv::Point3f(point_1_w_TC.x, point_1_w_TC.y, point_1_w_TC.z);

    pcl::PointXYZ point_1_h_TC   = pcl::transformPoint(point_1_h, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[2] = cv::Point3f(point_1_h_TC.x, point_1_h_TC.y, point_1_h_TC.z);

    pcl::PointXYZ point_1_d_TC   = pcl::transformPoint(point_1_d, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[3] = cv::Point3f(point_1_d_TC.x, point_1_d_TC.y, point_1_d_TC.z);

    pcl::PointXYZ point_2_wh_TC  = pcl::transformPoint(point_2_wh, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[4] = cv::Point3f(point_2_wh_TC.x, point_2_wh_TC.y, point_2_wh_TC.z);

    pcl::PointXYZ point_2_wd_TC  = pcl::transformPoint(point_2_wd, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[5] = cv::Point3f(point_2_wd_TC.x, point_2_wd_TC.y, point_2_wd_TC.z);

    pcl::PointXYZ point_2_dh_TC  = pcl::transformPoint(point_2_dh, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[6] = cv::Point3f(point_2_dh_TC.x, point_2_dh_TC.y, point_2_dh_TC.z);

    pcl::PointXYZ point_max_TC   = pcl::transformPoint(point_3_max, Eigen::Affine3f(transformationMatrixOC.cast<float>()) );
    vertices[7] = cv::Point3f(point_max_TC.x, point_max_TC.y, point_max_TC.z);


    // Project points to 2D
    cv::Mat cameraMatrix, distortionCoeffs;

    cameraMatrix = cv::Mat::zeros( 3, 3, CV_32FC1);
    cameraMatrix.at<float>(0,0) = 525.f; //fx
    cameraMatrix.at<float>(1,1) = 525.f; //fy
    cameraMatrix.at<float>(0,2) = img_size.width / 2.f; // cx
    cameraMatrix.at<float>(1,2) = img_size.height / 2.f; // cy
    cameraMatrix.at<float>(2,2) = 1;

    distortionCoeffs = cv::Mat::zeros( 1, 4, CV_32FC1);

    imagePoints.resize( vertices.size() );
    cv::Mat rot( 3, 1, CV_64FC1, 0.f );
    cv::Mat trans( 3, 1, CV_64FC1, 0.f );
    cv::projectPoints( vertices, rot, trans, cameraMatrix, distortionCoeffs, imagePoints );

}

void createWireFrame (cv::Mat &img, std::vector<cv::Point2f> &vertices){

    // draw lines on image
    cv::line(img, vertices[0], vertices[1], CV_RGB(255, 0, 0), 2, 8, 0);
    cv::line(img, vertices[0], vertices[2], CV_RGB(0, 255, 0), 2, 8, 0);
    cv::line(img, vertices[0], vertices[3], CV_RGB(0, 0, 255), 2, 8, 0);

    cv::line(img, vertices[1], vertices[4], CV_RGB(0, 255, 0), 2, 8, 0);
    cv::line(img, vertices[1], vertices[5], CV_RGB(0, 0, 255), 2, 8, 0);

    cv::line(img, vertices[2], vertices[4], CV_RGB(255, 0, 0), 2, 8, 0);
    cv::line(img, vertices[2], vertices[6], CV_RGB(0, 0, 255), 2, 8, 0);

    cv::line(img, vertices[3], vertices[5], CV_RGB(255, 0, 0), 2, 8, 0);
    cv::line(img, vertices[3], vertices[6], CV_RGB(0, 255, 0), 2, 8, 0);

    cv::line(img, vertices[4], vertices[7], CV_RGB(0, 0, 255), 2, 8, 0);
    cv::line(img, vertices[5], vertices[7], CV_RGB(0, 255, 0), 2, 8, 0);
    cv::line(img, vertices[6], vertices[7], CV_RGB(255, 0, 0), 2, 8, 0);

}

void getLine( pcl::PointXYZRGB &p, Line &line){

    line.point = Eigen::Vector3d(0.f, 0.f, 0.f);
    line.direction = Eigen::Vector3d(p.x,p.y,p.z) - line.point;
    line.direction.normalize();
}

void getLinePlaneIntersection(Line &line, Plane &plane, Eigen::Vector3d &ptIntersection){

    Eigen::Vector3d Po = plane.point;
    Eigen::Vector3d Lo = line.point;
    Eigen::Vector3d n  = plane.getNormal();
    Eigen::Vector3d l = line.direction;
    float d;

    d = ( (Po - Lo).dot( n ) ) / ( l.dot( n ) );

    // point of intersection can be found by subtituing d into the eqaution of line
    ptIntersection = d*l + Lo;

}



