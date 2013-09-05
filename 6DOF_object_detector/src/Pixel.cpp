/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
//
// Modified: Nima Razavi, BIWI, ETH Zurich
//
*/

// Modified by: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#include "Pixel.h"
#include <deque>

using namespace std;

void CRPixel::extractPixels(const Parameters& param, const cv::Mat& img, const cv::Mat& depthImg, const cv::Mat& maskImg, unsigned int n, int label, int imageID, CvRect* box, CvPoint* vCenter, cv::Point3f *cg, cv::Point3f *bbSize3D, Eigen::Matrix4f *transformationMatrixOC){

    // take a subset of image containing object in it using bounding box

    // extract features
    vector<cv::Mat> vImg;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    extractFeatureChannels(param, img, depthImg, vImg, normals);

    // debug for depth image

    if(0){
        cv::Mat temp(depthImg.rows, depthImg.cols,CV_32F);
        double minVal, maxVal;
        cv::minMaxLoc(depthImg, &minVal, &maxVal, 0, 0);
        for(int y =0; y<depthImg.rows; y++)
            for(int x=0; x<depthImg.cols; x++)
                temp.at<float>(y,x) = depthImg.at<unsigned short>(y,x)/maxVal;

        cv::imshow("depthimage", temp);
        cv::waitKey(0);

    }

    // see bounding box
    if( 0 ){
        cv::Mat img_1;
        img.copyTo(img_1);
        cv::rectangle(img_1, cv::Rect(*box), CV_RGB(255, 0, 0), 3);
        cv::imshow("iChannel", img_1);
        cv::waitKey(0);
    }

    //debug feature channels
    if( 0 ){
        for( unsigned int c = 0; c < vImg.size(); ++c ){
            imshow( "iChannel", vImg[ c ] );
            cv::waitKey( 0 );
        }
    }

    cv::Point2f imgCenter(img.cols/2.f,img.rows/2.f);

    // generate x,y locations

    //    float scale_factor = 0; // also defined in generateTest function in CRTree.cpp

    //    int sub_width = box->width * scale_factor;
    //    int sub_height = box->height * scale_factor;

    //    cv::Mat locations = cv::Mat( (box->width - sub_width) * (box->height-sub_height), 2, CV_32SC2 );
    //    cvRNG->fill(locations, CV_RAND_UNI, cv::Scalar( box->x + sub_width/2 , box->y + sub_height/2 ,0 ,0),cv::Scalar(box->x+box->width - sub_width/2, box->y+box->height - sub_height/2,0,0));
    ////    cvRNG->fill(locations, CV_RAND_UNI, cv::Point2f( box->x + sub_width/2 , box->y + sub_height/2 ), cv::Point2f(box->x+box->width - sub_width/2, box->y+box->height - sub_height/2));

    //    //    cv::Mat locations = cv::Mat( (box->width)*(box->height), 2, CV_32SC2 );
    //    //    cvRNG->fill(locations,CV_RAND_UNI, cv::Scalar( box->x, box->y,0,0),cv::Scalar(box->x+box->width,box->y+box->height,0,0));

    std::vector<cv::Point2f>locations;
    locations.reserve(n);
    int stepsize = static_cast<int>(std::sqrt(box->width*box->height/n)) + 1;

    // sample pixels uniformly from bounding box
    for( unsigned int y = box->y; y < box->height + box->y; y+=stepsize ){
        for( unsigned int x = box->x; x < box->width + box->x; x+=stepsize ){
            if( maskImg.at<uchar>(y,x) != 0 )
                locations.push_back(cv::Point(x,y));
        }
    }
    n = locations.size();
    // reserve memory
    unsigned int offset = vRPixels[label].size();
    vRPixels[label].reserve(offset+n);
    vImageIDs[label].reserve(offset+n);

    // save pixel features
    for( unsigned int i = 0; i < n ; i++ ) {

        cv::Point2f pt(locations[i]);

        PixelFeature* pf = new PixelFeature;

        vImageIDs[label].push_back(imageID); // assigning the image id to the Pixel

        pf->statFeature->pixelLocation = pt; // saving pixel location
        pf->statFeature->iWidth = img.cols; // image size
        pf->statFeature->iHeight = img.rows;
        pf->statFeature->bbox = *box; // saving Bounding box details

        float scalePt;
        if(depthImg.at<unsigned short>(pt) == 0)
            scalePt = FLT_MAX;
        else
            scalePt = (float)(1000.f/depthImg.at<unsigned short>(pt)); //scale is inverse of depth::depth value sampled pixel is in millimeter converted to meter
        pf->statFeature->scale = scalePt;

        if( bbSize3D != 0 )
            pf->statFeature->bbSize3D = *bbSize3D;
        else
            pf->statFeature->bbSize3D = cv::Point3f (0.f,0.f,0.f);


        // save all the information below for object class only
        if(vCenter!=0) {

            cv::Point2f objCenter(*vCenter);
            cv::Point3f rObjCenter;
            float scaleObjCenter;
            if(depthImg.at<unsigned short>(objCenter) == 0)
                scaleObjCenter =  FLT_MAX;
            else
                scaleObjCenter = (float)(1000.f/(depthImg.at<unsigned short>(objCenter)));
            rObjCenter = *cg;

            cv::Point3f rPt = P3toR3( pt, imgCenter, 1/scalePt );

            pf->statFeature->pixelLocation_real = rPt;
            pf->statFeature->disVector = rPt - rObjCenter;

            pf->statFeature->imgAppearance.resize(vImg.size());
            pf->statFeature->imgAppearance = vImg; //assign pointer to feature channel will it be released?? Is it memory efficient??;
            pf->statFeature->normals = normals;
            pf->statFeature->transformationMatrixOC = *transformationMatrixOC;

            if(0)
                drawTransformation(img, depthImg, *transformationMatrixOC);

            // visualize 3D bounding box
            if(0){

                cv::Mat img_show;
                img.copyTo(img_show);
                cv::Size2f img_size(img.cols, img.rows);
                std::vector<cv::Point2f> imagePoints;
                create3DBB(*bbSize3D, *transformationMatrixOC, img_size, imagePoints);
                createWireFrame(img_show,imagePoints);
                cv::circle(img_show, objCenter, 2, CV_RGB(255, 0, 255), 2, 8, 0);
                cv::imshow("img", img_show); cv::waitKey(0);
            }

        }

        vRPixels[label].push_back(pf);
        if (pf->statFeature->iHeight!=480)
            std::cout<<".................errrorrr............."<<endl;

        // debug visualize randomly generated pixels
        if(0){
            cv::Mat img_show;
            img.copyTo(img_show);
            cv::circle(img_show, pt, 1, CV_RGB( 255, 0, 0 ), 8, 8, 0);
            cv::circle(img_show, *vCenter, 1, CV_RGB( 0, 255, 0 ), 8, 8, 0);
            cv::line(img_show, pt, *vCenter, CV_RGB(255,0,255), 2, 8, 0);
            cv::imshow("img", img_show); cv::waitKey(0);
        }

        // debug visualize assignment of channel pointers
        if(0){
            for(unsigned int c=0; c<vImg.size(); ++c){
                cv::imshow("debug",pf->statFeature->imgAppearance[c]);
                cv::waitKey(0);
            }
        }

    }
}

void CRPixel::extractFeatureChannels(const Parameters& param, const cv::Mat& img, const cv::Mat& depthImg, std::vector<cv::Mat>& vImg, pcl::PointCloud<pcl::Normal>::Ptr& normals) {

    // 34 feature channels
    // 7 + 1: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy| + depth (currently using)
    // + 9 channels	HOGlike features with 9 bins (weighted orientations 5x5 neighborhood) (currently not using)
    // 17+17 channels: minfilter + maxfilter on 5x5 neighborhood (currently not using)

    int total_channels = 8;
    if( param.addHoG )
        total_channels += 9 ;
    if(param.addMinMaxFilt)
        total_channels *= 2;

    // currently we are using the first 7 raw feature channel
    vImg.resize(total_channels);
    for( unsigned int c = 0; c < total_channels; ++c ){
        vImg[ c ] = cv::Mat::zeros( img.rows, img.cols, CV_8UC1 );

    }

    // Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
    cv::Mat I_x = cv::Mat( img.rows, img.cols, CV_32FC1 );
    cv::Mat I_y = cv::Mat( img.rows, img.cols, CV_32FC1 );

    // Get intensity
    cv::cvtColor( img, vImg[ 0 ], CV_BGR2GRAY );

    // |I_x|, |I_y|
    cv::Sobel( vImg[ 0 ], I_x, CV_32F, 1, 0, 3 );
    cv::convertScaleAbs( I_x, vImg[ 3 ], 0.25 );

    cv::Sobel( vImg[ 0 ], I_y, CV_32F, 0, 1, 3 );
    cv::convertScaleAbs( I_y, vImg[ 4 ], 0.25 );

    // |I_xx|, |I_yy|

    cv::Mat I_xx = cv::Mat( img.rows, img.cols, CV_32FC1 );
    cv::Mat I_yy = cv::Mat( img.rows, img.cols, CV_32FC1 );

    cv::Sobel( vImg[ 0 ], I_xx, CV_32F, 2, 0, 3 );
    cv::convertScaleAbs( I_xx, vImg[5], 0.25);

    cv::Sobel( vImg[ 0 ], I_yy, CV_32F, 0, 2, 3 );
    cv::convertScaleAbs( I_yy, vImg[ 6 ], 0.25 );

    // L, a, b
    cv::Mat temp;
    cv::cvtColor(img, temp, CV_BGR2Lab  );

    std::vector < cv::Mat > Lab;
    Lab.resize( 3 );
    for( unsigned int t = 0; t < Lab.size(); t++ )
        Lab[ t ] = cv::Mat(temp.rows, temp.cols, CV_8UC1);
    cv::split(temp, Lab);
    Lab[ 0 ].copyTo( vImg[ 0 ]); Lab[ 1 ].copyTo( vImg[ 1 ]); Lab[ 2 ].copyTo(vImg[ 2 ] );

    depthImg.copyTo( vImg[ 7 ] );

    // Compute Normals
    computeNormals(img, depthImg, normals );

    if(param.addHoG){

        // generate magnitude and orientation images out of I_x, I_y

        cv::Mat Iorient = cv::Mat(I_x.rows, I_x.cols, CV_32FC1);
        cv::Mat Imag = cv::Mat(I_x.rows, I_x.cols, CV_32FC1);

        // direction
        for(int r = 0; r < Iorient.rows; r++ ){
            for(int c = 0; c < Iorient.cols; c++){

                float tx = I_x.at<float>(r,c) + (float)_copysign(0.000001f, I_x.at<float>(r,c));
                // Scaling [-pi/2 pi/2] -> [0 pi]
                float val =  (atan( I_y.at< float >( r, c ) / tx) ) * 180.f /PI + 90.f;
                Iorient.at< float >(r,c) = val;
            }
        }

        // magnitude
        cv::Mat sum = cv::Mat( I_x.rows, I_x.cols, CV_32FC1 );
        cv::Mat prodX = cv::Mat( I_x.rows, I_x.cols, CV_32FC1 );
        cv::Mat prodY = cv::Mat( I_x.rows, I_x.cols, CV_32FC1 );

        cv::multiply(I_x, I_x, prodX);
        cv::multiply(I_y, I_y, prodY);
        sum = prodX + prodY;
        cv::sqrt(sum, Imag);

        HoG hog;
        hog.extractOBin(Iorient, Imag, depthImg, vImg, 8);

    }

    if(param.addMinMaxFilt){

        int ksize = 5;

        minfilt( vImg, depthImg, param.scales, ksize );
        maxfilt( vImg, depthImg, param.scales, ksize );
    }

}

void CRPixel::computeNormals(const cv::Mat& img, const cv::Mat& depthImg, pcl::PointCloud<pcl::Normal>::Ptr& normals  ){

    // Initialize the cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);

    // Populate the cloud
    Surfel::imagesToPointCloud( depthImg, img, cloud);

    // Compute Normals
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;

    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR );
    ne.setNormalSmoothingSize(20.0f );
    ne.setRectSize( 21, 21 );
    ne.setInputCloud(cloud);

    ne.compute(*normals);

    // debug visualize
    if(0){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10);
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
    }
}

void CRPixel::calcObject2CameraTransformation( float &pose, float &pitch, cv::Point3f &rObjCenter, Eigen::Matrix4f& transformationMatrixOC){

    Eigen::Affine3f Rx,T, Rz;

    pcl::getTransformation(rObjCenter.x , rObjCenter.y, rObjCenter.z, 0.f, 0.f, 0.f, T);
    pcl::getTransformation(0.f,0.f,0.f, pcl::deg2rad( 90.f + pitch), 0.f, 0.f, Rx);
    pcl::getTransformation(0.f, 0.f, 0.f, 0.f, 0.f, pcl::deg2rad(-pose), Rz);

    transformationMatrixOC = T.matrix() * Rx.matrix() * Rz.matrix();
}

void CRPixel::drawTransformation(const cv::Mat& img, const cv::Mat& depthImg , const Eigen::Matrix4f &m_transformationMatrixOC){

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
    Surfel::imagesToPointCloud( depthImg, img, cloud);

    /* Transform point cloud */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    pcl::transformPointCloud(*cloud, *cloudTO, transformationMatrixOC.inverse());
    pcl::transformPointCloud(*cloudTO, *cloudTC, transformationMatrixOC);


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

cv::Point3f CRPixel::P3toR3(cv::Point2f &pixelCoordinates, cv::Point2f &center, float depth){

    float focal_length = 525.f; // in pixel
    cv::Point3f realCoordinates;
    realCoordinates.z = depth; // in meter
    realCoordinates.x = (pixelCoordinates.x - center.x)* depth / focal_length;
    realCoordinates.y = (pixelCoordinates.y - center.y)* depth / focal_length;
    return realCoordinates;
}

void CRPixel::R3toP3(cv::Point3f &realCoordinates, cv::Point2f &center, cv::Point2f &pixelCoordinates, float &depth){
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

Eigen::Matrix4f CRPixel::getTransformationAtQueryPixel( Eigen::Matrix3f &rotationMatrixOQ, Eigen::Matrix4f transformationMatrixOC,  cv::Point3f &pointLocation){

    Eigen::Matrix4f transformationMatrixQC = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f rotaitonMatrixOC = transformationMatrixOC.matrix().block<3,3>(0,0);

    Eigen::Matrix3f rotationMatrixQC = rotaitonMatrixOC * rotationMatrixOQ.inverse();
    transformationMatrixQC.block<3,3>(0,0) =rotationMatrixQC;

    Eigen::Vector3f translation(pointLocation.x, pointLocation.y, pointLocation.z);
    transformationMatrixQC.block<3,1>(0,3 ) = translation;

    return transformationMatrixQC;
}

void CRPixel::minfilt( std::vector< cv::Mat >& src, const cv::Mat& depthImg, const std::vector<float>scales, unsigned int kSize ){

    // for all scale generate binary images
    std::vector< cv::Mat> comp(scales.size());
    for( int scNr = 0; scNr < scales.size(); scNr++ ){
        int prevScale = scNr - 1;
        int minLimit, maxLimit;

        if( scNr == scales.size() -1 )
            minLimit = 0;
        else
            minLimit = 1000 / scales[ scNr ];
        if( prevScale < 0 )
            maxLimit = INT_MAX;
        else
            maxLimit = 1000 / scales[ prevScale ];

        cv::Mat compared_min, compared_max, binary;
        cv::compare(depthImg, minLimit, compared_min, CV_CMP_GE);
        cv::compare(depthImg, maxLimit, compared_max, CV_CMP_LT);
        cv::multiply(compared_min, compared_max, binary);

        comp[scNr] = binary;
    }


    for(int i = 0; i < src.size()/2; i++){ // for every channel

        // erode at different scales using different kernel window size
        cv::Mat sum = cv::Mat::zeros(src[i].rows, src[i].cols , src[i].depth());
        cv::Mat tmp; src[i].copyTo(tmp);
        for( int scNr = 0; scNr < scales.size(); scNr++ ){

            int adapKsize = int(kSize * scales[scNr]) + (int(kSize*scales[scNr]) % 2 == 0 );
            cv::Mat kernel = cv::Mat::ones(adapKsize, adapKsize, CV_8UC1);

            cv::erode(src[i], tmp, kernel);
            cv::add(sum, tmp, sum, comp[scNr], src[i].depth());
        }

        src[ src.size() / 2 + i ] = sum;
    }

}

void CRPixel::maxfilt( std::vector< cv::Mat >& src, const cv::Mat& depthImg, const std::vector<float>scales, unsigned int kSize ){

    // for all scale generate binary images
    std::vector< cv::Mat> comp(scales.size());
    for( int scNr = 0; scNr < scales.size(); scNr++ ){
        int prevScale = scNr - 1;
        int minLimit, maxLimit;

        if(scNr == scales.size() -1 )
            minLimit = 0;
        else
            minLimit = 1000 / scales[scNr];
        if(prevScale < 0)
            maxLimit = INT_MAX;
        else
            maxLimit = 1000 / scales[ prevScale];

        cv::Mat compared_min, compared_max, binary;
        cv::compare(depthImg, minLimit, compared_min, CV_CMP_GE);
        cv::compare(depthImg, maxLimit, compared_max, CV_CMP_LT);
        cv::multiply(compared_min, compared_max, binary);

        comp[scNr] = binary;
    }


    for(int i = 0; i < src.size()/2; i++){ // for every channel

        // erode at different scales using different kernel window size
        cv::Mat sum = cv::Mat::zeros(src[i].rows, src[i].cols , src[i].depth());
        cv::Mat tmp; src[i].copyTo(tmp);
        for( int scNr = 0; scNr < scales.size(); scNr++ ){

            int adapKsize = int(kSize * scales[scNr]) + (int(kSize*scales[scNr]) % 2 == 0 );
            cv::Mat kernel = cv::Mat::ones(adapKsize, adapKsize, CV_8UC1);

            cv::dilate(src[i], tmp, kernel);
            cv::add(sum, tmp, sum, comp[scNr], src[i].depth());
        }

        src[ i ] = sum;

    }

}

void CRPixel::maxfilt( cv::Mat src, unsigned int width ) {

    uchar* s_data = src.datastart;
    int step = src.step;
    CvSize size = cvSize(src.cols,src.rows);

    for(int  y = 0; y < size.height; y++) {

        maxfilt(s_data+y*step, 1, size.width, width);
    }

    uchar* s1_data = src.datastart;

    for(int  x = 0; x < size.width; x++)
        maxfilt( s1_data+x, step, size.height, width);

}

void CRPixel::minfilt(cv::Mat src, cv::Mat dst, unsigned int width) {

    uchar* s_data = src.datastart;
    int step = src.step;
    CvSize size = cvSize(src.cols,src.rows);

    uchar* d_data = dst.datastart;

    for(int  y = 0; y < size.height; y++)
        minfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

    uchar* s1_data = src.datastart;
    for(int  x = 0; x < size.width; x++)
        minfilt(s1_data+x, step, size.height, width);

}

void CRPixel::maxfilt( uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width ) {

    unsigned int d = int(( width + 1 ) / 2 ) * step;
    size *= step;
    width *= step;

    maxvalues[ 0 ] = data[ 0 ];
    for( unsigned int i = 0; i < d - step; i += step ) {
        for( unsigned int k = i; k < d + i; k += step ) {
            if( data[ k ] > maxvalues[ i ] ) maxvalues[ i ] = data[ k ];
        }
        maxvalues[ i + step ] = maxvalues[ i ];
    }

    maxvalues[ size - step ] = data[ size - step ];
    for( unsigned int i = size - step; i > size - d; i -= step ) {
        for( unsigned int k = i; k > i - d; k -= step ) {
            if( data[ k ] > maxvalues[ i ] ) maxvalues[ i ] = data[ k ];
        }
        maxvalues[ i - step ] = maxvalues[ i ];
    }

    deque< int > maxfifo;
    for( unsigned int i = step; i < size; i += step ) {
        if( i >= width ) {
            maxvalues[ i - d ] = data[ maxfifo.size() > 0 ? maxfifo.front(): i - step ];
        }

        if( data[ i ] < data[ i - step ] ) {

            maxfifo.push_back( i - step );
            if( i ==  width + maxfifo.front() )
                maxfifo.pop_front();

        } else {

            while(maxfifo.size() > 0) {
                if(data[i] <= data[maxfifo.back()]) {
                    if(i==  width+maxfifo.front())
                        maxfifo.pop_front();
                    break;
                }
                maxfifo.pop_back();
            }

        }

    }

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];

}

void CRPixel::maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    deque<uchar> tmp;

    tmp.push_back(data[0]);
    for(unsigned int k=step; k<d; k+=step) {
        if(data[k]>tmp.back()) tmp.back() = data[k];
    }

    for(unsigned int i=step; i < d-step; i+=step) {
        tmp.push_back(tmp.back());
        if(data[i+d-step]>tmp.back()) tmp.back() = data[i+d-step];
    }


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
            data[i-width] = tmp.front();
            tmp.pop_front();
        }

        if(data[i] < data[i-step]) {

            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
                minfifo.pop_front();

        } else {

            while(minfifo.size() > 0) {
                if(data[i] <= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                        minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }

        }

    }

    tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);

    for(unsigned int k=size-step-step; k>=size-d; k-=step) {
        if(data[k]>data[size-step]) data[size-step] = data[k];
    }

    for(unsigned int i=size-step-step; i >= size-d; i-=step) {
        data[i] = data[i+step];
        if(data[i-d+step]>data[i]) data[i] = data[i-d+step];
    }

    for(unsigned int i=size-width; i<=size-d; i+=step) {
        data[i] = tmp.front();
        tmp.pop_front();
    }

}

void CRPixel::minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    minvalues[0] = data[0];
    for(unsigned int i=0; i < d-step; i+=step) {
        for(unsigned int k=i; k<d+i; k+=step) {
            if(data[k]<minvalues[i]) minvalues[i] = data[k];
        }
        minvalues[i+step] = minvalues[i];
    }

    minvalues[size-step] = data[size-step];
    for(unsigned int i=size-step; i > size-d; i-=step) {
        for(unsigned int k=i; k>i-d; k-=step) {
            if(data[k]<minvalues[i]) minvalues[i] = data[k];
        }
        minvalues[i-step] = minvalues[i];
    }

    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
        }

        if(data[i] > data[i-step]) {

            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
                minfifo.pop_front();

        } else {

            while(minfifo.size() > 0) {
                if(data[i] >= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                        minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }

        }

    }

    minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];

}

void CRPixel::minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    deque<uchar> tmp;

    tmp.push_back(data[0]);
    for(unsigned int k=step; k<d; k+=step) {
        if(data[k]<tmp.back()) tmp.back() = data[k];
    }

    for(unsigned int i=step; i < d-step; i+=step) {
        tmp.push_back(tmp.back());
        if(data[i+d-step]<tmp.back()) tmp.back() = data[i+d-step];
    }


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
            data[i-width] = tmp.front();
            tmp.pop_front();
        }

        if(data[i] > data[i-step]) {

            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
                minfifo.pop_front();

        } else {

            while(minfifo.size() > 0) {
                if(data[i] >= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                        minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }

        }

    }

    tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);

    for(unsigned int k=size-step-step; k>=size-d; k-=step) {
        if(data[k]<data[size-step]) data[size-step] = data[k];
    }

    for(unsigned int i=size-step-step; i >= size-d; i-=step) {
        data[i] = data[i+step];
        if(data[i-d+step]<data[i]) data[i] = data[i-d+step];
    }

    for(unsigned int i=size-width; i<=size-d; i+=step) {
        data[i] = tmp.front();
        tmp.pop_front();
    }
}

void CRPixel::maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    maxvalues[0] = data[0];
    minvalues[0] = data[0];
    for(unsigned int i=0; i < d-step; i+=step) {
        for(unsigned int k=i; k<d+i; k+=step) {
            if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
            if(data[k]<minvalues[i]) minvalues[i] = data[k];
        }
        maxvalues[i+step] = maxvalues[i];
        minvalues[i+step] = minvalues[i];
    }

    maxvalues[size-step] = data[size-step];
    minvalues[size-step] = data[size-step];
    for(unsigned int i=size-step; i > size-d; i-=step) {
        for(unsigned int k=i; k>i-d; k-=step) {
            if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
            if(data[k]<minvalues[i]) minvalues[i] = data[k];
        }
        maxvalues[i-step] = maxvalues[i];
        minvalues[i-step] = minvalues[i];
    }

    deque<int> maxfifo, minfifo;

    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
            minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
        }

        if(data[i] > data[i-step]) {

            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
                minfifo.pop_front();
            while(maxfifo.size() > 0) {
                if(data[i] <= data[maxfifo.back()]) {
                    if (i==  width+maxfifo.front())
                        maxfifo.pop_front();
                    break;
                }
                maxfifo.pop_back();
            }

        } else {

            maxfifo.push_back(i-step);
            if (i==  width+maxfifo.front())
                maxfifo.pop_front();
            while(minfifo.size() > 0) {
                if(data[i] >= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                        minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }

        }

    }

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
    minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];

}
