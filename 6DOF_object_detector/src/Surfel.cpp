// Modified by: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#include "Pixel.h"
#include "Surfel.h"



using namespace std;

void Surfel::imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud) {

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


    //  // remove noise
    ////  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    ////  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_cluster_rgb(cloud_cluster);
    //  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
    //  sor1.setInputCloud (cloud);
    //  sor1.setMeanK (200);
    //  sor1.setStddevMulThresh (1.0);
    //  sor1.filter (*cloud);


}




void Surfel::imagesToPointCloud_( cv::Mat& depthImg, cv::Mat& colorImg, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, cv::Mat &mask ) {

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





void Surfel::houghPointCloud( std::vector< cv::Mat >& houghImg,  std::vector< float > &scales, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, float max_val ) {

    cloud->is_dense = true;
    cloud->height = houghImg[ 0 ].rows;// height;
    cloud->width = houghImg[ 0 ].cols; // width;
    cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 0.f );
    cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    cloud->points.reserve( houghImg[ 0 ].rows * houghImg[ 0 ].cols* scales.size() ) ;


    const float invfocalLength = 1.f / 525.f;
    const float centerX = houghImg[ 0 ].cols / 2.f;
    const float centerY = houghImg[ 0 ].rows / 2.f;

    float scaleInterval = ( scales[ scales.size() - 1 ] - scales[ 0 ]) / scales.size();
    int idx = 0;
    for ( unsigned int scNr = 0; scNr< houghImg.size(); scNr++ ){

        float scale = scales[ 0 ] + scNr * scaleInterval;
        float dist = 1 / ( scale + 0.00001f ); // I increased the scale here by 1000 to visualise it better

        for ( int y = 0; y < houghImg[ scNr ].rows; y++ ) {
            for ( int x = 0; x < houghImg[ scNr ].cols; x++ ) {

                float weight =  houghImg[ scNr ].at<float>( y, x  ) / max_val;

                if( weight > 0.9 ){
                    pcl::PointXYZRGB p;
                    float xf = x;
                    float yf = y;
                    p.x = ( xf - centerX ) * dist * invfocalLength;
                    p.y = ( yf - centerY ) * dist * invfocalLength;
                    p.z = dist;

                    float b, g, r;
                    if( weight > 0.9 ){
                        b = 0; //(1 - weight) * 255;
                        g = 0;
                        r = 255; // * weight;
                    }else{
                        b = 150;
                        g = 150;
                        r = 0;
                    }

                    pcl::RGB rgb;

                    rgb.r = ( uint8_t )r;
                    rgb.g = ( uint8_t )g;
                    rgb.b = ( uint8_t )b;
                    rgb.a = 1;

                    p.rgba = rgb.rgba;

                    cloud->push_back( p );
                    idx++;
                }
            }
        }
    }
}

void Surfel::computeSurfel(pcl::PointCloud<pcl::Normal>::Ptr normals, cv::Point2f pt1, cv::Point2f pt2, cv::Point2f center, SurfelFeature &sf, float depth1, float depth2){

    pcl::Normal n1 = normals->at(pt1.x, pt1.y);
    pcl::Normal n2 = normals->at(pt2.x, pt2.y);

    Eigen::Vector3f v1 = n1.getNormalVector3fMap();
    Eigen::Vector3f v2 = n2.getNormalVector3fMap();

    cv::Point3f  ptR1 = CRPixel::P3toR3(pt1, center, depth1);
    cv::Point3f  ptR2 = CRPixel::P3toR3(pt2, center, depth2);

    cv::Point3f temp = ptR1 - ptR2;
    Eigen::Vector3f distVec( temp.x, temp.y, temp.z );

    v1.normalize();
    v2.normalize();
    Eigen::Vector3f distVec_norm = distVec;
    distVec_norm.normalize();

    sf.fVector[0] = std::acos(v1.dot(v2));
    sf.fVector[1] = std::acos(v1.dot(distVec_norm));
    sf.fVector[2] = std::acos(v2.dot(distVec_norm));
    sf.fVector[3] = std::sqrt(distVec.dot(distVec));

}


void Surfel::calcSurfel2CameraTransformation(cv::Point3f &s1, cv::Point3f &s2, pcl::Normal &n1, pcl::Normal &n2, Eigen::Matrix4f &TransformationSC1, Eigen::Matrix4f &TransformationSC2){

    cv::Point3f distance = s1 - s2;
    Eigen::Vector3f t1(s1.x,s1.y,s1.z);
    Eigen::Vector3f t2(s2.x,s2.y,s2.z);
    Eigen::Vector3f dis(distance.x, distance.y, distance.z);
    dis.normalize();

    // First coordinate system
    TransformationSC1                 = Eigen::Matrix4f::Identity();

    Eigen::Vector3f normal1           = n1.getNormalVector3fMap();   normal1.normalize();
    Eigen::Vector3f u1                = n1.getNormalVector3fMap().cross(dis);
    u1.normalize();
    Eigen::Vector3f v1                = n1.getNormalVector3fMap().cross(u1);

    //    cout<< normal1 << endl;
    //    cout<< u1 << endl;
    //    cout<< v1 << endl;
    //    cout<< s1.getVector3fMap() << endl;

    TransformationSC1.block<3,1>(0,2) = normal1;
    TransformationSC1.block<3,1>(0,1) = u1;
    TransformationSC1.block<3,1>(0,0) = v1;
    TransformationSC1.block<3,1>(0,3) = t1;

    //    cout<< TransformationSC1 << endl;


    // Second cordinate syastem

    TransformationSC2                 = Eigen::Matrix4f::Identity();

    Eigen::Vector3f normal2           = n2.getNormalVector3fMap();    normal1.normalize();
    Eigen::Vector3f u2                = n2.getNormalVector3fMap().cross(dis);
    u2.normalize();
    Eigen::Vector3f v2                = n2.getNormalVector3fMap().cross(u2);

    //    cout<< normal2 << endl;
    //    cout<< u2 << endl;
    //    cout<< v2 << endl;
    //    cout<< s2.getVector3fMap() << endl;

    TransformationSC2.block<3,1>(0,2) = normal2;
    TransformationSC2.block<3,1>(0,1) = u2;
    TransformationSC2.block<3,1>(0,0) = v2;
    TransformationSC2.block<3,1>(0,3) = t2;

    //    cout<< TransformationSC2 << endl;

}


void Surfel::calcQueryPoint2CameraTransformation(cv::Point3f& s1, cv::Point3f& s2, cv::Point3f& q, const pcl::Normal& qn, Eigen::Matrix4f& TransformationQueryC1, Eigen::Matrix4f& TransformationQueryC2){

    cv::Point3f distance1 = s1 - q;
    cv::Point3f distance2 = s2 - q;

    Eigen::Vector3f dis1(distance1.x, distance1.y, distance1.z); dis1.normalize();
    Eigen::Vector3f dis2(distance2.x, distance2.y, distance2.z); dis2.normalize();

    // First coordinate system
    TransformationQueryC1             = Eigen::Matrix4f::Identity();

    Eigen::Vector3f normal1           = qn.getNormalVector3fMap();             normal1.normalize();
    Eigen::Vector3f u1                = qn.getNormalVector3fMap().cross(dis1); u1.normalize();
    Eigen::Vector3f v1                = normal1.cross(u1);
    Eigen::Vector3f translation1(q.x, q.y, q.z);

    //    cout<< normal1 << endl;
    //    cout<< u1 << endl;
    //    cout<< v1 << endl;
    //    cout<< translation1.getVector3fMap() << endl;

    TransformationQueryC1.block<3,1>(0,0) = u1;
    TransformationQueryC1.block<3,1>(0,1) = v1;
    TransformationQueryC1.block<3,1>(0,2) = normal1;

    TransformationQueryC1.block<3,1>(0,3) = translation1;

    //    cout<< TransformationQueryC1 << endl;


    // Second cordinate syastem
    TransformationQueryC2             = Eigen::Matrix4f::Identity();

    Eigen::Vector3f normal2           = qn.getNormalVector3fMap();              normal2.normalize();
    Eigen::Vector3f u2                = qn.getNormalVector3fMap().cross(dis2);  u2.normalize();
    Eigen::Vector3f v2                = normal2.cross(u2);
    Eigen::Vector3f translation2(q.x, q.y, q.z);

    //    cout<< normal2 << endl;
    //    cout<< u2 << endl;
    //    cout<< v2 << endl;
    //    cout<< translation2.getVector3fMap() << endl;

    TransformationQueryC2.block<3,1>(0,0) = u2;
    TransformationQueryC2.block<3,1>(0,1) = v2;
    TransformationQueryC2.block<3,1>(0,2) = normal2;

    TransformationQueryC2.block<3,1>(0,3) = translation2;

    //    cout<< TransformationQueryC2 << endl;

}
void  Surfel::addCoordinateSystem( Eigen::Matrix4f &transformationMatrixOC, boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, string id){

    //visualize coordinates in object frame
    pcl::Normal OX(transformationMatrixOC.matrix()(0,0), transformationMatrixOC.matrix()(1,0), transformationMatrixOC.matrix()(2,0));
    pcl::Normal OY(transformationMatrixOC.matrix()(0,1), transformationMatrixOC.matrix()(1,1), transformationMatrixOC.matrix()(2,1));
    pcl::Normal OZ(transformationMatrixOC.matrix()(0,2), transformationMatrixOC.matrix()(1,2), transformationMatrixOC.matrix()(2,2));
    pcl::Normal Origin(transformationMatrixOC.matrix()(0,3), transformationMatrixOC.matrix()(1,3), transformationMatrixOC.matrix()(2,3));

    pcl::PointXYZ O_(Origin.normal_x, Origin.normal_y, Origin.normal_z);
    pcl::PointXYZ X_(0.2 *OX.normal_x + Origin.normal_x, 0.2 *OX.normal_y + Origin.normal_y, 0.2 * OX.normal_z + Origin.normal_z);
    pcl::PointXYZ Y_(0.2 *OY.normal_x + Origin.normal_x, 0.2 *OY.normal_y + Origin.normal_y, 0.2 * OY.normal_z + Origin.normal_z);
    pcl::PointXYZ Z_(0.2 *OZ.normal_x + Origin.normal_x, 0.2 *OZ.normal_y + Origin.normal_y, 0.2 * OZ.normal_z + Origin.normal_z);

    viewer->addLine( O_, X_, 255, 0, 0, ( "lineX" + id).c_str() );
    viewer->addLine( O_, Y_, 0, 255, 0, ( "liney" + id).c_str() );
    viewer->addLine( O_, Z_, 0, 0, 255, ( "linez" + id).c_str() );

}

void Surfel::addCoordinateSystem(const Eigen::Matrix4f &transformationMatrixOC, boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, string id){

    Eigen::Matrix4f transformationMatrixOC1  =  transformationMatrixOC;
    addCoordinateSystem( transformationMatrixOC1, viewer, id);
}



//void imagesToPointCloud(const sensor_msgs::Image& depthImg, const sensor_msgs::Image colorImg,
//                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
//
//  cloud->header = depthImg.header;
//  cloud->is_dense = true;
//  cloud->height = depthImg.height;
//  cloud->width = depthImg.width;
//  cloud->sensor_origin_ = Eigen::Vector4f(0.f, 0.f, 0.f, 0.f);
//  cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
//  cloud->points.resize(depthImg.width * depthImg.height);
//
//  const float invfocalLength = 1.f / 525.f;
//  const float centerX = 319.5f;
//  const float centerY = 239.5f;
//
//  const float* depthdata = reinterpret_cast<const float*>(&depthImg.data[0]);
//  const unsigned char* colordata = &colorImg.data[0];
//  int idx = 0;
//  for (unsigned int y = 0; y < depthImg.height; y++) {
//    for (unsigned int x = 0; x < depthImg.width; x++) {
//
//      pcl::PointXYZRGB& p = cloud->points[idx];
//
//      float dist = (*depthdata);
//
//      if (isnan(dist)) {
//        p.x = std::numeric_limits<float>::quiet_NaN();
//        p.y = std::numeric_limits<float>::quiet_NaN();
//        p.z = std::numeric_limits<float>::quiet_NaN();
//      }
//      else {
//
//        float xf = x;
//        float yf = y;
//        p.x = (xf - centerX) * dist * invfocalLength;
//        p.y = (yf - centerY) * dist * invfocalLength;
//        p.z = dist;
//      }
//
//      depthdata++;
//
//      int r = (*colordata++);
//      int g = (*colordata++);
//      int b = (*colordata++);
//
//      int rgb = (r << 16) + (g << 8) + b;
//      p.rgb = *(reinterpret_cast<float*>(&rgb));
//
//      idx++;
//
//    }
//  }
//}
