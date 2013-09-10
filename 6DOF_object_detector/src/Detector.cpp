/*
// C++ Implementation CRForestDetector
//
// Description: The detector implementation.
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
*/


// Modified by: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#include <vector>
#include <algorithm>

#include "Detector.h"

using namespace std;

int COUNT;


// **********************************    LEAF ASSIGNMENT      ***************************************************** //

// matching the image to the forest and store the leaf assignments in vImgAssing
void CRForestDetector::assignCluster(const cv::Mat &img, const cv::Mat &depthImg, vector<cv::Mat> &vImgAssign, const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals){

    cv::Point pt;
    time_t t = time(NULL);
    int seed = (int)t;
    double value= 0.0;
    CvRNG pRNG(seed);
    vector< int > result;

    float scale;
    bool do_regression;

    //#pragma omp parallel for private( pt, seed, value, pRNG, result )
    for(int y=0; y < img.rows ; ++y) {
        for(int x=0; x < img.cols; ++x) {

            value = cvRandReal(&pRNG);

            do_regression = true;
            if (sample_points > 0 && value < sample_points)//  this "if" statement is always true, because sample_points = -1
                do_regression = false;

            // for each pixel as a upperleft corner regression is done for the patch of size width x height
            if (do_regression){

                pt.x = x;
                pt.y = y;
                if(depthImg.at<unsigned short>(pt) == 0)
                    scale = 1;
                else
                    scale = 1000.f/(float)depthImg.at<unsigned short>(pt); // convert from millimeter to meter


                crForest->regression( result, vImg, normals, pt, scale );// result has Leafnodes form all the trees matching with img
                // and id of leaf is saved for each tree
                for (unsigned int treeNr=0; treeNr < result.size(); treeNr++){
                    vImgAssign[treeNr].at<float>(pt) = float(result[treeNr]);
                }
            }
        } // end for x
    } // end for y

}

// Multi-scale cluster assignment into vvImgAssign.
void CRForestDetector::fullAssignCluster(const cv::Mat &img, const cv::Mat &depthImg, vector< cv::Mat > &vImgAssign, const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals){

    int ntrees = crForest->vTrees.size();
    cv::Scalar s(-1.0);

    vImgAssign.resize(ntrees);

    //looping over the trees in the forest
    for (int treeNr=0; treeNr < ntrees; treeNr++)// for each tree
        vImgAssign[treeNr] =cv::Mat(img.rows,img.cols, CV_32FC1, s); // image is initialized with -1 , which indicates not matched regions

    // matching the img to the forest and store the leaf assignments in vImgAssing
    assignCluster(img, depthImg, vImgAssign, vImg , normals);

}

// ************************************** CLASS CONFIDENCES ****************************************** //

// Getting the per class confidences TODO: this has to become scalable
void CRForestDetector::getClassConfidence(const std::vector<cv::Mat> &vImgAssign, std::vector<cv::Mat> &classConfidence){

    int nlabels = crForest->GetNumLabels();
    classConfidence.resize(nlabels);

    for ( int i=0; i < nlabels; i++)
        classConfidence[i] = cv::Mat::zeros( vImgAssign[0].rows,vImgAssign[0].cols, CV_32FC1);

    float ntrees = float(vImgAssign.size());

    // function variables
    int outer_window = 8; // TODO: this parameter shall move to the inputs.
    float inv_tree = 1.0f/ntrees;

    // looping over trees
    std::vector<std::vector<cv::Mat > >tmpClassProbs(ntrees);
    for (unsigned int trNr=0; trNr < ntrees; trNr++){

        tmpClassProbs[trNr].resize(nlabels);
        // here make a temporary structure of all the probabilities and then smooth it with a kernel.
        for (int cNr=0; cNr < nlabels; cNr++){
            tmpClassProbs[trNr][cNr] = cv::Mat::zeros( vImgAssign[trNr].rows,vImgAssign[trNr].cols, CV_32FC1);

            for ( int y = 0; y < vImgAssign[trNr].rows ; y++){
                for ( int x=0; x < vImgAssign[trNr].cols; x++){
                    int leaf_id = vImgAssign[trNr].at<float>(y,x);
                    LeafNode* tmp = crForest->vTrees[trNr]->getLeaf(leaf_id);

                    if ( leaf_id >= 0 ){
                        tmpClassProbs[trNr][cNr].at<float>(y,x) = tmp->vPrLabel[cNr]*inv_tree;

                    }
                }
            }
        }



        for (int cNr=0; cNr < nlabels; cNr++){
            //SMOOTHING AND SCALING IF NECESSARY
            double scaleFactor = 1.0;
            if ( sample_points >= 0 ) {
                scaleFactor = 1.0/(1.0-sample_points);
            }
            // now values of the tmpClassProbs are set we can blur it to get the average
            cv::GaussianBlur(tmpClassProbs[trNr][cNr], tmpClassProbs[trNr][cNr], cv::Size(outer_window+1, outer_window+1), 0);
            //cv::convertScaleAbs(tmpClassProbs[trNr][cNr], tmpClassProbs[trNr][cNr], scaleFactor);// to account for the sub-sampling // why scaling ??

            // add confidence of all the trees
            cv::add(classConfidence[cNr], tmpClassProbs[trNr][cNr], classConfidence[cNr]);

            if(0){
                //cv::imshow("prob", tmpClassProbs[trNr][cNr]);
                //				cv::imshow("prob", classConfidence[cNr]); cv::waitKey(0);
                double min, max;
                cv::Mat tmp;
                cv::Point max_loc, min_loc;
                cv::minMaxLoc(classConfidence[cNr], &min, &max, &max_loc, &min_loc, cv::Mat());
                cv::convertScaleAbs(classConfidence[cNr],tmp,255/max);
                // shows hough votes for center.
                cv::imshow("prob",tmp); cv::waitKey(0);

            }
        }
    }// end loop over tree
}

/********************************** FULL object detection ************************************/
void CRForestDetector::detectPosePeaks(vector< cv::Mat > &positiveAcc, vector< cv::Mat> &negativeAcc, Eigen::Matrix3f &positiveFinalOC, Eigen::Matrix3f &negativeFinalOC){

    int kernelSize = 5;
    float std = 0.1;
    int step = 50;

    unsigned int bins = positiveAcc.size();

    //define variables
    std::vector< cv::Mat > positiveSmoothAcc( bins );
    std::vector< cv::Mat > negativeSmoothAcc( bins );

    std::vector< cv::Mat > positiveSmoothAccTemp( bins );
    std::vector< cv::Mat > negativeSmoothAccTemp( bins );

    std::vector< cv::Mat > positiveDilatedImg( bins ), positiveComp( bins );
    std::vector< cv::Mat > negativeDilatedImg( bins ), negativeComp( bins );

    std::vector< cv::Mat > positiveLocalMax( bins );
    std::vector< cv::Mat > negativeLocalMax( bins );

    cv::Mat gaussKernel = cv::getGaussianKernel( kernelSize, std, CV_32F );

    for(unsigned int binNr = 0; binNr< bins; binNr++){

        // smoothing the accumulator matrix

        cv::GaussianBlur( positiveAcc[ binNr ], positiveSmoothAcc[ binNr ], cv::Size( kernelSize, kernelSize ), std );
        cv::GaussianBlur( negativeAcc[ binNr ], negativeSmoothAcc[ binNr ], cv::Size( kernelSize, kernelSize ), std );// smooth in direction of x and y
        positiveSmoothAccTemp[ binNr ] = positiveSmoothAcc[ binNr ];
        negativeSmoothAccTemp[ binNr ] = negativeSmoothAcc[ binNr ];
    }

    //    if( 0 )
    //        for( unsigned int scNr = 0; scNr < nScales; scNr++ ){
    //            cv::imshow( "smooth_hough", smoothAcc[ scNr ] ); cv::waitKey( 0 );
    //        }


    // Smoothing in third dimension
    for( int r = 0; r < bins; r++ ){
        for( int c = 0; c< bins; c++ ){

            for( unsigned int binNr = 0; binNr < bins; binNr++ ){
                int binBegin = std::max( 0, (int)binNr - int( kernelSize / 2 ) );
                int binEnd = std::min( (int)binNr + int( kernelSize / 2 ) + 1, (int)bins );

                float positiveConvSum = 0, negativeConvSum = 0;
                int k = 0;

                for( int td = binBegin; td < binEnd; td++, k++ ){

                    positiveConvSum +=  positiveSmoothAcc[ td ].at< float >( r, c ) * gaussKernel.at< float >( k );
                    negativeConvSum +=  negativeSmoothAcc[ td ].at< float >( r, c ) * gaussKernel.at< float >( k );

                }

                positiveSmoothAccTemp[ binNr ].at< float >( r, c ) = positiveConvSum;
                negativeSmoothAccTemp[ binNr ].at< float >( r, c ) = negativeConvSum;
            }
        }
    }

    if( 0 )
        for( unsigned int binNr = 0; binNr < bins; binNr++ ){
            cv::imshow( "+ve accumulator", positiveSmoothAccTemp[ binNr ]); cv::waitKey( 0 );
            cv::imshow( "-ve accumulator", negativeSmoothAccTemp[ binNr ]); cv::waitKey( 0 );
        }


    // find local maximum

    // ............ dilate the image.........

    // dilate in x and y direction

    cv::Mat dilationKernel = cv::Mat::ones( kernelSize, kernelSize, CV_32FC1 );

    for( unsigned int binNr = 0; binNr < bins; binNr++ ){

        cv::dilate(positiveSmoothAccTemp[ binNr ], positiveDilatedImg[ binNr ], dilationKernel );
        cv::dilate(negativeSmoothAccTemp[ binNr ], negativeDilatedImg[ binNr ], dilationKernel );

    }

    //    if( 0 )
    //        for( unsigned int scNr = 0; scNr < nScales; scNr++ ){
    //            cv::imshow( "dilated_hough", dilatedImg[ scNr ] ); cv::waitKey( 0 );
    //    }

    // dilate it in third dimension

    for(int r = 0; r < bins; r++ ){
        for( int c = 0; c < bins; c++){

            for( int binNr = 0; binNr < bins; binNr++ ){
                int binBegin = std::max( 0, (int)binNr - int( kernelSize / 2 ) );
                int binEnd = std::min( (int)binNr + int(kernelSize/2) + 1, (int)bins );

                float negativeMax_val = -1.f, positiveMax_val = -1.f;

                for( int td = binBegin; td < binEnd; td++ ){

                    positiveMax_val = std::max( negativeMax_val, negativeDilatedImg[ td ].at< float >( r, c ) );
                    negativeMax_val = std::max( negativeMax_val, negativeDilatedImg[ td ].at< float >( r, c ) );
                }

                positiveDilatedImg[ binNr ].at< float >( r, c ) = positiveMax_val;
                negativeDilatedImg[ binNr ].at< float >( r, c ) = negativeMax_val;
            }
        }
    }

    //    if( 0 )
    //        for(unsigned int scNr = 0; scNr < nScales; scNr++ ){
    //            cv::imshow( "dilated_hough_scales", dilatedImg[ scNr ]); cv::waitKey( 0 );
    //    }



    for(unsigned int binNr = 0; binNr < bins; binNr++ ){

        cv::compare( positiveSmoothAccTemp[ binNr ], positiveDilatedImg[ binNr ], positiveComp[ binNr ], CV_CMP_EQ ); //cv::imshow("compare", comp[ scNr ]); cv::waitKey(0);
        cv::multiply( positiveSmoothAccTemp[ binNr ], positiveComp[ binNr ], positiveLocalMax[ binNr ], 1/255.f, CV_32F ); //cv::imshow("localmax", localMax); cv::waitKey(0);

        cv::compare( negativeSmoothAccTemp[ binNr ], negativeDilatedImg[ binNr ], negativeComp[ binNr ], CV_CMP_EQ ); //cv::imshow("compare", comp[ scNr ]); cv::waitKey(0);
        cv::multiply( negativeSmoothAccTemp[ binNr ], negativeComp[ binNr ], negativeLocalMax[ binNr ], 1/255.f, CV_32F ); //cv::imshow("localmax", localMax); cv::waitKey(0);

    }

    //    if( 0 )
    //        for(unsigned int scNr = 0; scNr < nScales; scNr++ ){
    //            cv::imshow( "localMax", localMax[ scNr ]); cv::waitKey( 0 );
    //    }


    // Detect the maximum

    // For positive qw

    std::vector< cv::Point > max_loc_temp( bins );
    std::vector< double > max_val_temp( bins );

    for( unsigned int binNr = 0; binNr < bins; binNr++)
        cv::minMaxLoc( positiveLocalMax[ binNr ], 0, &max_val_temp[ binNr ], 0, &max_loc_temp[ binNr ], cv::Mat() );

    std::vector< double >::iterator it;
    it = std::max_element(max_val_temp.begin(),max_val_temp.end());
    int max_index = std::distance( max_val_temp.begin(), it );

    float positiveMax = positiveLocalMax[max_index].at<float>(max_loc_temp[max_index]);

    // Convert it back to qx qy qz values from bin indices

    float qx = (max_loc_temp[max_index].x + 1.f) * 2.f /step -1;
    float qy = (max_loc_temp[max_index].y + 1.f) * 2.f /step -1;
    float qz = (max_index + 1.f) * 2.f /step -1;

    Eigen::Vector3f axis(qx,qy,qz);
    float norm = axis.norm();
    cout<< norm <<"\n"<<endl;
    //    Eigen::Quaternionf positivePose(sqrt(1-norm*norm) , qx, qy, qz);
    Eigen::Quaternionf positivePose(0, qx, qy, qz);

    positivePose.normalize();
    positiveFinalOC = Eigen::Matrix3f(positivePose);

    cout<<"+ve\n "<< positiveFinalOC<<endl;


    // For negative qw

    for( unsigned int binNr = 0; binNr < bins; binNr++)
        cv::minMaxLoc( negativeLocalMax[ binNr ], 0, &max_val_temp[ binNr ], 0, &max_loc_temp[ binNr ], cv::Mat() );

    it = std::max_element(max_val_temp.begin(),max_val_temp.end());
    max_index = std::distance( max_val_temp.begin(), it );

    float negativeMax = negativeLocalMax[max_index].at<float>(max_loc_temp[max_index]);

    qx = (max_loc_temp[max_index].x + 1.f) * 2.f /step -1;
    qy = (max_loc_temp[max_index].y + 1.f) * 2.f /step -1;
    qz = (max_index + 1.f) * 2.f /step -1;

    axis = Eigen::Vector3f(qx,qy,qz);
    norm = axis.norm();

    //    Eigen::Quaternionf negativePose(-sqrt(1-norm*norm), qx, qy, qz);
    Eigen::Quaternionf negativePose(0, qx, qy, qz);

    negativePose.normalize();

    negativeFinalOC = Eigen::Matrix3f(negativePose);

    cout<<"-ve\n "<< negativeFinalOC<<endl;

}

void CRForestDetector::detectPosePeaks_meanShift(std::vector<std::vector<float> >&Qx, std::vector<std::vector<float> > &Qy,std::vector<std::vector<float> > &Qz, Eigen::Matrix3f &positiveFinalOC, Eigen::Matrix3f & negativeFinalOC){

    std::vector< Eigen::Quaternionf> finalOC(2);
    for(int i = 0; i < 2; i++){

        float meanshift_x = 0.f;
        float weightsum_x = 0.f;
        float meanshift_y = 0.f;
        float weightsum_y = 0.f;
        float meanshift_z = 0.f;
        float weightsum_z = 0.f;


        if(Qx[i].size() != 0){
            int midPoint_x = Qx[i].size()/2.f;
            int midPoint_y = Qy[i].size()/2.f;
            int midPoint_z = Qz[i].size()/2.f;

            std::sort(Qx[i].begin(), Qx[i].end());
            std::sort(Qy[i].begin(), Qy[i].end());
            std::sort(Qz[i].begin(), Qz[i].end());

            float median_x = Qx[i][midPoint_x];
            float median_y = Qy[i][midPoint_y];
            float median_z = Qz[i][midPoint_z];


            float windowScale = 0.05;

            // local meanshift from current width and height estimate
            const float XMeanShiftWindowSize =  windowScale ;//* median_width;
            const float YMeanShiftWindowSize =  windowScale ; // * median_height;
            const float ZMeanShiftWindowSize =  windowScale ; // * median_height;

            for( unsigned int f = 0; f < Qx[i].size(); f++ ) {
                if( fabsf( Qx[i][f] - median_x ) < XMeanShiftWindowSize ) {
                    meanshift_x += Qx[i][f];
                    weightsum_x += 1.f;
                }
                if( fabsf( Qy[i][f] - median_y ) < YMeanShiftWindowSize ) {
                    meanshift_y += Qy[i][f];
                    weightsum_y += 1.f;
                }
                if( fabsf( Qz[i][f] - median_z ) < ZMeanShiftWindowSize ) {
                    meanshift_z += Qz[i][f];
                    weightsum_z += 1.f;
                }
            }

            if( weightsum_x > std::numeric_limits<float>::epsilon() )
                meanshift_x = meanshift_x / weightsum_x;
            else
                meanshift_x= median_x;

            if( weightsum_y > std::numeric_limits<float>::epsilon() )
                meanshift_y = meanshift_y / weightsum_y;
            else
                meanshift_y= median_y;

            if( weightsum_z > std::numeric_limits<float>::epsilon() )
                meanshift_z = meanshift_z / weightsum_z;
            else
                meanshift_z= median_z;
        }
        finalOC[i] = Eigen::Quaternionf(std::pow(-1,i), meanshift_x, meanshift_y, meanshift_z);
        finalOC[i].normalize();
    }
    positiveFinalOC = Eigen::Matrix3f(finalOC[0]);
    negativeFinalOC = Eigen::Matrix3f(finalOC[1]);
}

void CRForestDetector::detectPosePeaks_meanShift_common(std::vector<float> &Qx, std::vector<float> &Qy, std::vector<float> &Qz, std::vector<float> &Qw, Eigen::Matrix3f & finalOC){

    Eigen::Quaternionf qFinalOC;


    float meanshift_x = 0.f;
    float weightsum_x = 0.f;
    float meanshift_y = 0.f;
    float weightsum_y = 0.f;
    float meanshift_z = 0.f;
    float weightsum_z = 0.f;
    float meanshift_w = 0.f;
    float weightsum_w = 0.f;


    if(Qx.size() != 0){

        int midPoint_x = Qx.size()/2.f;
        int midPoint_y = Qy.size()/2.f;
        int midPoint_z = Qz.size()/2.f;
        int midPoint_w = Qw.size()/2.f;

        std::sort(Qx.begin(), Qx.end());
        std::sort(Qy.begin(), Qy.end());
        std::sort(Qz.begin(), Qz.end());
        std::sort(Qw.begin(), Qw.end());

        float median_x = Qx[midPoint_x];
        float median_y = Qy[midPoint_y];
        float median_z = Qz[midPoint_z];
        float median_w = Qw[midPoint_w];


        float windowScale = 0.05;

        // local meanshift from current width and height estimate
        const float XMeanShiftWindowSize =  windowScale ;
        const float YMeanShiftWindowSize =  windowScale ;
        const float ZMeanShiftWindowSize =  windowScale ;
        const float WMeanShiftWindowSize =  windowScale ;

        for( unsigned int f = 0; f < Qx.size(); f++ ) {
            if( fabsf( Qx[f] - median_x ) < XMeanShiftWindowSize ) {
                meanshift_x += Qx[f];
                weightsum_x += 1.f;
            }
            if( fabsf( Qy[f] - median_y ) < YMeanShiftWindowSize ) {
                meanshift_y += Qy[f];
                weightsum_y += 1.f;
            }
            if( fabsf( Qz[f] - median_z ) < ZMeanShiftWindowSize ) {
                meanshift_z += Qz[f];
                weightsum_z += 1.f;
            }
            if( fabsf( Qw[f] - median_w ) < WMeanShiftWindowSize ) {
                meanshift_w += Qw[f];
                weightsum_w += 1.f;
            }
        }

        if( weightsum_x > std::numeric_limits<float>::epsilon() )
            meanshift_x = meanshift_x / weightsum_x;
        else
            meanshift_x= median_x;

        if( weightsum_y > std::numeric_limits<float>::epsilon() )
            meanshift_y = meanshift_y / weightsum_y;
        else
            meanshift_y= median_y;

        if( weightsum_z > std::numeric_limits<float>::epsilon() )
            meanshift_z = meanshift_z / weightsum_z;
        else
            meanshift_z= median_z;

        if( weightsum_w > std::numeric_limits<float>::epsilon() )
            meanshift_w = meanshift_w / weightsum_w;
        else
            meanshift_w= median_w;

    }

    qFinalOC = Eigen::Quaternionf(meanshift_w, meanshift_x, meanshift_y, meanshift_z);
    qFinalOC.normalize();

    finalOC = Eigen::Matrix3f(qFinalOC);

}

void CRForestDetector::detectPosePeaks_slerp(std::vector<Eigen::Quaternionf>& qMean,Eigen::Matrix3f &finalOC){

    Eigen::Quaternionf tempMean = qMean[0];

    for(unsigned int i = 1; i < qMean.size(); i++){

        tempMean = tempMean.slerp(0.5f,qMean[i]);
    }

    tempMean.normalize();
    finalOC = Eigen::Matrix3f(tempMean);

}

//void CRForestDetector::detectMaxima(const vector<vector<cv::Mat> >& poseHoughSpace,  Eigen::Quaternionf& finalOC, int& step, float& score){

//    // smoothing of the houghspace with gaussian kernel
//    float sigma = 1.f;
//    int kSize = 5;
//    cv::Mat gauss = cv::getGaussianKernel( kSize , sigma, CV_32F);

//    vector<vector<cv::Mat> > poseHoughSpace_smoothed(poseHoughSpace.size());

//    // smoothing in qx qy dimensions
//    for(int i = 0; i < step; i++){ // qw
//        poseHoughSpace_smoothed[i].resize(poseHoughSpace[i].size());

//        for (int j = 0; j < step; j++){ // qz
//            cv::GaussianBlur(poseHoughSpace[i][j], poseHoughSpace_smoothed[i][j], cv::Size(kSize, kSize), sigma);

//        }
//    }


//    score = 0;
//    cv::Mat maximaInXY = cv::Mat::zeros(step, step, CV_32FC1);
//    cv::Mat locationInX = cv::Mat::zeros(step, step, CV_32FC1);
//    cv::Mat locationInY = cv::Mat::zeros(step, step, CV_32FC1);

//    for(int i = 0; i < step; i++){ // qw
//        for (int j = 0; j < step; j++){ // qz
//            double maxVal, minVal;
//            cv::Point maxIdx, minIdx;
//            cv::minMaxLoc(poseHoughSpace[i][j], &minVal, &maxVal, &minIdx, &maxIdx);
//            maximaInXY.at<float>(j,i) = maxVal;
//            locationInX.at<float> (j,i) = maxIdx.x;
//            locationInY.at<float> (j,i) = maxIdx.y;

//        }
//    }

//    double maxVal, minVal;
//    cv::Point maxIdx, minIdx;
//    cv::minMaxLoc(maximaInXY, &minVal, &maxVal, &minIdx, &maxIdx);

//    score = maxVal;

//    float dqw = maxIdx.x;
//    float dqz = maxIdx.y;

//    float dqy = locationInY.at<float>(maxIdx);
//    float dqx = locationInX.at<float>(maxIdx);

//    float qx = (dqx + 1.f) * 2.f/(float)step - 1.f;
//    float qy = (dqy + 1.f) * 2.f/(float)step - 1.f;
//    float qz = (dqz + 1.f) * 2.f/(float)step - 1.f;
//    float qw = (dqw + 1.f) * 2.f/(float)step - 1.f;

//    Eigen::Quaternionf rotation(qw, qx, qy, qz);

//    rotation.normalize();
//    finalOC = Eigen::Matrix3f(rotation);

//}


void CRForestDetector::detectMaxima(const vector<cv::Mat> & poseHoughSpace,  Eigen::Quaternionf& finalOC, int& step, float& score){

    // smoothing of the houghspace with gaussian kernel
    float sigma = 1.f;
    int kSize = 5;
    cv::Mat gauss = cv::getGaussianKernel( kSize , sigma, CV_32F);

    vector< cv::Mat > poseHoughSpace_smoothed(poseHoughSpace.size());

    // smoothing in qx qy dimensions
    for(int i = 0; i < step; i++){ // qx, qy

        cv::GaussianBlur( poseHoughSpace[i], poseHoughSpace_smoothed[i], cv::Size(kSize, kSize), sigma );
    }

    vector<cv::Mat> poseHoughSmoothed3(step);

    for(int i = 0; i < step; i++){ // qx, qy

        poseHoughSmoothed3[i] = cv::Mat::zeros(step, step, CV_32FC1);
    }

    cv::Mat gaussKernel = cv::getGaussianKernel( kSize, sigma, CV_32F );

    // Smoothing in third dimension
    for( int r = 0; r < poseHoughSpace_smoothed[0].rows; r++ ){
        for( int c = 0; c< poseHoughSpace_smoothed[ 0 ].cols; c++ ){

            for( int s = 0; s < step; s++ ){
                int scBegin = std::max( 0, s - int( kSize / 2 ) );
                int scEnd = std::min( s + int( kSize/ 2 ) + 1, step );

                float convSum = 0;
                int k = 0;

                for( int td = scBegin; td < scEnd; td++, k++ ){

                    convSum += poseHoughSpace_smoothed[ td ].at< float >( r, c ) * gaussKernel.at< float >( k );

                }

                poseHoughSmoothed3[ s ].at< float >( r, c ) = convSum;
            }
        }
    }


    vector<cv::Point> max_loc;
    vector<double> max_val;
    for(int i = 0; i < step; i++) // qz

        cv::minMaxLoc(poseHoughSmoothed3[i], 0, &max_val[i], 0, &max_loc[i], cv::Mat());

    std::vector< double >::iterator it;
    it = std::max_element( max_val.begin(),max_val.end() );
    int max_index = std::distance( max_val.begin(), it );

    score = *it;

    float dqz = max_index;

    float dqy = max_loc[max_index].y;
    float dqx = max_loc[max_index].x;

    float qx = (dqx + 1.f) * 2.f/(float)step - 1.f;
    float qy = (dqy + 1.f) * 2.f/(float)step - 1.f;
    float qz = (dqz + 1.f) * 2.f/(float)step - 1.f;
    float qw = std::sqrt( 1 - qx*qx -qy*qy -qz*qz);

    Eigen::Quaternionf rotation(qw, qx, qy, qz);

    rotation.normalize();
    finalOC = Eigen::Matrix3f(rotation);

}


void CRForestDetector::axisOfSymmetry(std::vector<Eigen::Quaternionf> &qMean, Eigen::Quaternionf &qfinalOC){


    ofstream axisOfSym;
    axisOfSym.open ("axisOfSym.txt");

    int bins = 100;
    std::vector<cv::Mat> axisHoughSpace(bins);
    std::vector< float > angles(bins,0);
    for(unsigned int b = 0; b< bins; b++)
        axisHoughSpace[b] = cv::Mat::zeros(bins, bins, CV_32FC1);

    for(int i = 0; i < qMean.size(); i++ ){

        Eigen::Quaternionf qtemp = qfinalOC.inverse() * qMean[i];

        Eigen::AngleAxisf atemp = Eigen::AngleAxisf(qtemp);

        Eigen::Vector3f axis = atemp.axis();

        //        cout<< "axis: "<< axis.transpose()<<endl;

        axisOfSym << qtemp.x() << " " << qtemp.y() << " " << qtemp.z() << endl;

        int  aX = std::max(0.f, ( axis(0) + 1.f ) * (float)bins/2.f - 1.f );
        aX = std::min( aX, bins-1 );

        int  aY = std::max(0.f, ( axis(1) + 1.f ) * (float)bins/2.f - 1.f );
        aY = std::min( aY, bins-1 );

        int  aZ = std::max(0.f, ( axis(2) + 1.f ) * (float)bins/2.f - 1.f );
        aZ = std::min( aZ, bins-1 );

        //        cout<< " " << aX << " " << aY << " " << aZ << endl;

        axisHoughSpace[aZ].at<float>(aY, aX)++;
        int angle =  std::max(0.f, atemp.angle() * (float) bins/(2.f * PI));
        angle = std::min(angle, bins - 1);
        //        cout<< "\nangle: "<<atemp.angle() * 180.f/PI <<endl;
        angles[angle]++;

    }

    axisOfSym.close();

    //  find maxima in histogram for angles

    std::vector< float>::iterator it_angle;
    it_angle = std::max_element(angles.begin(),angles.end());
    int dAngle = std::distance( angles.begin(), it_angle );
    float votes = angles[dAngle];

    float angle = (dAngle * 2.f* PI) /((float)bins) ;


    cout<< "\nangle = " <<angle<<" votes: "<< votes<< " percentage votes: "<< votes*100.f/qMean.size()<< endl;


    //  find maxima in histogram for axis

    std::vector<double> maxVal(bins), minVal(bins);
    std::vector<cv::Point> maxIdx(bins), minIdx(bins);

    for (int j = 0; j < bins; j++){ // aZ

        cv::minMaxLoc(axisHoughSpace[j], &minVal[j], &maxVal[j], &minIdx[j], &maxIdx[j]);
    }

    std::vector< double >::iterator it;
    it = std::max_element(maxVal.begin(),maxVal.end());
    float daZ = std::distance( maxVal.begin(), it );
    float daX = maxIdx[daZ].x;
    float daY = maxIdx[daZ].y;

    //    cout<<" \n";
    cout<<"\nvotes: " <<maxVal[daZ];
    //    cout << daX <<" "<< daY<< " "<< daZ <<endl;

    float aX = (daX + 1.f) * 2.f/(float)bins - 1.f;
    float aY = (daY + 1.f) * 2.f/(float)bins - 1.f;
    float aZ = (daZ + 1.f) * 2.f/(float)bins - 1.f;


    Eigen::Vector3f symmetryAxis(aX, aY, aZ);
    symmetryAxis.normalize();

    cout<<" \nAxis of Symmetry: "<< symmetryAxis<< endl;

    double log_constant  = std::log(qMean.size());
    double constant  = qMean.size();
    // entropy of axis
    double entropy_sum_axis = 0.f;
    for(int z = 0; z < bins; z++){
        for(int y = 0; y < bins; y++){
            for(int x = 0; x < bins; x++){
                entropy_sum_axis += (-1.f) * axisHoughSpace[z].at<float>(y,x)/constant * (log(axisHoughSpace[z].at<float>(y,x)) - log_constant);
            }
        }
    }


    // entropy of angles
    double entropy_sum_angle = 0.f;
    for( int b = 0; b < bins; b++ ){

        entropy_sum_angle += (-1.f) * angles[b]/constant * (log(angles[b]) - log_constant);

    }

    cout<< "\nentropy of axis: "<< entropy_sum_axis << " entropy of angles: "<< entropy_sum_angle << endl;


}

void CRForestDetector::detectMaximaK_means(std::vector<Eigen::Quaternionf> &qMean, Eigen::Matrix3f &finalOC){

    //    cv::Mat xVectors(qMean.size(), 3, CV_32FC1);
    //    cv::Mat yVectors(qMean.size(), 3, CV_32FC1);
    //    cv::Mat zVectors(qMean.size(), 3, CV_32FC1);

    //    // gather data to input in kMeans

    //    for(int i = 0; i < qMean.size(); i++){

    //        Eigen::Matrix3f rotMat = Eigen::Matrix3f(qMean[i]);

    //        xVectors.at<float>(i,0) = rotMat(0,0); xVectors.at<float>(i,1) = rotMat(1,0); xVectors.at<float>(i,2) = rotMat(2,0);

    //        yVectors.at<float>(i,0) = rotMat(0,1); yVectors.at<float>(i,1) = rotMat(1,1); yVectors.at<float>(i,2) = rotMat(2,1);

    //        zVectors.at<float>(i,0) = rotMat(0,2); zVectors.at<float>(i,1) = rotMat(1,2); zVectors.at<float>(i,2) = rotMat(2,2);

    //    }

    //    int attempts = 5;
    //    int num_clusters = 3;

    //    cv::Mat output;
    //    vector<float> totalLabel(num_clusters);
    //    std::vector<Eigen::Vector3f> strongestKMeans(3);

    //    std::vector< XYZIndex> maxXYZ;
    //    XYZIndex temp;

    //    //    // for x
    //    cv::Mat xbestLabels, xCenters;
    //    cv::kmeans(xVectors, num_clusters, xbestLabels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0.0001, 10000), attempts, cv::KMEANS_PP_CENTERS, xCenters);

    //    for(int i = 0; i < num_clusters; i++ ){
    //        //        cv::Mat label = cv::Mat(xVectors.rows, 1, CV_32FC1, cv::Scalar(i));
    //        cv::Mat integralSum;
    //        cv::compare(xbestLabels, cv::Scalar(i), output, CV_CMP_EQ);
    //        output /= 255;
    //        cv::integral(output, integralSum, CV_32F);
    //        totalLabel[i] = integralSum.at<float>(integralSum.rows -1 , integralSum.cols-1);
    //    }

    //    std::vector< float >::iterator it;
    //    it = std::max_element(totalLabel.begin(), totalLabel.end());
    //    int X_k_means_index = std::distance( totalLabel.begin(), it );
    //    float X_totalPoints = totalLabel[X_k_means_index];
    //    strongestKMeans[0] = Eigen::Vector3f(xCenters.at<float>(X_k_means_index,0), xCenters.at<float>(X_k_means_index,1), xCenters.at<float>(X_k_means_index,2));


    //    temp.val = X_totalPoints;
    //    temp.index = 0;
    //    maxXYZ.push_back(temp);

    //    //    // for y
    //    cv::Mat ybestLabels, yCenters;
    //    cv::kmeans(yVectors, 3, ybestLabels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0.0001, 10000), attempts, cv::KMEANS_PP_CENTERS, yCenters);

    //    for(int i = 0; i < num_clusters; i++ ){
    //        //        cv::Mat label = cv::Mat(xVectors.rows, 1,CV_32FC1, cv::Scalar(i));
    //        cv::Mat integralSum;
    //        cv::compare(ybestLabels, cv::Scalar(i), output, CV_CMP_EQ);
    //        output /= 255;
    //        cv::integral(output, integralSum, CV_32F);
    //        totalLabel[i] = integralSum.at<float>(integralSum.rows -1 , integralSum.cols-1);
    //    }


    //    it = std::max_element(totalLabel.begin(), totalLabel.end());
    //    int Y_k_means_index = std::distance(totalLabel.begin(), it );
    //    float Y_totalPoints = totalLabel[Y_k_means_index];
    //    strongestKMeans[1] = Eigen::Vector3f(yCenters.at<float>(Y_k_means_index,0), yCenters.at<float>(Y_k_means_index,1), yCenters.at<float>(Y_k_means_index,2));

    //    temp.val = Y_totalPoints;
    //    temp.index = 1;
    //    maxXYZ.push_back(temp);

    //    // for z
    //    cv::Mat zbestLabels, zCenters;
    //    cv::kmeans(zVectors, 3, zbestLabels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0.0001, 10000), attempts, cv::KMEANS_PP_CENTERS, zCenters);


    //    for(int i = 0; i < num_clusters; i++ ){
    //        //        cv::Mat label = cv::Mat(xVectors.rows, 1, CV_32FC1,cv::Scalar(i));
    //        cv::Mat integralSum;
    //        cv::compare(zbestLabels, cv::Scalar(i), output, CV_CMP_EQ);
    //        output /= 255;
    //        cv::integral(output, integralSum, CV_32F);
    //        totalLabel[i] = integralSum.at<float>(integralSum.rows -1 , integralSum.cols - 1);
    //    }


    //    it = std::max_element(totalLabel.begin(), totalLabel.end());
    //    int Z_k_means_index = std::distance( totalLabel.begin(), it );
    //    float Z_totalPoints = totalLabel[Z_k_means_index];
    //    strongestKMeans[0] = Eigen::Vector3f(zCenters.at<float>(Z_k_means_index,0), zCenters.at<float>(Z_k_means_index,1), zCenters.at<float>(Z_k_means_index,2));

    //    temp.val = Z_totalPoints;
    //    temp.index = 2;
    //    maxXYZ.push_back(temp);

    //    sort(maxXYZ.begin(), maxXYZ.end());

    //    Eigen::Vector3f firsthighest = strongestKMeans[maxXYZ[2].index];
    //    Eigen::Vector3f secondhighest = strongestKMeans[maxXYZ[1].index];
    //    Eigen::Vector3f smallest;
    //    if(abs(maxXYZ[2].index - maxXYZ[1].index) == 1){
    //        smallest = (maxXYZ[2].index - maxXYZ[1].index) * (-1) * firsthighest.cross(secondhighest);
    //    }else{

    //        smallest = (maxXYZ[2].index - maxXYZ[1].index)/2.f *firsthighest.cross(secondhighest);
    //    }

    //    finalOC.block<3,1>(0, maxXYZ[2].index) = firsthighest;
    //    finalOC.block<3,1>(0, maxXYZ[1].index) = secondhighest;
    //    finalOC.block<3,1>(0, maxXYZ[0].index) = smallest;


}

void CRForestDetector::voteForPose(const cv::Mat img, const std::vector< std::vector< std::vector< std::vector<std::vector< std::pair< cv::Point, int > > > > > >& voterImages, const std::vector<cv::Mat>& vImgAssign, const std::vector<std::vector<cv::Mat> >& vImgDetect, std::vector<Candidate>& candidates, const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const int kernel_width, const std::vector<float>&scales, const float thresh, const bool DEBUG, const bool addPoseScore){

    if(candidates.size() > 0){

        std::vector< Eigen::Matrix4f, Eigen::aligned_allocator< Eigen::Matrix4f> > candPoses;
        candPoses.reserve(candidates.size());
        int nTrees = vImgAssign.size();

        for ( unsigned int cand = 0; cand < candidates.size(); cand++ ){ // loop on candidates we will take for now only the first candidate

            if(candidates[cand].weight < thresh)
                continue;

            int step = 20, subsample_count = 0;
            int cNr = candidates[cand].c;

            std::vector <Eigen::Quaternionf> qMean;
            std::vector <cv::Mat> poseHoughSpace(step);
            for(int i = 0; i<step; i++)
                poseHoughSpace[i] = cv::Mat::zeros(step, step, CV_32FC1);

            int x = candidates[ cand ].center.x;
            int y = candidates[ cand ].center.y;
            int scale_number = candidates[ cand ].scale * scales.size()/(scales[scales.size() - 1 ]- scales[ 0 ] ) - 1 ;
            cv::Point2f img_center( vImg[ 0 ].cols/2.f, vImg[ 0 ].rows/2.f );

            // since hough space was smoothed by kernel_width window size in xy space and by scale winodow size in z dimension, all the votes contributed to this peak should be
            int min_s = std::max( 0, scale_number - 1 );
            int max_s = std::min( int(scales.size()), scale_number + 2);

            cv::Point2f oCenter( x, y );
            cv::Point3f oCenter_real = CRPixel::P3toR3( oCenter, img_center, 1/candidates[ cand ].scale );

            for(int scNr = min_s; scNr < max_s; scNr++ ){ //scales

                int min_x = std::max( 0, x - int( kernel_width * scales[scNr] ) );
                int max_x = std::min( vImg[0].cols, x + int( kernel_width * scales[scNr] ) );

                int min_y = std::max( 0, y - int( kernel_width * scales[scNr]) );
                int max_y = std::min( vImg[0].rows, y + int( kernel_width * scales[scNr] ) );

                for(int cy =  min_y; cy < max_y; cy++ ){ // y
                    for( int cx =  min_x; cx < max_x; cx++ ){ //x

                        float weight_ = vImgDetect[cNr][scNr].at< float >(y,x) / nTrees;

                        for ( unsigned int trNr = 0; trNr < vImgAssign.size(); trNr ++ ){ // loop for all the trees

                            for ( unsigned int pVotes = 0; pVotes < voterImages[ trNr ][scNr][ cy  ][ cx ].size(); pVotes++ ){ // loop for all the training pixels voted for the center

                                weight_ /= voterImages[ trNr ][scNr][ cy  ][ cx ].size();

                                cv::Point2f qPixel = voterImages[ trNr ][ scNr ][ cy  ][ cx ][ pVotes ].first;
                                int index = voterImages[ trNr ][ scNr ][ cy  ][ cx ][ pVotes ].second;
                                int leafID = vImgAssign[trNr].at< float >(qPixel);
                                LeafNode* L = crForest->getLeaf( trNr, leafID );// getLeaf(index);
                                unsigned int depth = L->depth;

                                if(L->vPose[ cNr ].size() == 0)
                                    continue;

                                // get normals from feature channel
                                pcl::Normal qn = normals->at( qPixel.x, qPixel.y );
                                if(isnan(qn.normal_x) || isnan(qn.normal_y) || isnan(qn.normal_z) )
                                    continue;

                                // find the parent node, collect the test and create two offsets

                                std::vector< int > nodeIDs(L->depth);
                                InternalNode* node  = crForest->getNode( trNr, L->parent );

                                for(int d = 0; d < L->depth; d++){

                                    nodeIDs[ L->depth -1 - d ] = node->idN;
                                    node = crForest->getNode( trNr, node->parent );
                                }

                                Eigen::Matrix4f transformationMatrixQuery1C, transformationMatrixQuery2C;


                                for( unsigned int d = 0; d < depth; d++ ){ // depth of node

                                    node  = crForest->getNode( trNr, nodeIDs[d] );
                                    // find the test and generate offsets cv::point2f p1 and p2
                                    cv::Point2f pt1,pt2;

                                    pt1.x = std::max( 0, int( qPixel.x + node->data[ 0 ] * 1000.f/vImg[ 7 ].at< unsigned short >( qPixel ) ) );
                                    pt1.x = std::min( int( pt1.x ), vImg[ 0 ].cols - 1 );

                                    pt1.y = std::max( 0, int( qPixel.y + node->data[ 1 ] * 1000.f/vImg[ 7 ].at< unsigned short >( qPixel ) ) );
                                    pt1.y = std::min( int( pt1.y ), vImg[ 0 ].rows - 1 );

                                    pt2.x = std::max( 0, int( qPixel.x + node->data[ 2 ] * 1000.f/vImg[ 7 ].at< unsigned short >( qPixel ) ) );
                                    pt2.x = std::min( int( pt2.x ), vImg[ 0 ].cols - 1 );

                                    pt2.y = std::max( 0, int( qPixel.y + node->data[ 3 ] * 1000.f/vImg[ 7 ].at< unsigned short >( qPixel ) ) );
                                    pt2.y = std::min( int( pt2.y ), vImg[ 0 ].rows - 1 );

                                    // convert points to 3D
                                    cv::Point3f pt1_real = CRPixel::P3toR3( pt1, img_center, vImg[ 7 ].at< unsigned short >( pt1 )/1000.f );
                                    cv::Point3f pt2_real = CRPixel::P3toR3( pt2, img_center, vImg[ 7 ].at< unsigned short >( pt2 )/1000.f );
                                    cv::Point3f qPoint   = CRPixel::P3toR3( qPixel, img_center, vImg[ 7 ].at< unsigned short >(qPixel )/1000.f );

                                    // create coordinate system TSC

                                    Surfel::calcQueryPoint2CameraTransformation(pt1_real, pt2_real, qPoint, qn, transformationMatrixQuery1C, transformationMatrixQuery2C);

                                    // convert matrix to quternion and multiply it with TOS
                                    Eigen::Matrix3f rotationMatrixQ1C = transformationMatrixQuery1C.block<3,3>(0,0);
                                    Eigen::Matrix3f rotationMatrixQ2C = transformationMatrixQuery2C.block<3,3>(0,0);

                                    Eigen::Quaternionf QuaternionQ1C(rotationMatrixQ1C);
                                    Eigen::Quaternionf QuaternionQ2C(rotationMatrixQ2C);

                                    float alpha1 = L->alpha[cNr][index][d].first;
                                    float alpha2 = L->alpha[cNr][index][d].second;

                                    Eigen::Quaternionf alphaInverseRotation1( cos((-alpha1)/2.f), 0.f, 0.f, sin((-alpha1)/2.f) );
                                    Eigen::Quaternionf alphaInverseRotation2( cos((-alpha2)/2.f), 0.f, 0.f, sin((-alpha2)/2.f) );

                                    Eigen::Quaternionf QuaternionO1C = QuaternionQ1C * alphaInverseRotation1 * L->vPose[cNr][index].first;
                                    Eigen::Quaternionf QuaternionO2C = QuaternionQ2C * alphaInverseRotation2 * L->vPose[cNr][index].second;

                                    // for first offset
                                    if( !(isnan( QuaternionO1C.coeffs()[ 0 ] ) || isnan( QuaternionO1C.coeffs()[ 1 ] ) || isnan( QuaternionO1C.coeffs()[ 2 ] ) || isnan( QuaternionO1C.coeffs()[ 3 ] ) ) ){

                                        QuaternionO1C.normalize();
                                        qMean.push_back(QuaternionO1C);


                                        int qx = std::max( 0,  int( ( QuaternionO1C.x() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qx = std::min( qx, step - 1 );
                                        int qy = std::max( 0,  int( ( QuaternionO1C.y() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qy = std::min( qy, step - 1 );
                                        int qz = std::max( 0,  int( ( QuaternionO1C.z() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qz = std::min( qz, step - 1 );
                                        int qw = std::max( 0,  int( ( QuaternionO1C.w() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qw = std::min( qw, step - 1 );

                                        // add weight of the hough image
                                        poseHoughSpace[ qz ].at< float >( qy, qx ) += weight_;


                                    }

                                    // for second offset
                                    if( !(isnan( QuaternionO2C.coeffs()[ 0 ] ) || isnan( QuaternionO2C.coeffs()[ 1 ] ) || isnan( QuaternionO2C.coeffs()[ 2 ] ) || isnan( QuaternionO2C.coeffs()[ 3 ] ) ) ){

                                        QuaternionO1C.normalize();
                                        qMean.push_back(QuaternionO2C);


                                        int qx = std::max( 0,  int( ( QuaternionO2C.x() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qx = std::min( qx, step - 1 );
                                        int qy = std::max( 0,  int( ( QuaternionO2C.y() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qy = std::min( qy, step - 1 );
                                        int qz = std::max( 0,  int( ( QuaternionO2C.z() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qz = std::min( qz, step - 1 );
                                        int qw = std::max( 0,  int( ( QuaternionO2C.w() + 1.f ) * (float)step / 2.f - 1 ) );
                                        qw = std::min( qw, step - 1 );

                                        poseHoughSpace[ qz ].at< float >( qy, qx )+= weight_;


                                    }
                                    subsample_count++;

                                }// end of all the node tests

                            }//end of votes voted for object center

                        } // end of trees

                    } //end of cx

                }  // end of cy

            } // end of scale

            //smooth the accumulator and find the peak

            Eigen::Quaternionf qfinalOC;

            float poseScore;
            detectMaxima(poseHoughSpace, qfinalOC, step, poseScore);
            Eigen::Matrix3f finalOC = Eigen::Matrix3f(qfinalOC);

            if(addPoseScore)
                candidates[cand].weight += poseScore;

            if(0){
                // estimate axis of symmetry
                axisOfSymmetry(qMean, qfinalOC);

            }

            //            cout<<"mean rotation\n" << finalOC <<endl;
            Eigen::Matrix4f tempOC = Eigen::Matrix4f::Identity();
            tempOC.block<3,3>(0,0) = finalOC;
            tempOC.block<3,1>(0,3) = Eigen::Vector3f(oCenter_real.x, oCenter_real.y, oCenter_real.z);

            candPoses.push_back(tempOC) ;

            candidates[cand].coordinateSystem = tempOC;

            // debug
            if(0){
                std::vector<Eigen::Quaternionf> rotation;
                Eigen::Quaternionf rotation1, rotation2, rotation3, rotation4;
                float  angle1 = PI/2.f, angle2 = PI, angle3 = PI/4.f, angle4 = 3.f*PI/4.f;
                Eigen::Vector3f v(0.f ,0.f, 1.f);

                Eigen::Quaternionf rotation0;
                rotation0.setIdentity();
                rotation1 = Eigen::Quaternionf( cos(angle1/2.f), 0.f, 0.f, sin(angle1/2.f));
                rotation2 = Eigen::Quaternionf( cos(angle2/2.f), 0.f, 0.f, sin(angle2/2.f));
                rotation3 = Eigen::Quaternionf( cos(angle3/2.f), 0.f, 0.f, sin(angle3/2.f));
                rotation4 = Eigen::Quaternionf( cos(angle4/2.f), 0.f, 0.f, sin(angle4/2.f));

                Eigen::Matrix4f tempQ = Eigen::Matrix4f::Identity();
                tempQ.block<3,3>(0,0) = Eigen::Matrix3f(rotation0);
                tempQ.block<3,1>(0,3) = v;

                Eigen::Matrix4f tempQ1 = Eigen::Matrix4f::Identity();
                tempQ1.block<3,3>(0,0) = Eigen::Matrix3f(rotation1);
                tempQ1.block<3,1>(0,3) = v;

                rotation.push_back(rotation0);
                rotation.push_back(rotation1);
                rotation.push_back(rotation2);
                rotation.push_back(rotation3);
                rotation.push_back(rotation1);
                Eigen::Quaternionf rotation5 = quatInterp(qMean);

                cout<<"mean rotation\n" << Eigen::Matrix3f(rotation5)<<endl;
                Eigen::Matrix4f tempQ2 = Eigen::Matrix4f::Identity();
                tempQ2.block<3,3>(0,0) = Eigen::Matrix3f(rotation5);
                tempQ2.block<3,1>(0,3) = v;
                //                Surfel::addCoordinateSystem(tempQ2, viewer, "4");
            }

        }

        if( DEBUG ){

            // get rgbimg and depthImg
            cv::Mat depthImg = vImg[7];

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
            Surfel::imagesToPointCloud( depthImg, img, cloud );

            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
            viewer->setBackgroundColor(1,1,1);
            viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.5, "sample cloud");
            viewer->addCoordinateSystem(0.1, 0.f, 0.f, 0.f, 0.f);

            for(int trnNr = 0; trnNr < candPoses.size(); trnNr++){
                stringstream ss; ss << trnNr;
                Surfel::addCoordinateSystem(candPoses[trnNr] , viewer, ss.str() );
            }

            while (!viewer->wasStopped ()){
                viewer->spinOnce (100);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
            }
        }
    }
}


void CRForestDetector::detectCenterPeaks(std::vector<Candidate >& candidates, const std::vector<std::vector<cv::Mat> >& imgDetect, const std::vector<cv::Mat>& vImgAssign, const std::vector< std::vector< std::vector< std::vector<std::vector< std::pair< cv::Point, int > > > > > >& voterImages, const  cv::Mat& depthImg, const cv::Mat& img, const Parameters& param, int this_class){

    //    p.scales, p.max_candidates, p.kernel_width[0], p.kernel_width[1], true, -1, this_class, p.thresh_detection, p.DEBUG;
    candidates.clear();

    // this is just to access a non-empty detect image for getting sizes and so on
    int default_class = 0;
    if ((this_class >= 0) )
        default_class = this_class;

    // for third dimension
    int kernelSize = 3;
    float std = 0.1;

    unsigned int nScales = param.scales.size();

    //define variables
    std::vector< cv::Mat > smoothAcc( nScales );
    std::vector< cv::Mat > smoothAccTemp( nScales );
    std::vector< cv::Mat > dilatedImg( nScales ), comp( nScales);
    std::vector< cv::Mat > localMax(nScales);
    cv::Mat gaussKernel = cv::getGaussianKernel( kernelSize, std, CV_32F );


    for ( unsigned int cNr = 0; cNr < imgDetect.size(); cNr++ ){

        if ( ( this_class >= 0 ) && ( this_class != (int)cNr ) )
            continue;

        if( 0 )
            for(unsigned int scNr = 0; scNr < nScales; scNr++ ){
                cv::imshow("hough", imgDetect[cNr][scNr]); cv::waitKey(0);
            }

        for(unsigned int scNr = 0; scNr< nScales; scNr++){

            // smoothing the accumulator matrix
            int adapKwidth = int( param.kernel_width[0] * param.scales[scNr] / 2.0f ) * 2 + 1;
            float adapKstd  = param.kernel_width[1] * param.scales[scNr];

            smoothAcc[scNr] = imgDetect[cNr][scNr].clone();
            cv::GaussianBlur( smoothAcc[scNr], smoothAccTemp[scNr], cv::Size( adapKwidth, adapKwidth ), adapKstd ); // smooth in direction of x and y
            smoothAcc[scNr] = smoothAccTemp[scNr];
        }


        // Smoothing in third dimension
        for( int r = 0; r < smoothAcc[ 0 ].rows; r++ ){
            for( int c = 0; c< smoothAcc[ 0 ].cols; c++ ){

                for( int scNr = 0; scNr < ( int )nScales; scNr++ ){
                    int scBegin = std::max( 0, scNr - int( kernelSize / 2 ) );
                    int scEnd = std::min( scNr + int( kernelSize / 2 ) + 1, ( int )nScales );

                    float convSum = 0;
                    int k = 0;

                    for( int td = scBegin; td < scEnd; td++, k++ ){

                        convSum += smoothAcc[ td ].at< float >( r, c ) * gaussKernel.at< float >( k );

                    }

                    smoothAccTemp[ scNr ].at< float >( r, c ) = convSum;
                }
            }
        }

        if( 0 )
            for( unsigned int scNr = 0; scNr < nScales; scNr++ ){
                cv::imshow( "smooth_hough_scales", smoothAcc[ scNr ]); cv::waitKey( 0 );
            }


        // find local maximum


        //............ dilate the image.........

        //dilate in x and y direction
        for( unsigned int scNr = 0; scNr < nScales; scNr++ ){

            smoothAcc[scNr] = smoothAccTemp[scNr].clone();
            int adapKwidth = int( param.kernel_width[0] * param.scales[scNr] / 2.0f ) * 2 + 1;
            cv::Mat kernel = cv::Mat::ones( adapKwidth, adapKwidth, CV_32FC1 );
            cv::dilate(smoothAcc[ scNr ], dilatedImg[ scNr ], kernel );

        }

        if( 0 )
            for( unsigned int scNr = 0; scNr < nScales; scNr++ ){
                cv::imshow( "dilated_hough", dilatedImg[ scNr ] ); cv::waitKey( 0 );
            }

        // dilate it in third dimension
        for(int r = 0; r < dilatedImg[ 0 ].rows; r++ ){
            for( int c = 0; c < dilatedImg[ 0 ].cols; c++){

                for( int scNr = 0; scNr < ( int )nScales; scNr++ ){
                    int scBegin = std::max( 0, scNr - int( kernelSize / 2 ) );
                    int scEnd = std::min( scNr + int(kernelSize/2) + 1, ( int )nScales );

                    float max_val = -1.f;

                    for( int td = scBegin; td < scEnd; td++ ){

                        max_val = std::max(max_val, dilatedImg[td].at<float>( r, c ) );
                    }

                    dilatedImg[scNr].at< float >( r, c ) = max_val;
                }
            }
        }

        if( 0 )
            for(unsigned int scNr = 0; scNr < nScales; scNr++ ){
                cv::imshow( "dilated_hough_scales", dilatedImg[ scNr ]); cv::waitKey( 0 );
            }


        for(unsigned int scNr = 0; scNr < nScales; scNr++ ){

            cv::compare( smoothAcc[ scNr ], dilatedImg[ scNr ], comp[ scNr ], CV_CMP_EQ ); //cv::imshow("compare", comp[ scNr ]); cv::waitKey(0);
            cv::multiply( smoothAcc[ scNr ], comp[ scNr ], localMax[ scNr ], 1/255.f, CV_32F ); //cv::imshow("localmax", localMax); cv::waitKey(0);
        }

        if( 0 )
            for(unsigned int scNr = 0; scNr < nScales; scNr++ ){
                cv::imshow( "localMax", localMax[ scNr ]); cv::waitKey( 0 );
            }


        // each candidate is a six element vector weight, x, y, scale, class, ratio
        int candNr = 0;
        bool goodCandidate;

        for ( int count = 0; count < param.max_candidates; count++ ){// count can go until infinity

            bool flag = false;
            std::vector< cv::Point > max_loc_temp( nScales );
            std::vector< double > max_val_temp( nScales );

            Candidate max_position;// weight, x, y, scNr, cNr, rNr, bb.x, bb.y, bb.z
            goodCandidate = 1;
            // detect the maximum

            for( unsigned int scNr = 0; scNr < nScales; scNr++)
                cv::minMaxLoc( localMax[scNr], 0, &max_val_temp[scNr], 0, &max_loc_temp[scNr], cv::Mat() );

            std::vector< double >::iterator it;
            it = std::max_element( max_val_temp.begin(),max_val_temp.end() );
            int max_index = std::distance( max_val_temp.begin(), it );

            double max_value = max_val_temp[max_index];
            if ( ( max_value >= param.thresh_detection) && (max_value > max_position.weight) ){
                flag = true;
                max_position.weight = max_value;

                localMax[max_index].at<float>(max_loc_temp[max_index].y, max_loc_temp[max_index].x) = 0.f;

                // take average to get the sub depth accuracy
                int spatialRadius = param.kernel_width[0];
                int scaleRadius = 1;
                float score_sum = 0;
                int num_votes = 0;
                float avgScale = 0, avgWeight = 0, avgX = 0, avgY = 0;
                int minS = std::max(0, max_index - scaleRadius);
                int maxS = std::min(int(param.scales.size()), max_index + scaleRadius);

                for(unsigned int trNr = 0; trNr < vImgAssign.size(); trNr++){

                    for( int wscale = minS; wscale < maxS; wscale++ ){

                        int minX = std::max( 0, int (max_loc_temp[ max_index ].x - spatialRadius * param.scales[wscale]) );
                        int maxX = std::min( imgDetect[ cNr ][ 0 ].cols, int(max_loc_temp[ max_index ].x + spatialRadius * param.scales[wscale] ));
                        int minY = std::max( 0, int( max_loc_temp[ max_index ].y - spatialRadius * param.scales[wscale] ));
                        int maxY = std::min( imgDetect[ cNr ][ 0 ].rows, int(max_loc_temp[ max_index ].y + spatialRadius * param.scales[wscale] ));

                        for( int wx = minX; wx < maxX ; wx++ ){
                            for(int wy = minY; wy < maxY ; wy++ ){

                                score_sum  = score_sum + smoothAcc[ wscale ].at<float>(wy,wx);

                                // averaging the scale
                                avgWeight += smoothAcc[ wscale ].at<float>(wy,wx);
                                avgScale += param.scales[wscale] * smoothAcc[ wscale ].at<float>(wy,wx) ;
                                avgX += wx * smoothAcc[ wscale ].at<float>(wy,wx) ;
                                avgY += wy * smoothAcc[ wscale ].at<float>(wy,wx) ;

                                // averaging the bounding box size

                                unsigned int total_votes = voterImages[ trNr ][ wscale ][ wy ][ wx ].size();
                                num_votes += total_votes;



                                //                                for( unsigned int count = 0; count < total_votes; count++ ){
                                //                                    int voteIndex = voterImages[ trNr ][ wscale][ wy][ wx][count].second;
                                //                                    cv::Point2f pt = voterImages[ trNr ][ wscale][ wy][ wx][count].first;
                                //                                    LeafNode* tmp = crForest->vTrees[ trNr ]->getLeaf(vImgAssign[ trNr ].at<float>(pt));
                                //                                    //                                    cv::Point3f pt_real = CRPixel::P3toR3(pt, imgCenter, depthImg.at<unsigned short> (pt)/1000.f);

                                //                                    //                                    cv::Point3f objCenterPoint = pt_real - ( tmp->vCenter[cNr][voteIndex] );
                                //                                    //                                    cv::Point2f objCenterPixel;
                                //                                    //                                    float objCenterdepth;

                                //                                    //                                    CRPixel::R3toP3( objCenterPoint, imgCenter, objCenterPixel, objCenterdepth );

                                //                                    //                                    float tempScale  = 1.f/objCenterdepth;
                                //                                    //                                    if( int( objCenterPixel.y ) >= 0 && int( objCenterPixel.y ) < localMax[ 0 ].rows && int(objCenterPixel.x) >= 0 && int(objCenterPixel.x) < localMax[ 0 ].cols){
                                //                                    //                                        mX.push_back(objCenterPixel.x);
                                //                                    //                                        mY.push_back(objCenterPixel.y);
                                //                                    //                                        mScale.push_back(tempScale);
                                //                                    //                                    }

                                //                                    float tmpWidth = tmp->bbSize3D[cNr][voteIndex].x;// * tmp->vPrLabel[cNr] ;
                                //                                    float tmpHeight = tmp->bbSize3D[cNr][voteIndex].y;// * tmp->vCenterWeights[cNr][voteIndex] * tmp->vPrLabel[cNr] ;
                                //                                    float tmpDepth = tmp->bbSize3D[cNr][voteIndex].z;
                                //                                    //                                    float fwidth = std::max( std::abs(tmpWidth*cos(rotAngle)), std::abs(tmpHeight*sin(rotAngle)));
                                //                                    //                                    float fheight = std::max( std::abs(tmpHeight*cos(rotAngle)), std::abs(tmpWidth*sin(rotAngle)));
                                //                                    //                                    fwidth = fwidth*(1000.f/depthImg.at<unsigned short> (pt))/1.5f;
                                //                                    //                                    fheight = fheight*(1000.f/depthImg.at<unsigned short> (pt))/1.5f;

                                //                                    bbWidth.push_back(tmpWidth);
                                //                                    bbHeight.push_back(tmpHeight);
                                //                                    bbDepth.push_back(tmpDepth);
                                //                                }



                            }
                        }
                    }
                }

                if( avgWeight > 0) {
                    max_position.center.x = avgX/avgWeight;
                    max_position.center.y = avgY/avgWeight;
                    max_position.scale = avgScale/avgWeight;
                }else{
                    max_position.center.x = float(max_loc_temp[max_index].x);
                    max_position.center.y = float(max_loc_temp[max_index].y);
                    max_position.scale = param.scales[max_index];
                }


                //                max_position[ 0 ] = score_sum/num_pixels;

                //                // Mean shift for x y scale
                //                float meanshift_x = 0.f;
                //                float weightsum_x =0.f;

                //                float meanshift_y = 0.f;
                //                float weightsum_y =0.f;

                //                float meanshift_scale = 0.f;
                //                float weightsum_scale =0.f;

                //                int midPoint_spatial = mX.size()/2;

                //                if(midPoint_spatial !=0){

                //                    std::sort(mX.begin(), mX.end());
                //                    std::sort(mY.begin(), mY.end());
                //                    std::sort(mScale.begin(), mScale.end());

                //                    float median_x = mX[midPoint_spatial];
                //                    float median_y = mY[midPoint_spatial];
                //                    float median_scale = mScale[midPoint_spatial];

                //                    float windowScale_scale = 0.3;
                //                    float windowScale_xy = 5;

                //                    // local meanshift from current width and height estimate
                //                    const float scaleMeanShiftWindowSize =  windowScale_scale;// * median_width;
                //                    const float XYMeanShiftWindowSize =  windowScale_xy;// * median_height;

                //                    for( unsigned int f = 0; f < mX.size(); f++ ) {
                //                        if( fabsf( mX[f] - median_x ) < XYMeanShiftWindowSize ) {
                //                            meanshift_x += mX[f];
                //                            weightsum_x += 1.f;
                //                        }
                //                        if( fabsf( mY[f] - median_y ) < XYMeanShiftWindowSize ) {
                //                            meanshift_y += mY[f];
                //                            weightsum_y += 1.f;
                //                        }
                //                        if( fabsf( mScale[f] - median_scale ) < scaleMeanShiftWindowSize ) {
                //                            meanshift_scale += mScale[f];
                //                            weightsum_scale += 1.f;
                //                        }
                //                    }

                //                    if( weightsum_x > std::numeric_limits<float>::epsilon() )
                //                        meanshift_x = meanshift_x / weightsum_x;
                //                    else
                //                        meanshift_x = median_x;

                //                    if( weightsum_y > std::numeric_limits<float>::epsilon() )
                //                        meanshift_y = meanshift_y / weightsum_y;
                //                    else
                //                        meanshift_y = median_y;

                //                    if( weightsum_scale > std::numeric_limits<float>::epsilon() )
                //                        meanshift_scale = meanshift_scale / weightsum_scale;
                //                    else
                //                        meanshift_scale = median_scale;

                //                    if( meanshift_x > 640 )
                //                        cout<<"there is a problem"<<endl;

                //                    max_position.center.x = meanshift_x;
                //                    max_position.center.y = meanshift_y;
                //                    max_position.scale = meanshift_scale;

                //                }else{


                //                }

                // Mean shift bounding box dimensions

                //                double meanshift_width = 0.f;
                //                double weightsum_width = 0.f;
                //                double meanshift_height = 0.f;
                //                double weightsum_height = 0.f;
                //                double meanshift_depth = 0.f;
                //                double weightsum_depth = 0.f;

                //                int midPoint = bbWidth.size()/2;
                //                if(midPoint !=0){

                //                    std::sort(bbWidth.begin(), bbWidth.end());
                //                    std::sort(bbHeight.begin(), bbHeight.end());
                //                    std::sort(bbDepth.begin(), bbDepth.end());


                //                    float median_width = bbWidth[midPoint];
                //                    float median_height = bbHeight[midPoint];
                //                    float median_depth = bbDepth[midPoint];

                //                    float windowScale = 0.5;

                //                    // local meanshift from current width and height estimate
                //                    const float bbWidthMeanShiftWindowSize =  windowScale * median_width;
                //                    const float bbHeightMeanShiftWindowSize =  windowScale * median_height;
                //                    const float bbDepthMeanShiftWindowSize =  windowScale * median_depth;

                //                    for( unsigned int f = 0; f < bbWidth.size(); f++ ) {
                //                        if( fabsf( bbWidth[f] - median_width ) < bbWidthMeanShiftWindowSize ) {
                //                            meanshift_width += bbWidth[f];
                //                            weightsum_width += 1.f;
                //                        }
                //                        if( fabsf( bbHeight[f] - median_height ) < bbHeightMeanShiftWindowSize ) {
                //                            meanshift_height += bbHeight[f];
                //                            weightsum_height += 1.f;
                //                        }
                //                        if( fabsf( bbDepth[f] - median_depth ) < bbDepthMeanShiftWindowSize ) {
                //                            meanshift_depth += bbDepth[f];
                //                            weightsum_depth += 1.f;
                //                        }
                //                    }
                //                    if( weightsum_width > std::numeric_limits<float>::epsilon() )
                //                        meanshift_width = meanshift_width / weightsum_width;
                //                    else
                //                        meanshift_width = median_width;
                //                    if( weightsum_height > std::numeric_limits<float>::epsilon() )
                //                        meanshift_height = meanshift_height / weightsum_height;
                //                    else
                //                        meanshift_height = median_height;
                //                    if( weightsum_depth > std::numeric_limits<float>::epsilon() )
                //                        meanshift_depth = meanshift_depth / weightsum_depth;
                //                    else
                //                        meanshift_depth = median_depth;

                //                    max_position.bbSize.x = meanshift_width;
                //                    max_position.bbSize.y = meanshift_height;
                //                    max_position.bbSize.z = meanshift_depth;
                //                    goodCandidate = 1;

                //                }else{

                //                    cout<< "this happened" << endl;

                //                    goodCandidate = 0;
                //                    //                    //                      continue;
                //                    //                    max_position.bbSize.x = 2 * kernel_width / 525.f;
                //                    //                    max_position.bbSize.y = 2 * kernel_width / 525.f;
                //                    //                    max_position.bbSize.z = 2 * kernel_width / 525.f;
                //                }


                max_position.c = cNr;
                max_position.r = 0;

                max_position.bbSize = param.bbSize;

                if(num_votes < 10)
                    goodCandidate = 0;
            }

            if (!flag)
                break;
            else
                candNr++;

            // push the candidate in the stack
            if( goodCandidate )
                candidates.push_back( max_position );

            if (0){
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
                Surfel::imagesToPointCloud( depthImg, img, cloud );
                //Surfel::houghPointCloud( imgDetect[cNr],  scales, cloud, max_value );

                boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
                viewer->setBackgroundColor(1,1,1);
                viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
                while (!viewer->wasStopped ()){
                    viewer->spinOnce (100);
                    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
                }
            }

            float min_dimension = 525.f * std::min(std::min (max_position.bbSize.x , max_position.bbSize.y) , max_position.bbSize.z);

            //            int scBegin = std::max( 0, max_index - int( kernelSize / 2 ) );
            //            int scEnd = std::min( max_index + int( kernelSize / 2 )  , ( int )nScales );

            for( int scNr = 0; scNr < param.scales.size(); scNr++){

                // remove the maximum region with the supporting kernel width
                int adapKwidth = int( min_dimension * param.scales[ scNr ] / 2.f);
                int adapKheight = int( min_dimension * param.scales[ scNr ] / 2.f );

                int cx = int( max_position.center.x );
                int cy = int( max_position.center.y );

                int x = std::max( 0, cx - adapKwidth );
                int y = std::max( 0, cy - adapKheight );

                int rwidth =  std::max( 1, std::min( cx + adapKwidth, localMax[ cNr ].cols - 1 ) - x + 1 );
                int rheight =  std::max( 1, std::min( cy + adapKwidth, localMax[ cNr ].rows - 1 ) - y + 1 );

                if ( max_position.c >= 0 && cNr != max_position.c )
                    continue;

                if( 0 )
                    cv::imshow( "original", localMax[scNr] );


                cv::Mat ROI = localMax[ scNr ](cv::Rect( x, y, rwidth, rheight ) );
                ROI = cv::Scalar( 0 );

                if( 0 ){

                    double min, max;
                    cv::Mat tmp;
                    cv::Point max_loc, min_loc;

                    cv::minMaxLoc( localMax[scNr], &min, &max, &max_loc, &min_loc, cv::Mat());
                    cv::convertScaleAbs( localMax[scNr], tmp, 255/max );
                    cv::Mat kernel = cv::Mat::ones( 10, 10, CV_32FC1 );
                    cv::dilate( tmp, tmp, kernel );// cv::imshow("dilate", dilatedImg); cv::waitKey(0);
                    cv::imshow("removed", tmp); cv::waitKey(0);

                }

            }// for each scale

        }// for each candidate

    }// for each class

}

// given the cluster assignment images, we are voting into the voting space vImgDetect
void CRForestDetector::voteForCenter(const std::vector<cv::Mat>& vImgAssign, std::vector< std::vector<cv::Mat> >& vImgDetect, const  cv::Mat& depthImg, std::vector< std::vector< std::vector< std::vector< std::vector< std::pair< cv::Point, int > > > > > >& voterImages, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::vector<float>& scales, int& this_class, cv::Rect* focus, const float& prob_threshold, const std::vector<cv::Mat>& classProbs, bool addPoseInformation,  bool addScaleInformation ){


    // vImgDetect are all initialized before

    if (vImgAssign.size()<1)
        return;


    unsigned ntrees = vImgAssign.size();

    unsigned int nScales = scales.size();


    // Initialize voter Images to use it for pose voting
    voterImages.resize(ntrees);

    for ( unsigned int trNr = 0; trNr < ntrees; trNr++ ){ // for all the trees

        voterImages[ trNr ].resize( nScales );

        for ( unsigned int scNr = 0; scNr < nScales; scNr++ ){ // for all scale

            voterImages[ trNr ][ scNr ].resize( vImgAssign[ trNr ].rows );

            for ( int y = 0 ; y < vImgAssign[ trNr ].rows; y++ ){ // for all pixel y coordinates

                voterImages[ trNr ][ scNr ][ y ].resize( vImgAssign[ trNr ].cols );

            }
        }
    }


    for ( unsigned int trNr = 0; trNr < ntrees; trNr++ ){

        cv::Point2f imgCenterPixel( vImgAssign[ trNr ].cols/2.f, vImgAssign[ trNr ].rows/2.f );

        for ( int y = 0 ; y < vImgAssign[ trNr ].rows; y++ ){

            for ( int x = 0; x < vImgAssign[ trNr ].cols; x++ ){
                // get the leaf_id
                if( vImgAssign[ trNr ].at< float >( y, x ) < 0 )
                    continue;

                cv::Point2f qPixel(x,y);

                float qScale;
                if( depthImg.at< unsigned short >( y, x ) == 0 )
                    continue; //qScale = FLT_MAX;
                else
                    qScale = 1000.f/(float)depthImg.at< unsigned short >( qPixel );

                cv::Point3f qPoint = CRPixel::P3toR3(qPixel, imgCenterPixel, 1/qScale);
                LeafNode* tmp = crForest->vTrees[ trNr ]->getLeaf(vImgAssign[ trNr ].at<float>(qPixel));

                for (unsigned int cNr = 0; cNr < vImgDetect.size(); cNr++){

                    if ((this_class >= 0 ) && (this_class != (int)cNr)) // the voting should be done on a single class only
                        continue;

                    bool condition;
                    if (prob_threshold < 0)
                        condition = (Class_id[ trNr ][ cNr ]> 0 && tmp->vPrLabel[ cNr ]*Class_id[ trNr ].size()>1);
                    else{
                        condition = ( Class_id[ trNr ][ cNr ]> 0 &&  classProbs[ cNr ].at<float>( y, x ) > prob_threshold );
                    }

                    if ( condition ){

                        //                        int addPoseInformation = 0; // TODO:: take this variable out in config file
                        int nanData = 0;

                        const pcl::Normal& qNormal = normals->at(qPixel.x, qPixel.y);

                        if( isnan(qNormal.normal_x) || isnan(qNormal.normal_y) || isnan(qNormal.normal_z) ){

                            addPoseInformation = 0;

                        }else{

                            bool onceVoted = 0;

                            if( addPoseInformation ){

                                float w = tmp->vPrLabel[ cNr ] / ntrees;
                                float wScale = 1;

                                std::vector<int> nodeIDs(tmp->depth);
                                InternalNode* node = crForest->getNode( trNr, tmp->parent );


                                for(int d = 0; d < tmp->depth; d++){

                                    nodeIDs[tmp->depth - 1 - d] = node->idN;
                                    node = crForest->getNode( trNr, node->parent );

                                }

                                for( int d = 0; d < tmp->depth; d++ ){

                                    InternalNode* tmpNode = crForest->getNode( trNr, nodeIDs[d] );
                                    cv::Point2f sPixel1, sPixel2;

                                    sPixel1.x = std::max( 0, int( qPixel.x + tmpNode->data[ 0 ] * 1000.f/depthImg.at< unsigned short >( qPixel ) ) );
                                    sPixel1.x = std::min( int( sPixel1.x ), depthImg.cols - 1 );

                                    sPixel1.y = std::max( 0, int( qPixel.y + tmpNode->data[ 1 ] * 1000.f/depthImg.at< unsigned short >( qPixel ) ) );
                                    sPixel1.y = std::min( int( sPixel1.y ), depthImg.rows - 1 );

                                    sPixel2.x = std::max( 0, int( qPixel.x + tmpNode->data[ 2 ] * 1000.f/depthImg.at< unsigned short >( qPixel ) ) );
                                    sPixel2.x = std::min( int( sPixel2.x ), depthImg.cols - 1 );

                                    sPixel2.y = std::max( 0, int( qPixel.y + tmpNode->data[ 3 ] * 1000.f/depthImg.at< unsigned short >( qPixel ) ) );
                                    sPixel2.y = std::min( int( sPixel2.y ), depthImg.rows - 1 );

                                    cv::Point3f sPoint1 = CRPixel::P3toR3( sPixel1, imgCenterPixel, depthImg.at< unsigned short >( sPixel1 ) );
                                    cv::Point3f sPoint2 = CRPixel::P3toR3( sPixel2, imgCenterPixel, depthImg.at< unsigned short >( sPixel2 ) );

                                    Eigen::Matrix4f transformationFromQueryPixel1toCamera, transformationFromQueryPixel2toCamera;
                                    Surfel::calcQueryPoint2CameraTransformation(sPoint1, sPoint2, qPoint, qNormal, transformationFromQueryPixel1toCamera, transformationFromQueryPixel2toCamera);


                                    float alpha1, alpha2;

                                    for(int trainingPixels = 0; trainingPixels < tmp->vCenter[cNr].size(); trainingPixels++){ // loop on number of training pixels saved in this leaf

                                        float itW = tmp->vCenterWeights[cNr][trainingPixels];

                                        Eigen::Vector3f qDisVector1 = tmp->QdisVector[cNr][trainingPixels].first;
                                        Eigen::Vector3f qDisVector2 = tmp->QdisVector[cNr][trainingPixels].second;


                                        alpha1 = tmp->alpha[cNr][trainingPixels][d].first;
                                        alpha2 = tmp->alpha[cNr][trainingPixels][d].second;

                                        Eigen::Quaternionf alphaInverseRotation1(cos((-alpha1) / 2.f),0.f,0.f,sin((-alpha1) / 2.f));
                                        Eigen::Quaternionf alphaInverseRotation2(cos((-alpha2) / 2.f),0.f,0.f,sin((-alpha2) / 2.f));


                                        Eigen::Vector3f disVector1 = transformationFromQueryPixel1toCamera.block<3,3>(0,0)* Eigen::Matrix3f(alphaInverseRotation1) * qDisVector1;
                                        Eigen::Vector3f disVector2 = transformationFromQueryPixel2toCamera.block<3,3>(0,0)* Eigen::Matrix3f(alphaInverseRotation2) * qDisVector2;

                                        int sCount = 0;

                                        if( isnan(disVector1[0]) && isnan(disVector1[1]) && isnan(disVector1[2]) && isnan(disVector2[0]) && isnan(disVector2[1]) && isnan(disVector2[2])){
                                            nanData = true;

                                        }else{

                                            // for first surfel
                                            cv::Point3f disVectorPoint1(disVector1[0], disVector1[1], disVector1[2]);
                                            cv::Point3f objCenterPoint1 = qPoint + disVectorPoint1;
                                            cv::Point2f objCenterPixel;
                                            float objCenterdepth;

                                            if(objCenterdepth < 0.f || objCenterdepth > 10.f){
                                                nanData = 1;
                                                sCount++;
                                            }else{

                                                CRPixel::R3toP3( objCenterPoint1, imgCenterPixel, objCenterPixel, objCenterdepth );

                                                int scNr = int(( scales.size() / ( scales.back() - scales.front() )) / objCenterdepth) - 1; // for scale ranges from (0,2] in 10 equal interval
                                                if( scNr < 0)
                                                    scNr = 0;
                                                if( scNr > int( scales.size() - 1 ))
                                                    scNr = scales.size() - 1;

                                                if(addScaleInformation)
                                                    wScale = 1/std::pow(scales[scNr],2);

                                                if( int(objCenterPixel.y) >= 0 && int(objCenterPixel.y) < vImgDetect[ cNr ][ scNr ].rows && int(objCenterPixel.x) >= 0 && int(objCenterPixel.x) < vImgDetect[ cNr ][ scNr ].cols ) {
                                                    vImgDetect[ cNr ][ scNr ].at< float >( int(objCenterPixel.y), int(objCenterPixel.x)) += ( itW ) * w/(tmp->depth * 2.f) * wScale;

                                                    if(!onceVoted){
                                                        voterImages[ trNr ][scNr][ int(objCenterPixel.y ) ][ int(objCenterPixel.x ) ].push_back( std::pair< cv::Point, int > (cv::Point( x, y ), trainingPixels ));
                                                        onceVoted = 1;
                                                    }
                                                }
                                            }

                                            // for second surfel

                                            cv::Point3f disVectorPoint2( disVector2[0], disVector2[1], disVector2[2] );
                                            cv::Point3f objCenterPoint2 = qPoint + disVectorPoint2;
                                            CRPixel::R3toP3( objCenterPoint2, imgCenterPixel, objCenterPixel, objCenterdepth );

                                            if(objCenterdepth < 0.f || objCenterdepth > 10.f){
                                                nanData = 1;
                                                sCount++;
                                            }else{
                                                int scNr = int(( scales.size() / ( scales.back() - scales.front() )) / objCenterdepth) - 1; // for scale ranges from (0,2] in 10 equal interval
                                                if( scNr < 0 )
                                                    scNr = 0;
                                                if( scNr > int(scales.size() - 1 ))
                                                    scNr = scales.size() - 1;

                                                if(addScaleInformation)
                                                    wScale = 1/std::pow(scales[scNr],2);

                                                if( int(objCenterPixel.y) >= 0 && int(objCenterPixel.y) < vImgDetect[ cNr ][ scNr ].rows && int(objCenterPixel.x) >= 0 && int(objCenterPixel.x) < vImgDetect[ cNr ][ scNr ].cols ) {
                                                    vImgDetect[ cNr ][ scNr ].at< float >( int(objCenterPixel.y), int(objCenterPixel.x)) += ( itW ) * w/(tmp->depth * 2.f) *wScale;

                                                    if(!onceVoted){
                                                        voterImages[ trNr ][scNr][ int(objCenterPixel.y ) ][ int(objCenterPixel.x ) ].push_back( std::pair< cv::Point, int > (cv::Point( x, y ), trainingPixels ));
                                                        onceVoted = 1;
                                                    }
                                                }
                                            }
                                        }

                                        if(nanData){

                                            nanData = false;
                                            cv::Point3f objCenterPoint = qPoint - ( tmp->vCenter[cNr][trainingPixels] );
                                            cv::Point2f objCenterPixel;
                                            float objCenterdepth;

                                            CRPixel::R3toP3( objCenterPoint, imgCenterPixel, objCenterPixel, objCenterdepth );

                                            int scNr = int(( scales.size() / ( scales.back() - scales.front() )) / objCenterdepth) - 1; // for scale ranges from (0,2] in 10 equal interval
                                            if (scNr < 0)
                                                scNr = 0;
                                            if(scNr > int(scales.size() - 1))
                                                scNr = scales.size() - 1;

                                            if(addScaleInformation)
                                                wScale = 1/std::pow(scales[scNr],1);//1/std::pow(scales[scNr],2);

                                            if ( focus==NULL ){
                                                if( int(objCenterPixel.y) >= 0 && int(objCenterPixel.y) < vImgDetect[ cNr ][ scNr ].rows && int(objCenterPixel.x) >= 0 && int(objCenterPixel.x) < vImgDetect[ cNr ][ scNr ].cols ) {
                                                    vImgDetect[ cNr ][ scNr ].at< float >( int(objCenterPixel.y), int(objCenterPixel.x)) += ( itW ) * w/(tmp->depth) * wScale;
                                                    if(!onceVoted){
                                                        voterImages[ trNr ][scNr][ int(objCenterPixel.y ) ][ int(objCenterPixel.x ) ].push_back( std::pair< cv::Point, int > (cv::Point( x, y ), trainingPixels ));
                                                        onceVoted = 1;
                                                    }
                                                }
                                            }else{
                                                if ( isInsideRect( focus, x, y) ){

                                                    vImgDetect[ cNr ][ scNr ].at< float >(int(objCenterPixel.y - focus->y) , int( objCenterPixel.x - focus->x )) += ( itW ) * w/(tmp->depth + sCount%2) * wScale ;
                                                    if(!onceVoted){
                                                        voterImages[ trNr ][scNr][ int(objCenterPixel.y ) ][ int(objCenterPixel.x) ].push_back( std::pair< cv::Point, int > (cv::Point( x, y ), trainingPixels ));
                                                        onceVoted = 1;
                                                    }
                                                }
                                            }

                                        }// end nondata if statement

                                    }
                                }
                            }
                        }

                        if(!addPoseInformation){

                            float w = tmp->vPrLabel[ cNr ] / ntrees;
                            float wScale = 1;
                            int sample_factor = 20;
                            int count = 0;
                            // vote for all points stored in a leaf
                            vector<float>::iterator itW = tmp->vCenterWeights[ cNr ].begin();
                            for( vector< cv::Point3f >::const_iterator it = tmp->vCenter[ cNr ].begin() ; it!=tmp->vCenter[ cNr ].end(); ++it, itW++ ){

                                int voteIndex = std::distance( tmp->vCenterWeights[cNr].begin(), itW );

                                cv::Point3f objCenterPoint = qPoint - ( *it );
                                cv::Point2f objCenterPixel;
                                float objCenterdepth;

                                CRPixel::R3toP3( objCenterPoint, imgCenterPixel, objCenterPixel, objCenterdepth );

                                int scNr = int(( scales.size() / ( scales.back() - scales.front() )) / objCenterdepth) - 1; // for scale ranges from (0,2] in 10 equal interval
                                if (scNr < 0)
                                    scNr = 0;
                                if(scNr > int(scales.size() - 1))
                                    scNr = scales.size() - 1;

                                if(addScaleInformation)
                                    wScale = 1.f/scales[scNr];//std::pow(scales[scNr],2);

                                if ( focus==NULL ){
                                    if( int(objCenterPixel.y) >= 0 && int(objCenterPixel.y) < vImgDetect[ cNr ][ scNr ].rows && int(objCenterPixel.x) >= 0 && int(objCenterPixel.x) < vImgDetect[ cNr ][ scNr ].cols ) {
                                        vImgDetect[ cNr ][ scNr ].at< float >( int(objCenterPixel.y), int(objCenterPixel.x)) += ( *itW ) * w * wScale;

                                        if( count % sample_factor == 0 )
                                            voterImages[ trNr ][scNr][ int(objCenterPixel.y ) ][ int(objCenterPixel.x ) ].push_back( std::pair< cv::Point, int > (cv::Point( x, y ), voteIndex ));
                                    }
                                }else{
                                    if ( isInsideRect( focus, x, y) ){

                                        vImgDetect[ cNr ][ scNr ].at< float >(int(objCenterPixel.y - focus->y) , int( objCenterPixel.x - focus->x )) += ( *itW ) * w * wScale;

                                        if( count % sample_factor == 0 )
                                            voterImages[ trNr ][scNr][ int(objCenterPixel.y ) ][ int(objCenterPixel.x) ].push_back( std::pair< cv::Point, int > (cv::Point( x, y ), voteIndex ));
                                    }
                                }

                                count++;
                            }
                        }
                    }
                }
            }
        }
    }
}

void CRForestDetector::detectObject(const cv::Mat &img, const cv::Mat &depthImg,  const vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::vector< cv::Mat >& vImgAssign, const std::vector<cv::Mat>& classProbs,  const Parameters& p, int this_class, std::vector<Candidate >& candidates){

    std::vector<std::vector<cv::Mat> > vImgDetect(crForest->GetNumLabels());
    std::vector< std::vector< std::vector< std::vector< std::vector< std::pair< cv::Point, int > > > > > > voterImages;

    for ( unsigned int cNr = 0; cNr < crForest->GetNumLabels(); cNr++ ){ // cNr = class  number
        if ( (this_class >= 0 ) && (this_class != (int)cNr) )
            continue;
        vImgDetect[cNr].resize(p.scales.size());
        for(unsigned int scNr = 0; scNr < p.scales.size(); scNr++){

            vImgDetect[cNr][scNr] = cv::Mat::zeros( vImgAssign[0].rows, vImgAssign[0].cols, CV_32FC1);
        }
    }

    // vote for object center in hough space
    int tstart = clock();
    voteForCenter( vImgAssign, vImgDetect, depthImg, voterImages, normals, p.scales, this_class, NULL, p.thresh_vote, classProbs, p.addPoseInformation, p.addScaleInformation);
    cout << "\t Time for voting for center..\t" << (double)(clock() - tstart)/CLOCKS_PER_SEC << " sec" << endl;

    if( p.DEBUG ){

        for ( unsigned int cNr = 0; cNr < crForest->GetNumLabels(); cNr++ ){ // cNr = class  number
            if ( (this_class >= 0 ) && (this_class != (int)cNr) )
                continue;

            pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGB > );
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
            Surfel::imagesToPointCloud( depthImg, img, cloud);

            Surfel::houghPointCloud( vImgDetect[cNr],  p.scales,  cloud );

            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
            viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "Hough votes");
            viewer->setBackgroundColor(1,1,1);
            viewer->addCoordinateSystem(0.1, 0.f, 0.f, 0.f, 0.f);

            while (!viewer->wasStopped ()){
                viewer->spinOnce (100);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
            }
        }

    }

    // detecting the peaks in the voting space to find the prominent center of the object
    tstart = clock();
    detectCenterPeaks(candidates, vImgDetect, vImgAssign, voterImages, depthImg, img, p, this_class);
    cout << "\t Time for detecting center...\t" << (double)(clock() - tstart)/CLOCKS_PER_SEC << " sec" << endl;

    // detecting pose of the found candidates
    tstart = clock();
    voteForPose( img, voterImages, vImgAssign, vImgDetect, candidates, vImg, normals, p.kernel_width[0], p.scales, p.thresh_detection, p.DEBUG, p.addPoseScore);
    cout << "\t Time for detecting pose.....\t" << (double)(clock() - tstart)/CLOCKS_PER_SEC << " sec" << endl;
}
