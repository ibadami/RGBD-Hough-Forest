/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <vector>
#include <iostream>
#include "HoG.h"

#define PI 3.14159265

using namespace std;

HoG::HoG() {
    bins = 9;
    g_w = 11;
    binsize = (180.f)/float(bins);

}


void HoG::extractOBin( cv::Mat& Iorient, cv::Mat& Imagn, const cv::Mat& depthImg, std::vector< cv::Mat >& out, int off ) {

    for ( int r = 0; r < Iorient.rows; r++ ){
        for( int c = 0; c < Iorient.cols; c++ ){

            float scale;
            if( depthImg.at< unsigned short >( r , c ) > 0)
                scale = 1000.f / depthImg.at< unsigned short >( r , c );
            else
                scale = 1;
            int adap_g_w = int ( g_w * scale ) + ( int( g_w * scale ) % 2 == 0 ); // it should be an odd number

            float sigma = 0.5 * adap_g_w;
            Gauss = cv::getGaussianKernel( adap_g_w, sigma, CV_32FC1 );

            vector< float > desc( bins, 0.f );
            calcHoGBin( r, c, Iorient, Imagn, desc );

            for( unsigned int l = 0; l < bins; l++ )
                out[ l + off ].at< unsigned char >( r, c ) = ( int )( desc[ l ] );

        }
    }
}

void HoG::calcHoGBin( int r, int c, cv::Mat& Iorient, cv::Mat& Imagn,  vector< float >& desc ) {

    for( int y = 0; y < Gauss.rows; ++y ) {
        for( int x = 0; x < Gauss.cols; ++x ) {

            int row = std::max( y - int(Gauss.rows/2) + r , 0 );
            row = std::min( row, Iorient.rows -1 );

            int col = std::max( x - int(Gauss.cols/2) + c , 0 );
            col = std::min( col, Iorient.cols -1 );

            binning( Iorient.at< float >( row, col) / binsize, Imagn.at< float>( row, col ) * Gauss.at< float >( y, x ), desc, bins );
        }
    }
}

void HoG::binning( float v, float w, vector<float>& desc, int maxb ) {

    int bin1 = int( v );
    int bin2;
    float delta = v - bin1 - 0.5f;
    if( delta < 0 ) {
        bin2 = bin1 < 1 ? maxb  -1 : bin1 - 1;
        delta = -delta;
    } else
        bin2 = bin1 < maxb - 1 ? bin1 + 1 : 0;
    if(bin1 > 0 && bin1 < bins)
        desc[ bin1 ] += ( 1 - delta ) * w;
    if(bin2 > 0 && bin2 < bins)
        desc[ bin2 ] += delta * w;
}



//void HoG::extractOBin( const cv::Mat& rgbImg, const cv::Mat& depthImage, std::vector< cv::Mat >& out, int off ){

//    std::vector< cv::Mat > temp_out(bins);
//    for(unsigned int b = 0; b < bins; b++)
//        temp_out[b] = cv::Mat::zeros(rgbImg.rows, rgbImg.cols, CV_32FC1);

//    // First calculate integral images
//    std::vector< cv::Mat > integral_images;
//    integral_images = calculateIntegralHOG(rgbImg, bins);
////    for(int care = 0; care < bins; care++){
////        cv::imshow("wtf",integral_images[care]);
////        cv::waitKey(0);
////    }

//    // now for each pixel generate depth normalized window
//    for(unsigned int r = 0; r < rgbImg.rows; r++){
//        for(unsigned int c = 0; c < rgbImg.cols; c++){

//            cv::Rect roi;
//            std::vector<float> hog_cell(bins,0);
//            float scale = 1000.f /(float)depthImage.at<unsigned short>(r,c);
//            roi.width = g_w * scale;
//            roi.height = g_w * scale;

//            roi.x = max(0 , int(c - roi.width/2));
//            roi.y = max(0, int(r - roi.height/2));

//            if(roi.x + roi.width > rgbImg.cols - 1 )
//                roi.width = 1;
//            if(roi.y + roi.height > rgbImg.rows - 1 )
//                roi.height = 1;

//            calculateHOG_rect( hog_cell,  integral_images, roi, bins, -1);

//            for( unsigned int b = 0; b < bins; b++){

//                temp_out[b].at< float> (r, c) =  hog_cell[b];

//            }

//        }
//    }

//    // find maxima and normalize entire hog channels

//    std::vector< cv::Point > max_loc_temp( bins );
//    std::vector< double > max_val_temp( bins );

//    // detect the maximum

//    for( unsigned int b = 0; b < bins; b++)
//        cv::minMaxLoc( temp_out[ b ], 0, &max_val_temp[ b ], 0, &max_loc_temp[ b ], cv::Mat() );

//    std::vector< double >::iterator it;
//    it = std::max_element( max_val_temp.begin(),max_val_temp.end() );
//    int max_index = std::distance( max_val_temp.begin(), it );

//    double max_value = max_val_temp[max_index];

//    for( unsigned int b = 0; b < bins; b++){
//        cv::Mat temp_copy;
//        cv::convertScaleAbs( temp_out[ b ], temp_copy, 255/max_value, 0 );
//        temp_copy.copyTo(out[b+off]);
//    }

//}

///*Function to calculate the integral histogram*/
//std::vector< cv::Mat > HoG::calculateIntegralHOG(const cv::Mat& _in, int& _nbins){
//    /*Convert the input image to grayscale*/

//    cv::Mat img_gray;

//    cv::cvtColor(_in,img_gray,CV_BGR2GRAY);
//    cv::equalizeHist(img_gray, img_gray);

//    /* Calculate the derivates of the grayscale image in the x and y
//         directions using a sobel operator and obtain 2 gradient images
//            for the x and y directions*/

//    cv::Mat xsobel, ysobel;
//    cv::Sobel(img_gray,xsobel,CV_32FC1,1,0);
//    cv::Sobel(img_gray,ysobel,CV_32FC1,0,1);

//    img_gray.release();

//    /* Create an array of 9 images (9 because I assume bin size 20 degrees
//         and unsigned gradient ( 180/20 = 9), one for each bin which will have
//            zeroes for all pixels, except for the pixels in the original image
//               for which the gradient values correspond to the particular bin.
//                These will be referred to as bin images. These bin images will be then
//                    used to calculate the integral histogram, which will quicken
//                        the calculation of HOG descriptors */

//    std::vector<cv::Mat> bins(_nbins);
//    for (int i = 0; i < _nbins ; i++)
//    {
//        bins[i] = cv::Mat::zeros(_in.size(), CV_32FC1);
//    }

//    /* Create an array of 9 images ( note the dimensions of the image,
//        the cvIntegral() function requires the size to be that), to store
//            the integral images calculated from the above bin images.
//                These 9 integral images together constitute the integral histogram */

//    std::vector<cv::Mat> integrals(_nbins);
//    for (int i = 0; i < _nbins ; i++)
//    {
//        integrals[i] = cv::Mat(cv::Size(_in.size().width + 1, _in.size().height + 1), CV_64FC1);
//    }

//    /* Calculate the bin images. The magnitude and orientation of the gradient
//          at each pixel is calculated using the xsobel and ysobel images.
//            {Magnitude = sqrt(sq(xsobel) + sq(ysobel) ), gradient = itan (ysobel/xsobel) }.
//                Then according to the orientation of the gradient, the value of the
//                    corresponding pixel in the corresponding image is set */

//    int x, y;
//    float temp_gradient, temp_magnitude;
//    for (y = 0; y <_in.size().height; y++)
//    {
//        /* ptr1 and ptr2 point to beginning of the current row in the xsobel and ysobel images
//            respectively. ptrs[i] point to the beginning of the current rows in the bin images */

//        float* xsobelRowPtr = (float*) (xsobel.row(y).data);
//        float* ysobelRowPtr = (float*) (ysobel.row(y).data);
//        float** binsRowPtrs = new float *[_nbins];
//        for (int i = 0; i < _nbins ;i++)
//        {
//            binsRowPtrs[i] = (float*) (bins[i].row(y).data);
//        }

//        /*For every pixel in a row gradient orientation and magnitude
//                    are calculated and corresponding values set for the bin images. */
//        for (x = 0; x <_in.size().width; x++)
//        {
//            /* if the xsobel derivative is zero for a pixel, a small value is
//                added to it, to avoid division by zero. atan returns values in radians,
//                    which on being converted to degrees, correspond to values between -90 and 90 degrees.
//                        90 is added to each orientation, to shift the orientation values range from {-90-90} to {0-180}.
//                           This is just a matter of convention. {-90-90} values can also be used for the calculation. */
//            if (xsobelRowPtr[x] == 0)
//            {
//                temp_gradient = ((atan(ysobelRowPtr[x] / (xsobelRowPtr[x] + 0.00001))) * (180/ PI)) + 90;
//            }
//            else
//            {
//                temp_gradient = ((atan(ysobelRowPtr[x] / xsobelRowPtr[x])) * (180 / PI)) + 90;
//            }
//            temp_magnitude = sqrt((xsobelRowPtr[x] * xsobelRowPtr[x]) + (ysobelRowPtr[x] * ysobelRowPtr[x]));

//            /*The bin image is selected according to the gradient values.
//                The corresponding pixel value is made equal to the gradient
//                    magnitude at that pixel in the corresponding bin image */
//            float binStep = 180/_nbins;

//            for (int i=1 ; i<=_nbins ; i++)
//            {
//                if (temp_gradient <= binStep*i)
//                {
//                    binsRowPtrs[i-1][x] = temp_magnitude;
//                    break;
//                }
//            }
//        }
//    }


//    xsobel.release();
//    ysobel.release();

//    /*Integral images for each of the bin images are calculated*/

//    for (int i = 0; i <_nbins ; i++)
//    {
//        cv::integral(bins[i], integrals[i]);
////        cv::imshow("bins", bins[i]);
////        cv::waitKey(0);
//    }

//    for (int i = 0; i <_nbins ; i++)
//    {
//        bins[i].release();
//    }

//    /*The function returns an array of 9 images which consitute the integral histogram*/
//    return (integrals);
//}

///*The following demonstrates how the integral histogram calculated using
// the above function can be used to calculate the histogram of oriented
// gradients for any rectangular region in the image:*/

///* The following function takes as input the rectangular cell for which the
// histogram of oriented gradients has to be calculated, a matrix hog_cell
// of dimensions 1x9 to store the bin values for the histogram, the integral histogram,
// and the normalization scheme to be used. No normalization is done if normalization = -1 */

//void HoG::calculateHOG_rect(std::vector<float>& _hogCell, std::vector<cv::Mat> _integrals,
//                            cv::Rect _roi, int _nbins, int _normalization)
//{
//    if (_roi.width == 0 || _roi.height == 0)
//    {
//        _roi.x = 0; _roi.y = 0;
//        _roi.width = _integrals[0].size().width-1;
//        _roi.height = _integrals[0].size().height-1;
//    }
//    /* Calculate the bin values for each of the bin of the histogram one by one */
//    for (int i = 0; i < _nbins ; i++)
//    {
//        IplImage intImgIpl = _integrals[i];

//        float a = ((double*)(intImgIpl.imageData + (_roi.y)
//                             * (intImgIpl.widthStep)))[_roi.x];
//        float b = ((double*) (intImgIpl.imageData + (_roi.y + _roi.height)
//                              * (intImgIpl.widthStep)))[_roi.x + _roi.width];
//        float c = ((double*) (intImgIpl.imageData + (_roi.y)
//                              * (intImgIpl.widthStep)))[_roi.x + _roi.width];
//        float d = ((double*) (intImgIpl.imageData + (_roi.y + _roi.height)
//                              * (intImgIpl.widthStep)))[_roi.x];

//        if(((a + b) - (c + d)) > 0.f)
//            _hogCell[i] = (a + b) - (c + d);
//    }
//    /*Normalize the matrix*/
//    if (_normalization != -1)
//    {
//        cv::normalize(_hogCell, _hogCell, 0, 1, CV_MINMAX);
//    }
//}


