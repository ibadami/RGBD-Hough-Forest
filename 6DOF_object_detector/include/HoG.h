/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once


#include "utils.h"
#include <vector>

class HoG {
public:

    HoG();
    ~HoG() {}
    void extractOBin( cv::Mat& Iorient, cv::Mat& Imagn, const cv::Mat& depthImage, std::vector<cv::Mat>& out, int off );
//    void extractOBin( const cv::Mat& rgbImg, const cv::Mat& depthImage, std::vector< cv::Mat >& out, int off );

private:

//    void calculateHOG_rect( std::vector<float>& _hogCell, std::vector<cv::Mat> _integrals, cv::Rect _cell, int _nbins, int _normalization=cv::NORM_MINMAX );
//    std::vector< cv::Mat > calculateIntegralHOG(const cv::Mat& _in, int& _nbins);
    void calcHoGBin(int y, int x, cv::Mat& Iorient, cv::Mat& Imagn, vector<float>& desc);
    void binning(float v, float w, vector<float>& desc, int maxb);

    int bins;
    float binsize;

    int g_w;
    cv::Mat Gauss;

};





