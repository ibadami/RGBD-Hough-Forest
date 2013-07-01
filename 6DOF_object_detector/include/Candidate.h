//
// C++ Implementation: Candidate
//
// Description: this class holds an structure for keeping all the information necessary for a candidate object hypothesis
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
//
//
//

#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <stdio.h>
#include <string>
#include "utils.h"

#include "Forest.h"


/************************************/
/****     CLASS Candidate        ****/
/************************************/
//
class Candidate{
public:
    // class constructors
    Candidate(){

        weight = -1.f;

    }
    Candidate(const CRForest* crForest, cv::Mat img, std::vector<float> candidate, int id, bool do_bpr=true);
    ~Candidate(){}


public:
    void clear();

    void getBoundingBox();

    void getBBfromBpr(int thresh=2, bool do_sym=false);

    void print(char* prefix){
        // saving the backprojection mask
        if(bpr) save_bp( prefix);
    }

    void read(char* prefix){}

private:
    void save_bp(char* prefix);

public:
    float weight;
    cv::Point2f center;
    float scale;
    int c;//class
    int r;//ratio
    int n_trees;
    int n_classes;
    cv::Point3f bbSize;
    Eigen::Matrix4f coordinateSystem;
    int id;// candidate ID
    bool bpr; // if the backprojection is held

public:
    cv::Rect bb; // bounding box structure
    IplImage* backproj_mask; // backprojection image

};
