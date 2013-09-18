// Author: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>

using namespace std;

struct Parameters{

    Parameters(){ scale_tree = -1.0f; sample_points_test = -1.0; }

    // name of config file
    string configFileName;

    // name of an object
    string  objectName;

    // suffix for output folder name
    string suffix;

    // Path to images
    string testimagepath;

    // File with names of images
    string testimagefiles;

    // Output path
    string outpath;

    // object model path
    string object_models_path;

    // Path to training examples
    string trainclasspath;

    // File listing training examples from each class
    string trainclassfiles;

    // Number of trees
    int ntrees;

    //Tree depth
    int treedepth;

    // Number of classes
    int nlabels;

    // Subset of positive images -1: all images
    int subsample_images_pos;

    // Subset of positive images -1: all images
    int subsample_images_neg;

    // Sample pixels from pos. examples
    unsigned int samples_pixel_pos;

    // Sample pixels from neg. examples
    unsigned int samples_pixel_neg;

    //Scales
    vector< float > scales;

    // running the detection
    //int do_detect =1;
    // The smoothing kernel parameters
    vector< float > kernel_width;

    // maximum width and height of an object's bounding box at distance of 1 meter
    pair<float, float> objectSize;

    // threshold for the detection
    float thresh_detection;

    // threshold for bounding box
    float thresh_bb;

    // the Alpha in []
    float thresh_vote;

    // type of training: allVSBG=0 multiClassTraining=1 MixedMultiClassBG=2
    int training_mode;

    // number of candidates per class
    int max_candidates;

    // set this variable to enable skipping the already calculated detection
    bool doSkip;

    // backprojecting the bounding box
    bool do_bpr;

    // debug the code
    bool DEBUG;

    // add pose information in position voting
    bool addPoseInformation;

    // add pose information in position voting
    bool addPoseMeasure;

    // add scale information of the pixel to weight those pixel more who are far from the camera
    bool addScaleInformation;

    // addPoseScore
    bool addPoseScore;

    // add surfel Channel
    bool addSurfel;

    // add hog channel
    bool addHoG;

    // add minmaxfilt
    bool addMinMaxFilt;

    // add intensity
    bool addIntensity;

    // setting these variables to determine what classes to do detection/training and test with
    vector<int> train_classes, detect_classes, emp_classes;

    // Class structure
    vector< int > class_structure;

    // offset for saving tree number
    int off_tree;

    // test image number to sprocess
    int off_test;

    // test class to process
    int select_test_class;

    // test set number
    int select_test_set;

    // number of test images to be processed
    int test_num;

    // scale of the tree
    float scale_tree;

    // sampling probability
    double sample_points_test;

    int file_test_num;

    // Path to trees
    string treepath;

    // Path to candidate
    string candidatepath;

    // Path to bounding box
    string bbpath;

    std::vector<cv::Point3f> vbbSize;



};

#endif // PARAMETERS_H
