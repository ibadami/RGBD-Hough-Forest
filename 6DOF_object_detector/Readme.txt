// Author: Ishrat Badami AIS University of Bonn
// Email: badami@cs.uni-bonn.de
// Alternative Email: badami@vision.rwth-aachen.de
//
// This is the C++ implementation of class specific object detection 
// and pose estimation. When using this software, please acknowledge the
// effort that went into its development by referencing the papers:
// 
// [1] Badami I., Stückler J., Behnke S., Multi-Scale, Categorical Object
// Detection and Pose Estimation using Hough Forest in RGB-D Images
// 
// [2] Razavi N., Gall J., and van Gool L., Scalable Multi-class Object
// Detection. In proceedings of the IEEE International Conference on 
// Computer Vision and Pattern Recognition, 2011. (CVPR'11)
//
// [3] Gall J., Yao A., Razavi N., van Gool L., and Lempitsky V., Hough Forests 
// for Object Detection, Hough Forests for Object Detection, Tracking, and
// Action Recognition, IEEE Transactions on Pattern 
// Analysis and Machine Intelligence, To appear. 
//
// Since this code is a modification of the code for [2] and [3], the following
// License agreement applies to this code too. 
// 
// You may use, copy, reproduce, and distribute this Software for any 
// non-commercial purpose, subject to the restrictions of the 
// Microsoft Research Shared Source license agreement ("MSR-SSLA"). 
// Some purposes which can be non-commercial are teaching, academic 
// research, public demonstrations and personal experimentation. You 
// may also distribute this Software with books or other teaching 
// materials, or publish the Software on websites, that are intended 
// to teach the use of the Software for academic or other 
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works 
// in any form for commercial purposes. Examples of commercial 
// purposes would be running business operations, licensing, leasing, 
// or selling the Software, distributing the Software for use with 
// commercial products, using the Software in the creation or use of 
// commercial products or any other activity which purpose is to 
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create 
// derivative works of such portions of the Software and distribute 
// the modified Software for non-commercial purposes, as provided 
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO 
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT 
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A 
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR 
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR 
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL 
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST 
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR 
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE 
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA, 
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL 
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT 
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF 
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE 
// WORKS.


The provided software has three directories:
 
detector: contains the c++ source code of the multi-class detector. 

matlab: the matlab functions for calculating the hierarchies and sharing matrices

VOC: pre-trained forests and the config files for the VOC'06 and VOC'07 
datasets are included in this directory.  


// detector directory: //

Source code for the multi-class detector. 

# compile # (needs OpenCV 2.0)
make all

# clean #
make clean

# run #
./run.sh
run this script to get information about the program inputs

A config.txt example is given in the subdirectory 'example'

# multiclass training file #
You can use the function matlab/readTrainingFiles.m to read the training data into matlab.

3// number of classes (including background)
1 class_1.txt// the first positive class 
1 class_2.txt// the second positive class
0 class_bg.txt// the background class

# each training file #
50 1 // number of images + dummy value (1)
/PATH/image.png 0 0 74 36 37 18 // filename + boundingbox (top left - bottom right) + center of bounding box

# multiclass test data #
The test data is organized in a very similar way to the training data. You can use the matlab/readTestFiles.m to read them. 

# hierarchy #
The hierarchy is stored as a text files with the following format:
19 // number of nodes - 1 
19 2 1 0.816167 -1 10 0 1 2 3 4 5 6 7 8 9// node_id leftChild rightChild linkage parent(-1 root) nSubClasses subclasses(10 in this case)
...

# outputs: #
When running the detector it produces a directory for each candidate. The name of the directory is set to:
detect_o[O]_n[N]-[S]-[I]_cand_all
O: tree offset
N: number of trees
S: test set
I: image number in that set

In each directory, two files are written: boundingboxes.txt and candidates.txt

candidates.txt:
5000 // number of detected candidates
0.0292673 313.1 141.6 1.8 19 0 // detection confidence x_position y_position scale class_id dummy
...

boundingboxes.txt
5000 // number of bounding boxes
520 221 600 309 // x_min y_min x_max y_max
...

// matlab directory // 

readTxtForest.m // reading the forest into matlab
readTrainingFiles.m // reading the multi-class training files.
readTestFiles.m // reading the multiclass test files.
getSharingMatrixAppearance.m // calculating the appearance sharing matrix from the forest.
getSharingMatrixAppearanceLocation.m // calculating the appearance and location sharing matrix from the forest.
clusterSharingMatrix.m // calculating the hierarchy by clustering the sharing matrix


//

Nima Razavi
27 July 2011
BIWI, ETH Zurich
