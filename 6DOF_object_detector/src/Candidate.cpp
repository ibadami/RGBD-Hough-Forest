//
// C++ Implementation: Candidate
//
// Description: this class holds an structure for keeping the data for a candidate object hypothesis
//
// Author: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
//
//
//


#include "Candidate.h"


Candidate::Candidate(const CRForest* crForest, cv::Mat img, std::vector<float> candidate, int candNr, bool do_bpr){

    bpr = do_bpr;
    weight = candidate[0];
    //        x = candidate[1];
    //        y = candidate[2];
    scale = candidate[3];
    c = int(candidate[4]);
    r = int(candidate[5]);
    //        bbWidth = int(candidate[6]);
    //        bbHeight = int(candidate[7]);

    id = candNr;
    n_trees = int(crForest->vTrees.size());
    n_classes = int(crForest->GetNumLabels());

    if (bpr){

        // initialize the backprojection mask
        backproj_mask = cvCreateImage(cvSize(int(img.cols),int(img.rows)), IPL_DEPTH_32F,1);
        cvSetZero(backproj_mask);
    }

}



