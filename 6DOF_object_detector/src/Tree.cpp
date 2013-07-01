/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
//
//	Modified: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
//
*/

// Modified by: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#include "Tree.h"
#include <fstream>
#include <algorithm>
#include <limits.h>


using namespace std;

/////////////////////// Constructors /////////////////////////////

// Read tree from file
CRTree::CRTree(const char* filename, bool& success) {
    cout << "Load Tree " << filename << endl;

    int dummy;

    ifstream in(filename);
    success = true;
    if(in.is_open()) {
        // get the scale of the tree
        in >> scale;
        in >> max_depth;

        in >> num_nodes;
        nodes.resize(num_nodes);

        in >> num_leaf;
        leafs.resize(num_leaf);

        in >> num_labels;

        // class structure
        class_id = new int[num_labels];
        for(unsigned int n=0;n<num_labels;++n)
            in >> class_id[n];

        int node_id;
        //int isLeaf;
        // read tree nodes
        for(unsigned int n=0; n<num_nodes; ++n) {
            in >> node_id;
            nodes[node_id].idN = node_id;
            in >> nodes[node_id].depth;
            in >> nodes[node_id].isLeaf;
            in >> nodes[node_id].parent;
            in >> nodes[node_id].leftChild;
            in >> nodes[node_id].rightChild;
            nodes[node_id].data.resize(6);
            for (unsigned int i=0; i< 6; ++i){
                in >> nodes[node_id].data[i];
            }
        }

        // read tree leafs
        LeafNode* ptLN;
        for( unsigned int l = 0; l < num_leaf; ++l ){
            ptLN = &leafs[l];
            in >> ptLN->idL;
            in >> ptLN->depth;
            in >> ptLN->parent;
            in >> ptLN->cL;
            //in >> ptLN->eL;
            //in >> ptLN->fL;
            ptLN->vPrLabel.resize(num_labels);
            ptLN->vCenter.resize(num_labels);
            ptLN->vCenterWeights.resize(num_labels);
            ptLN->vID.resize(num_labels);
            //ptLN->vLabelDistrib.resize(num_labels);
            ptLN->nOcc.resize(num_labels);

            ptLN->vPose.resize(num_labels);
            ptLN->QdisVector.resize(num_labels);
            ptLN->alpha.resize(num_labels);

            ptLN->bbSize3D.resize(num_labels);



            for(unsigned int c=0; c<num_labels;++c) {
                in >> ptLN->vPrLabel[c];
                in >> dummy;

                if (ptLN->vPrLabel[c]<0){
                    std::cerr<<ptLN->vPrLabel[c]<<std::endl;
                }

                ptLN->vCenter[c].resize(dummy);
                ptLN->vCenterWeights[c].resize(dummy);
                ptLN->vID[c].resize(dummy);
                ptLN->nOcc[c] = dummy;

                ptLN->vPose[c].resize(dummy);
                ptLN->QdisVector[c].resize(dummy);
                ptLN->alpha[c].resize(dummy);

                ptLN->bbSize3D[c].resize(dummy);


                float temp_weight = 1.0f/float(ptLN->nOcc[c]);

                //                cout << "label: "<< c << " "<< dummy << " ";
                for(int i=0; i<dummy; ++i) {

                    char string[10], *stopstring;
                    in >> string; ptLN->vCenter[c][i].x= strtof(string, &stopstring);
                    in >> string; ptLN->vCenter[c][i].y= strtof(string, &stopstring);
                    in >> string; ptLN->vCenter[c][i].z= strtof(string, &stopstring);

                    //                    in >> string; ptLN->PCdimension[c][i].width = strtof(string, &stopstring);
                    //                    in >> string; ptLN->PCdimension[c][i].height = strtof(string, &stopstring);
                    //                    in >> string; ptLN->PCrotAngle[c][i] = strtof(string, &stopstring);
                    in >> string; ptLN->bbSize3D[c][i].x = strtof(string, &stopstring);
                    in >> string; ptLN->bbSize3D[c][i].y = strtof(string, &stopstring);
                    in >> string; ptLN->bbSize3D[c][i].z = strtof(string, &stopstring);

                    ptLN->vCenterWeights[c][i] = temp_weight;
                    in >> *(ptLN->vID[c][i]);
                    ptLN->alpha[c][i].resize(ptLN->depth);

                    in >> string; ptLN->vPose[c][i].first.coeffs()[0] = strtof(string, &stopstring);
                    in >> string; ptLN->vPose[c][i].first.coeffs()[1] = strtof(string, &stopstring);
                    in >> string; ptLN->vPose[c][i].first.coeffs()[2] = strtof(string, &stopstring);
                    in >> string; ptLN->vPose[c][i].first.coeffs()[3] = strtof(string, &stopstring);

                    in >> string; ptLN->vPose[c][i].second.coeffs()[0] = strtof(string, &stopstring);
                    in >> string; ptLN->vPose[c][i].second.coeffs()[1] = strtof(string, &stopstring);
                    in >> string; ptLN->vPose[c][i].second.coeffs()[2] = strtof(string, &stopstring);
                    in >> string; ptLN->vPose[c][i].second.coeffs()[3] = strtof(string, &stopstring);

                    in >> string; ptLN->QdisVector[c][i].first[0] = strtof(string, &stopstring);
                    in >> string; ptLN->QdisVector[c][i].first[1] = strtof(string, &stopstring);
                    in >> string; ptLN->QdisVector[c][i].first[2] = strtof(string, &stopstring);
                    //                    in >> string; ptLN->QdisVector[c][i].first[3] = strtof(string, &stopstring);

                    in >> string; ptLN->QdisVector[c][i].second[0] = strtof(string, &stopstring);
                    in >> string; ptLN->QdisVector[c][i].second[1] = strtof(string, &stopstring);
                    in >> string; ptLN->QdisVector[c][i].second[2] = strtof(string, &stopstring);
                    //                    in >> string; ptLN->QdisVector[c][i].second[3] = strtof(string, &stopstring);

                    for(unsigned int j = 0; j < ptLN->depth; j++){

                        in >> string; ptLN->alpha[c][i][j].first = strtof( string, &stopstring );
                        in >> string; ptLN->alpha[c][i][j].second = strtof( string, &stopstring );
                    }
                }
            }

            //            cout << "\n" << endl;
        }

    } else {
        success = false;
        cerr << "Could not read tree: " << filename << endl;
    }

    in.close();
    cout << GetScale() << endl;

}

/////////////////////// IO Function /////////////////////////////
bool CRTree::saveTree(const char* filename) const {
    cout << "Save Tree " << filename << endl;

    bool done = false;

    ofstream out(filename);
    if(out.is_open()) {

        out << scale << " " << max_depth << " " << num_nodes << " " << num_leaf << " " << num_labels << endl;

        // store class structure
        for(unsigned int n=0;n<num_labels;++n)
            out << class_id[n] << " ";
        out<< endl;

        // save tree nodes
        for(unsigned int n=0; n<num_nodes; ++n) {

            out <<  nodes[n].idN;
            out << " " << nodes[n].depth;
            out << " " << nodes[n].isLeaf;
            out << " " << nodes[n].parent;
            out << " " << nodes[n].leftChild;
            out << " " << nodes[n].rightChild;

            for (unsigned int i=0; i< 6; ++i){
                out << " " << nodes[n].data[i];
            }
            out << endl;
        }
        out << endl;
        // save tree leaves
        for(unsigned int l=0; l<num_leaf; ++l) {
            const LeafNode* ptLN = &leafs[l];

            out << ptLN->idL << " ";
            out << ptLN->depth << " ";
            out << ptLN->parent << " ";
            out << ptLN->cL << " \n";
            //out << ptLN->eL << " ";
            //out << ptLN->fL << "\n ";

            for(unsigned int c = 0; c < num_labels;++c) {
                out << ptLN->vPrLabel[c] << " " << ptLN->vCenter[c].size() << " " << " \n ";

                for(unsigned int i=0; i<ptLN->vCenter[c].size(); ++i) {
                    out << ptLN->vCenter[c][i].x << " " << ptLN->vCenter[c][i].y << " " << ptLN->vCenter[c][i].z << " " << ptLN->bbSize3D[c][i].x << " " << ptLN->bbSize3D[c][i].y << " " << ptLN->bbSize3D[c][i].z<< " " << ptLN->vID[c][i] << "\n ";
                    out << ptLN->vPose[c][i].first.coeffs()[0] << " ";
                    out << ptLN->vPose[c][i].first.coeffs()[1] << " ";
                    out << ptLN->vPose[c][i].first.coeffs()[2] << " ";
                    out << ptLN->vPose[c][i].first.coeffs()[3] << " \t";
                    out << ptLN->vPose[c][i].second.coeffs()[0] << " ";
                    out << ptLN->vPose[c][i].second.coeffs()[1] << " ";
                    out << ptLN->vPose[c][i].second.coeffs()[2] << " ";
                    out << ptLN->vPose[c][i].second.coeffs()[3] << " \n";

                    out << ptLN->QdisVector[c][i].first[0] << " ";
                    out << ptLN->QdisVector[c][i].first[1] << " ";
                    out << ptLN->QdisVector[c][i].first[2] << " \t";
                    //                out << ptLN->QdisVector[c][i].first[3] << " ";

                    out << ptLN->QdisVector[c][i].second[0] << " ";
                    out << ptLN->QdisVector[c][i].second[1] << " ";
                    out << ptLN->QdisVector[c][i].second[2] << " \n";
                    //                out << ptLN->QdisVector[c][i].second[3] << " \n";
                    // save pose information
                    for(unsigned int j = 0; j<ptLN->depth; j++){
                        out << ptLN->alpha[c][i][j].first << " ";
                        out << ptLN->alpha[c][i][j].second << " \n";

                    }

                }
            }

            out << endl;
        }

        out.close();

        done = true;
    }

    return done;
}


bool CRTree::loadHierarchy(const char* filename){
    ifstream in(filename);
    int number_of_nodes = 0;
    if(in.is_open()) {
        in >> number_of_nodes;
        hierarchy.resize(number_of_nodes);
        int temp;
        for (int nNr=0 ; nNr < number_of_nodes; nNr++){
            in >> hierarchy[nNr].id;
            in >> hierarchy[nNr].leftChild;
            in >> hierarchy[nNr].rightChild;
            in >> hierarchy[nNr].linkage;
            in >> hierarchy[nNr].parent;
            in >> temp;
            hierarchy[nNr].subclasses.resize(temp);
            for (int sNr=0; sNr < temp; sNr++)
                in >> hierarchy[nNr].subclasses[sNr];
        }
        in.close();
        return true;
    }else{
        std::cerr << " failed to read the hierarchy file: " << filename << std::endl;
        return false;
    }
}

/////////////////////// Training Function /////////////////////////////

// Start grow tree
void CRTree::growTree( const Parameters& param, CRPixel& TrData, int samples, int trNr ) {
    // Get inverse numbers of pixels
    vector<float> vRatio(TrData.vRPixels.size());

    vector<vector< PixelFeature*> > TrainSet( TrData.vRPixels.size() );
    vector<vector<int*> > TrainIDs(TrData.vImageIDs.size());

    for( unsigned int l = 0; l < TrainSet.size(); ++l ) {// l is nlabels of each class
        TrainSet[l].resize( TrData.vRPixels[l].size() );
        TrainIDs[l].resize( TrData.vImageIDs[l].size() );

        if ( TrainSet[l].size() > 0 ){
            vRatio[l] = 1.0f/(float)TrainSet[l].size();
        }else{
            vRatio[l] = 0.0f;
        }
        for(unsigned int i=0; i<TrainSet[l].size(); ++i) {

            TrainSet[l][i] = new PixelFeature;

            TrainSet[l][i]->statFeature = TrData.vRPixels[l][i].statFeature;
            TrainSet[l][i]->dynFeature->alpha.clear();
            TrainSet[l][i]->dynFeature->alpha.reserve(max_depth);
        }
        for(unsigned int i=0; i<TrainIDs[l].size(); ++i) {
            TrainIDs[l][i] = &TrData.vImageIDs[l][i];
        }
    }
    // Grow tree
    grow(param, TrainSet, TrainIDs, 0, 0, samples, vRatio , trNr );
}

// Called by growTree
void CRTree::grow(const Parameters& param, const vector<vector< PixelFeature*> >& TrainSet, const vector<vector<int*> >& TrainIDs, int node, unsigned int depth, int samples, vector<float>& vRatio, int trNr) {

    if(depth < max_depth) {
        vector<vector< PixelFeature*> > SetA;
        vector<vector< PixelFeature*> > SetB;
        vector<vector<int*> > idA;
        vector<vector<int*> > idB;

        int test[6];

        // Set measure mode for split: -1         - classification,
        //                             otherwise  - regression (for locations)
        int stat[ TrainSet.size() ]; //stat has labels of all classes in the TrainSet
        int count_stat = getStatSet( TrainSet, stat ); // nlables

        cout << "SetSize: ";
        for( unsigned int l = 0; l < TrainSet.size(); ++l )
            cout << TrainSet[l].size() << " ";
        cout << endl;

        bool check_test = false;
        int count_test = 0;

        while( !check_test ) {
            int measure_mode = 0;
            if( count_stat > 1 )
                measure_mode = ( cvRNG->operator ()( 4 ) ) - 1;
            else
                measure_mode = ( cvRNG->operator ()( 2 ) ) + 1;

            cout << "MeasureMode: " <<  measure_mode << ", Depth = " << depth << ", Tree: " << trNr << endl;

            // Find optimal test
            if( check_test = optimizeTest(param, SetA, SetB, idA, idB, TrainSet, TrainIDs, test, samples, measure_mode, vRatio, node) ) {

                // Store binary test for current node
                InternalNode* ptT = &nodes[node];
                ptT->data.resize(6);
                for( int t = 0; t < 6; ++t)
                    ptT->data[t] = test[t];

                double countA = 0;
                double countB = 0;
                for(unsigned int l=0; l<TrainSet.size(); ++l) {
                    countA += SetA[l].size(); countB += SetB[l].size();
                }

                //make an empty node and push it to the tree
                InternalNode temp;
                temp.rightChild = -1;
                temp.leftChild = -1;
                temp.parent = node;
                temp.data.resize(6,0);
                temp.depth = depth +1;

                // Go left
                temp.idN = nodes.size();
                nodes[node].leftChild = temp.idN;

                // If enough patches are left continue growing else stop
                if(countA > min_samples) {
                    temp.isLeaf = false;
                    nodes.push_back(temp);
                    num_nodes +=1;
                    grow(param, SetA, idA, temp.idN, depth+1, samples, vRatio, trNr);
                } else {
                    // the leaf id will be assigned to the left child in the makeLeaf
                    // isLeaf will be set to true
                    temp.isLeaf = true;
                    nodes.push_back(temp);
                    num_nodes +=1;
                    makeLeaf(SetA, idA, vRatio, temp.idN);
                }

                // Go right
                temp.idN = nodes.size();
                nodes[node].rightChild = temp.idN;
                // If enough patches are left continue growing else stop
                if(countB > min_samples) {
                    temp.isLeaf = false;
                    nodes.push_back(temp);
                    num_nodes += 1;
                    grow(param, SetB, idB, temp.idN, depth+1, samples, vRatio, trNr);
                } else {
                    temp.isLeaf = true;
                    nodes.push_back(temp);
                    num_nodes +=1;
                    makeLeaf(SetB, idB, vRatio, temp.idN);
                }

            } else {

                if(++count_test>3) {
                    cout << "Invalid Test" << endl;

                    // Could not find split (only invalid splits)
                    nodes[node].isLeaf = true;
                    nodes[node].leftChild = -1;
                    nodes[node].rightChild = -1;
                    nodes[node].data.resize(6,0);
                    makeLeaf(TrainSet, TrainIDs, vRatio, node);

                    check_test = true;
                }
            }
        }
    } else {
        // maximum depth is reached
        nodes[node].isLeaf = true;
        nodes[node].leftChild = -1;
        nodes[node].rightChild = -1;
        nodes[node].data.resize(6,0);
        // do not change the parent
        makeLeaf(TrainSet, TrainIDs, vRatio, node);
    }

}

// Create leaf node from patches
void CRTree::makeLeaf(const std::vector<std::vector< PixelFeature*> >& TrainSet, const std::vector<std::vector< int*> >& TrainIDs, std::vector<float>& vRatio, int node) {

    // setting the leaf pointer
    nodes[node].leftChild = num_leaf;

    LeafNode L;
    L.idL = num_leaf;
    L.depth = nodes[node].depth;
    L.parent = nodes[node].parent;

    L.vCenter.resize(TrainSet.size());
    L.vPrLabel.resize(TrainSet.size());
    L.vID.resize(TrainSet.size());

    L.vPose.resize(TrainSet.size());
    L.QdisVector.resize(TrainSet.size());
    L.alpha.resize(TrainSet.size());

    L.bbSize3D.resize(TrainSet.size());

    // Store data
    float invsum = 0;
    float invsum_pos = 0;
    for(unsigned int l=0; l<TrainSet.size(); ++l) {

        L.vPrLabel[l] = (float)TrainSet[l].size() * vRatio[l];
        invsum += L.vPrLabel[l];

        if (class_id[l] > 0)
            invsum_pos += L.vPrLabel[l];

        L.vCenter[l].resize( TrainSet[l].size() );
        L.vID[l].resize(TrainIDs[l].size());

        L.vPose[l].resize(TrainSet[l].size());
        L.QdisVector[l].resize(TrainSet[l].size());
        L.alpha[l].resize(TrainSet[l].size());

        L.bbSize3D[l].resize(TrainSet[l].size());

        for(unsigned int i = 0; i<TrainSet[l].size(); ++i) {

            L.vCenter[l][i] = TrainSet[l][i]->statFeature->disVector;
            L.vID[l][i] =  TrainIDs[l][i];

            L.bbSize3D[l][i] = TrainSet[l][i]->statFeature->bbSize3D;

            L.vPose[l][i] = TrainSet[l][i]->dynFeature->pointPairTransformation;
            L.QdisVector[l][i] = TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate;
            L.alpha[l][i].resize(TrainSet[l][i]->dynFeature->alpha.size());

            for(unsigned int j = 0; j < TrainSet[l][i]->dynFeature->alpha.size(); j++)
                L.alpha[l][i][j] = TrainSet[l][i]->dynFeature->alpha[j];

        }
    }

    // Normalize probability
    invsum = 1.0f/invsum;
    if (invsum_pos > 0){
        invsum_pos = 1.0f/invsum_pos;
        for(unsigned int l=0; l<TrainSet.size(); ++l){
            L.vPrLabel[l] *= invsum;
        }
        L.cL = invsum/invsum_pos;
    }else{ // there is no positive patch in this leaf
        for(unsigned int l=0; l<TrainSet.size(); ++l){
            L.vPrLabel[l] *= invsum;
        }
        L.cL = 0.0f;
    }

    leafs.push_back(L);

    // Increase leaf counter
    ++num_leaf;
}

bool CRTree::optimizeTest(const Parameters& param, vector<vector< PixelFeature*> >& SetA, vector<vector< PixelFeature*> >& SetB, vector<vector<int*> >& idA, vector<vector<int*> >& idB , const vector<vector< PixelFeature*> >& TrainSet, const vector<vector<int*> >& TrainIDs , int* test, unsigned int iter, unsigned int measure_mode,const std::vector<float>& vRatio, int node) {

    bool found = false;
    int subsample = 1000*TrainSet.size();

    // sampling pixels proportional to the class to keep the balance of the classes
    std::vector<int> subsample_perclass;
    subsample_perclass.resize(TrainSet.size(),0);
    // first find out how many pixels are there
    int all_patches = 0;
    for (unsigned int sz=0; sz < TrainSet.size(); sz++)
        all_patches += TrainSet[sz].size();
    // the calculate the sampling rate for each set
    float sample_rate = float(subsample)/float(all_patches);
    for ( unsigned int sz=0; sz < TrainSet.size(); sz++){
        subsample_perclass[sz] = int(sample_rate*float(TrainSet[sz].size()));
    }
    // now we can subsample the patches and their associated ids
    vector<vector< PixelFeature*> > tmpTrainSet;
    vector<vector< int* > > tmpTrainIDs;
    tmpTrainSet.resize(TrainSet.size());
    tmpTrainIDs.resize(TrainSet.size());

    // sample the patches in a regular grid and copy them to the tree
    for (unsigned int sz=0; sz < TrainSet.size() ; sz++){
        tmpTrainSet[sz].resize(std::min(int(TrainSet[sz].size()),subsample_perclass[sz]));
        tmpTrainIDs[sz].resize(tmpTrainSet[sz].size());
        if (tmpTrainSet[sz].size()==0)
            continue;

        float float_rate = float(TrainSet[sz].size())/float(tmpTrainSet[sz].size());
        for (unsigned int j=0; j < tmpTrainSet[sz].size(); j++){
            tmpTrainSet[sz][j] = TrainSet[sz][int(float_rate*j)];
            tmpTrainIDs[sz][j] = TrainIDs[sz][int(float_rate*j)];
        }
    }

    double tmpDist;
    double bestDist = -DBL_MAX;
    int tmpTest[6];

    // find non-empty class
    int check_label = 0;
    while(check_label<(int)tmpTrainSet.size() && tmpTrainSet[check_label].size()==0)
        ++check_label;

    // Find best test of ITER iterations
    for(unsigned int i =0; i<iter; ++i) {
        // temporary data for split into Set A and Set B
        vector<vector< PixelFeature*> > tmpA(tmpTrainSet.size());
        vector<vector< PixelFeature*> > tmpB(tmpTrainSet.size());
        vector<vector<int*> > tmpIDA(tmpTrainIDs.size());
        vector<vector<int*> > tmpIDB(tmpTrainIDs.size());
        // temporary data for finding best test
        vector<vector<IntIndex> > tmpValSet(tmpTrainSet.size());
        // generate binary test without threshold
        generateTest(param, &tmpTest[0], class_size[0].first, class_size[0].second, tmpTrainSet[check_label][0]->statFeature->imgAppearance.size());

        // compute value for each patch
        evaluateTest( tmpValSet, &tmpTest[0], tmpTrainSet, node, false, param.addPoseMeasure);

        // find min/max values for threshold
        int vmin = INT_MAX;
        int vmax = INT_MIN;
        for(unsigned int l = 0; l<tmpTrainSet.size(); ++l) {
            if(tmpValSet[l].size()>0) {
                if(vmin>tmpValSet[l].front().val)  vmin = tmpValSet[l].front().val;
                if(vmax<tmpValSet[l].back().val )  vmax = tmpValSet[l].back().val;
            }
        }
        int d = vmax-vmin;

        if(d>0) {

            // Find best threshold
            for(unsigned int j=0; j < 10; ++j) {

                // Generate some random thresholds
                int tr = (cvRNG->operator ()(d)) + vmin;

                // Split training data into two sets A,B accroding to threshold t
                split(tmpA, tmpB, tmpIDA, tmpIDB, tmpTrainSet, tmpValSet, tmpTrainIDs, tr);// include idA , idB, TrainIDs
                int countA = 0;
                int countB = 0;
                for( int l = 0; l< (int)tmpTrainSet.size(); ++l) {
                    if ((int)tmpA[l].size()> countA)
                        countA = tmpA[l].size();
                    if ((int)tmpB[l].size() > countB)
                        countB = tmpB[l].size();
                }

                // Do not allow empty set split (all patches end up in set A or B)

                if( countA>10 && countB>10 ) {
                    // Measure quality of split with measure_mode 0 - classification, 1 - regression
                    tmpDist = measureSet(tmpA, tmpB, measure_mode, vRatio, param.addPoseMeasure);

                    // Take binary test with best split
                    if(tmpDist > bestDist) {

                        found = true;
                        bestDist = tmpDist;
                        for(int t=0; t<5;++t) test[t] = tmpTest[t];
                        test[5] = tr;
                    }
                }
            } // end for
        }
    } // end iter

    if (found){
        // here we should evaluate the test on all the data
        vector<vector<IntIndex> > valSet(TrainSet.size());
        evaluateTest( valSet, &test[0], TrainSet, node, true, false);
        // now we can keep the best Test and split the whole set according to the best test and threshold
        SetA.resize(TrainSet.size());
        SetB.resize(TrainSet.size());
        idA.resize(TrainSet.size());
        idB.resize(TrainSet.size());
        split(SetA, SetB, idA, idB, TrainSet, valSet, TrainIDs, test[5]);
    }
    // return true if a valid test has been found
    // test is invalid if only splits with with members all less than 10 in set A or B has been created
    return found;
}

void CRTree::evaluateTest( std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector< PixelFeature*> >& TrainSet, int node, bool addTransformation, bool addPoseMeasure) {

    for( unsigned int l = 0; l < TrainSet.size(); ++l ) {
        valSet[ l ].resize( TrainSet[ l ].size() );

        for( unsigned int i = 0; i < TrainSet[ l ].size(); ++i ) {

            //reconstruct pixel pair using vectors saved in test array
            cv::Mat ptC;
            cv::Point2f pt1,pt2;
            PixelFeature* pf = TrainSet[ l ][ i ];


            pt1.x = std::max( int( 0.f /*pf->bbox.x */), int( pf->statFeature->pixelLocation.x + test[ 0 ] * pf->statFeature->scale ) );
            pt1.x = std::min( int( pt1.x ), pf->statFeature->iWidth /*pf->bbox.width + pf->bbox.x*/ - 1 );

            pt1.y = std::max( int( 0.f /*pf->bbox.y*/ ), int(pf->statFeature->pixelLocation.y + test[ 1 ] * pf->statFeature->scale ) );
            pt1.y = std::min( int( pt1.y), pf->statFeature->iHeight /*pf->bbox.height + pf->bbox.y*/ -1);

            pt2.x = std::max( int( 0.f /*pf->bbox.x*/ ), int(pf->statFeature->pixelLocation.x + test[ 2 ] * pf->statFeature->scale ) );
            pt2.x = std::min( int( pt2.x), pf->statFeature->iWidth /*pf->bbox.width + pf->bbox.x*/ -1);

            pt2.y = std::max( int( 0.f /*pf->bbox.y*/ ), int( pf->statFeature->pixelLocation.y + test[ 3 ] * pf->statFeature->scale ) );
            pt2.y = std::min( int( pt2.y),  pf->statFeature->iHeight /*pf->bbox.height + pf->bbox.y*/ -1 );

            //debug
            if(0){

                cv::Mat img_show = cv::Mat( ptC.rows, ptC.cols, CV_8UC3,cv::Scalar(0) );
                cv::cvtColor( pf->statFeature->imgAppearance[ 0 ], img_show, CV_GRAY2RGB );
                cv::Point pixel = pf->statFeature->pixelLocation;

                cv::circle(img_show, pt1, 1, CV_RGB( 255, 0, 0 ), 8, 8, 0);
                cv::circle(img_show, pt2, 1, CV_RGB( 255, 0, 0 ), 8, 8, 0);
                cv::circle(img_show, pixel, 1, CV_RGB( 0, 255, 0 ), 8, 8, 0);
                cv::line(img_show,pt1, pixel, CV_RGB( 255, 0, 255 ), 2, 8, 0);
                cv::line(img_show, pt2, pixel, CV_RGB( 255, 0, 255 ), 2, 8, 0);
                cv::imshow("img",img_show); cv::waitKey(0);

            }

            // if the channel is not Surfel feature
            if( test[4] < pf->statFeature->imgAppearance.size() && test[4] != 7 && test[4] != 15 && test[4] != 24 ){
                // pointer to channel
                ptC = pf->statFeature->imgAppearance[test[4]];
                // get pixel values
                int p1 = (int)(ptC.at<unsigned char>(pt1));
                int p2 = (int)(ptC.at<unsigned char>(pt2));
                valSet[l][i].val = p1 - p2;
            }else if(test[4] == 7 || test[4] == 15 || test[4] == 24){
                // pointer to channel
                ptC = pf->statFeature->imgAppearance[test[4]];
                // get pixel values
                int p1 = (int)(ptC.at<unsigned short>(pt1));
                int p2 = (int)(ptC.at<unsigned short>(pt2));
                valSet[l][i].val = p1 - p2;
            }else{ // if the channel is Surfel feature

                // calculate surfel feature
                SurfelFeature sf;
                Surfel::computeSurfel(pf->statFeature->normals, cv::Point2f(pt1.x, pt1.y), cv::Point2f(pt2.x, pt2.y), cv::Point2f(pf->statFeature->iWidth/2.f, pf->statFeature->iHeight/2.f), sf, pf->statFeature->imgAppearance[7].at<unsigned short>(pt1)/1000.f, pf->statFeature->imgAppearance[7].at<unsigned short>(pt2)/1000.f  );
                float  tempVal = sf.fVector[test[4] - TrainSet[l][i]->statFeature->imgAppearance.size()];
                if(isnan(tempVal)){
                    if(i == 0)
                        tempVal = 0;
                    else
                        valSet[l][i-1].val;
                }
                valSet[l][i].val = tempVal;
            }

            if (addTransformation ){

                cv::Point2f center(pf->statFeature->iWidth/2.f, TrainSet[l][i]->statFeature->iHeight/2.f);
                cv::Point3f s1 = CRPixel::P3toR3(pt1, center, pf->statFeature->imgAppearance[7].at< unsigned short >(pt1)/1000.f );
                cv::Point3f s2 = CRPixel::P3toR3(pt2, center, pf->statFeature->imgAppearance[7].at< unsigned short >(pt2)/1000.f );
                cv::Point3f q = pf->statFeature->pixelLocation_real;

                pcl::Normal qn = pf->statFeature->normals->at(pf->statFeature->pixelLocation.x, pf->statFeature->pixelLocation.y);

                // find transformation from query pixel coordinate system to object coordinate system
                Eigen::Matrix4f transformationMatrixQuery1C, transformationMatrixQuery2C;
                Surfel::calcQueryPoint2CameraTransformation(s1, s2, q, qn, transformationMatrixQuery1C, transformationMatrixQuery2C);

                Eigen::Matrix4f transformationMatrixOQuery1 , transformationMatrixOQuery2;
                float alpha1, alpha2;

                if( node == 0){

                    // find transformation from query pixel coordinate system to object coordinate system
                    transformationMatrixOQuery1 = transformationMatrixQuery1C.inverse() * pf->statFeature->transformationMatrixOC;
                    transformationMatrixOQuery2 = transformationMatrixQuery2C.inverse() * pf->statFeature->transformationMatrixOC;

                    // Extract rotation matrix out of transformation matrix
                    Eigen::Matrix3f rotationMatrixOQuery1 = transformationMatrixOQuery1.block< 3, 3 >( 0, 0 );
                    Eigen::Matrix3f rotationMatrixOQuery2 = transformationMatrixOQuery2.block< 3, 3 >( 0, 0 );

                    // Convert rotation matrix to quaternions
                    Eigen::Quaternionf QuaternionOQuery1( rotationMatrixOQuery1);
                    Eigen::Quaternionf QuaternionOQuery2( rotationMatrixOQuery2 );

                    // Extract translation from the transfomation matrix
                    Eigen::Vector3f QOdisVector1 = transformationMatrixOQuery1.block< 3, 1 >( 0, 3 );
                    Eigen::Vector3f QOdisVector2 = transformationMatrixOQuery2.block< 3, 1 >( 0, 3 );

                    // save only for first node
                    if(!isnan(QOdisVector1[0]))
                        TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate.first = QOdisVector1;
                    else
                        TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate.first = Eigen::Vector3f(0.f,0.f,0.f);
                    if(!isnan(QOdisVector2[0]))
                        TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate.second = QOdisVector2;
                    else
                        TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate.second = Eigen::Vector3f(0.f,0.f,0.f);
                    if(!isnan(QuaternionOQuery1.coeffs()[ 0 ]))
                        TrainSet[l][i]->dynFeature->pointPairTransformation.first           = QuaternionOQuery1;
                    else
                        TrainSet[l][i]->dynFeature->pointPairTransformation.first           = Eigen::Quaternionf::Identity();
                    if(!isnan(QuaternionOQuery2.coeffs()[ 0 ]))
                        TrainSet[l][i]->dynFeature->pointPairTransformation.second          = QuaternionOQuery2;
                    else
                        TrainSet[l][i]->dynFeature->pointPairTransformation.second          = Eigen::Quaternionf::Identity();

                    TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.first  = transformationMatrixQuery1C.block< 3, 3 >( 0, 0 );
                    TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.second = transformationMatrixQuery2C.block< 3, 3 >( 0, 0 );

                    Eigen::Matrix3f alphaRotationMatrix1 = TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.first.inverse() * transformationMatrixQuery1C.block< 3, 3 >( 0, 0 );
                    Eigen::Matrix3f alphaRotationMatrix2 = TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.second.inverse() * transformationMatrixQuery2C.block< 3, 3 >( 0, 0 );

                    alpha1 = std::atan2(alphaRotationMatrix1(1,0), alphaRotationMatrix1(0,0));
                    alpha2 = std::atan2(alphaRotationMatrix2(1,0), alphaRotationMatrix2(0,0));

                    if(isnan(alpha1)) alpha1 = 0;
                    if(isnan(alpha2)) alpha2 = 0;

                    TrainSet[l][i]->dynFeature->alpha.push_back( std::pair< float, float >( alpha1, alpha2 ));

                }else{

                    // save only relative angle and discretized with 2*PI/255 step size

                    Eigen::Matrix3f alphaRotationMatrix1 = TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.first.inverse() * transformationMatrixQuery1C.block< 3, 3 >( 0, 0 );
                    Eigen::Matrix3f alphaRotationMatrix2 = TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.second.inverse() * transformationMatrixQuery2C.block< 3, 3 >( 0, 0 );

                    alpha1 = std::atan2(alphaRotationMatrix1(1,0), alphaRotationMatrix1(0,0));
                    alpha2 = std::atan2(alphaRotationMatrix2(1,0), alphaRotationMatrix2(0,0));

                    if(isnan(alpha1)) alpha1 = 0;
                    if(isnan(alpha2)) alpha2 = 0;

                    TrainSet[l][i]->dynFeature->alpha.push_back(std::pair< float, float >(alpha1, alpha2));
                }

                //debug
                if(0){

                    cv::Point3f O = TrainSet[ l ][ i ]->statFeature->pixelLocation_real - TrainSet[ l ][ i ]->statFeature->disVector;

                    Eigen::Quaternionf alphaRotation(cos(alpha1/2.f),0.f,0.f,sin(alpha1/2.f));

                    Eigen::Vector4f QdisVector1_t = transformationMatrixQuery1C.inverse() * Eigen::Vector4f(O.x, O.y, O.z,1);

                    cout<< "saved vector\n" << TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate.first << "\n" << endl;
                    cout<< "debugged Vector\n" << QdisVector1_t <<"\n"<<endl;

                }


                // visualize coordinate syatem, and two surfel coordinate systems
                if(0){

                    if( !(isnan(qn.normal_x) || q.z == 0 ) ){ // when query point has no depth or has no normal

                        // get rgbimg and depthImg

                        std::vector<cv::Mat> in(3);
                        for( int channel = 0; channel < 3; channel++ ){

                            in[ channel ] = TrainSet[l][i]->statFeature->imgAppearance[ channel ];
                        }

                        cv::Mat Lab, rgbImg, depthImg;
                        cv::merge(in, Lab);
                        cv::cvtColor(Lab , rgbImg, CV_Lab2BGR);
                        cv::circle(rgbImg, pt1, 2, CV_RGB( 0, 255, 0 ), -1, 8, 0);
                        cv::circle(rgbImg, pt2, 2, CV_RGB( 0, 255, 0 ), -1, 8, 0);
                        cv::circle(rgbImg, TrainSet[ l ][ i ]->statFeature->pixelLocation, 2, CV_RGB( 255, 0, 0 ), -1, 8, 0);

                        depthImg = TrainSet[l][i]->statFeature->imgAppearance[7];

                        //                        CRPixel::drawTransformation(rgbImg, depthImg ,transformationMatrixOQuery1 );

                        // calculate transformations

                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
                        Surfel::imagesToPointCloud( depthImg, rgbImg, cloud );

                        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
                        viewer->addCoordinateSystem	(0.1f,0.f,0.f,0.f, 0 );
                        viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
                        viewer->setBackgroundColor(1.f,1.f,1.f);
                        Surfel::addCoordinateSystem(TrainSet[l][i]->statFeature->transformationMatrixOC, viewer, "1");
                        Surfel::addCoordinateSystem(transformationMatrixQuery1C, viewer, "2");


                        Eigen::Quaternionf alphaRotation(cos(alpha1/2.f),0.f,0.f,sin(alpha1/2.f));

                        //                        Eigen::Quaternionf alphaRotationcheck = Eigen::Quaternionf(alphaRotation.toRotationMatrix());

                        Eigen::Matrix3f debugRotationMatrix =  TrainSet[l][i]->dynFeature->firstQuerytoCameraTransformation.first * Eigen::Matrix3f(alphaRotation);
                        Eigen::Matrix4f debugTransformationMatrix = Eigen::Matrix4f::Identity();
                        debugTransformationMatrix.block<3,3>(0,0) = debugRotationMatrix;
                        debugTransformationMatrix.block<3,1>(0,3) = transformationMatrixQuery1C.block<3,1>(0,3);

                        Eigen::Matrix3f OCRoataionCalculated = transformationMatrixQuery1C.block<3,3>(0,0) * Eigen::Matrix3f(alphaRotation).inverse()*Eigen::Matrix3f(TrainSet[l][i]->dynFeature->pointPairTransformation.first);
                        Eigen::Vector3f OCTranslationCalculated = transformationMatrixQuery1C.block<3,3>(0,0) * Eigen::Matrix3f(alphaRotation).inverse()*TrainSet[l][i]->dynFeature->disVectorInQueryPixelCordinate.first +  Eigen::Vector3f(TrainSet[l][i]->statFeature->pixelLocation_real.x, TrainSet[l][i]->statFeature->pixelLocation_real.y, TrainSet[l][i]->statFeature->pixelLocation_real.z);

                        Eigen::Matrix4f OCTransformationCalculated =  Eigen::Matrix4f::Identity();
                        OCTransformationCalculated.block<3,3>(0,0) = OCRoataionCalculated;
                        OCTransformationCalculated.block<3,1>(0,3) = OCTranslationCalculated;

                        Surfel::addCoordinateSystem(debugTransformationMatrix, viewer,"3");
                        Surfel::addCoordinateSystem(OCTransformationCalculated, viewer,"4");
                        while (!viewer->wasStopped ()){
                            viewer->spinOnce (100);
                            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
                        }

                    }
                }
            }else{

                if(addPoseMeasure){

                    cv::Point2f center( pf->statFeature->iWidth/2.f, TrainSet[l][i]->statFeature->iHeight/2.f );
                    cv::Point3f s1 = CRPixel::P3toR3( pt1, center, pf->statFeature->imgAppearance[ 7 ].at< unsigned short >( pt1 )/1000.f );
                    cv::Point3f s2 = CRPixel::P3toR3( pt2, center, pf->statFeature->imgAppearance[ 7 ].at< unsigned short >( pt2 )/1000.f );
                    cv::Point3f q = pf->statFeature->pixelLocation_real;

                    pcl::Normal qn = pf->statFeature->normals->at( pf->statFeature->pixelLocation.x, pf->statFeature->pixelLocation.y );

                    // find transformation from query pixel coordinate system to object coordinate system
                    Eigen::Matrix4f transformationMatrixQuery1C, transformationMatrixQuery2C;
                    Surfel::calcQueryPoint2CameraTransformation(s1, s2, q, qn, transformationMatrixQuery1C, transformationMatrixQuery2C);

                    Eigen::Matrix4f transformationMatrixOQuery1 , transformationMatrixOQuery2;


                    // find transformation from query pixel coordinate system to object coordinate system
                    transformationMatrixOQuery1 = transformationMatrixQuery1C.inverse() * pf->statFeature->transformationMatrixOC;
                    transformationMatrixOQuery2 = transformationMatrixQuery2C.inverse() * pf->statFeature->transformationMatrixOC;

                    // Extract rotation matrix out of transformation matrix
                    Eigen::Matrix3f rotationMatrixOQuery1 = transformationMatrixOQuery1.block< 3, 3 >( 0, 0 );
                    Eigen::Matrix3f rotationMatrixOQuery2 = transformationMatrixOQuery2.block< 3, 3 >( 0, 0 );

                    // Convert rotation matrix to quaternions
                    Eigen::Quaternionf QuaternionOQuery1( rotationMatrixOQuery1 );
                    Eigen::Quaternionf QuaternionOQuery2( rotationMatrixOQuery2 );

                    pf->dynFeature->transformationMatrixOQuery_at_current_node.first = QuaternionOQuery1;
                    pf->dynFeature->transformationMatrixOQuery_at_current_node.second = QuaternionOQuery2;
                }
            }
            valSet[l][i].index = i;
        }
        sort( valSet[l].begin(), valSet[l].end() );
    }
}

void CRTree::split(vector<vector< PixelFeature*> >& SetA, vector<vector< PixelFeature*> >& SetB, vector<vector<int* > >& idA, vector<vector<int*> >& idB , const vector<vector< PixelFeature*> >& TrainSet, const vector<vector<IntIndex> >& valSet, const vector<vector<int*> >& TrainIDs ,int t) {
    for(unsigned int l = 0; l<TrainSet.size(); ++l) {
        // search largest value such that val<t
        vector<IntIndex>::const_iterator it = valSet[l].begin();
        while(it!=valSet[l].end() && it->val<t) {
            ++it;
        }

        SetA[l].resize(it-valSet[l].begin());
        idA[l].resize(SetA[l].size());
        SetB[l].resize(TrainSet[l].size()-SetA[l].size());
        idB[l].resize(SetB[l].size());

        it = valSet[l].begin();
        for(unsigned int i=0; i<SetA[l].size(); ++i, ++it) {
            SetA[l][i] = TrainSet[l][it->index];
            idA[l][i] = TrainIDs[l][it->index];
        }

        it = valSet[l].begin()+SetA[l].size();
        for(unsigned int i=0; i<SetB[l].size(); ++i, ++it) {
            SetB[l][i] = TrainSet[l][it->index];
            idB[l][i] = TrainIDs[l][it->index];
        }

    }
}

double CRTree::orientationMeanMC(const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB){

    vector<double> dist(num_labels, 0);

    // For setA
    Eigen::Quaternionf interpA1;
    Eigen::Quaternionf interpA2;

    vector< Eigen::Quaternionf > vTransA1;
    vector< Eigen::Quaternionf > vTransA2;

    //For setB
    Eigen::Quaternionf interpB1;
    Eigen::Quaternionf interpB2;

    vector< Eigen::Quaternionf > vTransB1;
    vector< Eigen::Quaternionf > vTransB2;


    float sample_pose = 0.9; // ignore with this probability

    for( unsigned int c = 0; c < num_labels; ++c ) {

        if( class_id[ c ] > 0 ) { // only for object class

            // For SetA
            vTransA1.clear();
            vTransA2.clear();

            vTransA1.reserve(SetA[c].size());
            vTransA2.reserve(SetA[c].size());

            for(vector< PixelFeature* >::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) { // for all training data of class c in SetA

                cv::RNG rng(cv::getCPUTickCount());
                float val = rng.uniform( 0.f, 1.f ); // generate a random number within [0,1]

                if( val > sample_pose || SetA[c].size()*sample_pose < 100.f ){

                    // 1 generate realtive transformation for first offset vector
                    Eigen::Quaternionf t1 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.first;

                    if(! ( isnan(t1.w()) || isnan(t1.x()) || isnan(t1.y()) || isnan(t1.z()) ) ){
                        // push quaternion into stack
                        vTransA1.push_back(t1);

                    }

                    // 1 generate realtive transformation for first offset vector
                    Eigen::Quaternionf t2 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.second;

                    if(! ( isnan(t2.w()) || isnan(t2.x()) || isnan(t2.y()) || isnan(t2.z()) ) ){
                        // push quaternion into stack
                        vTransA2.push_back(t2);
                    }
                }
            }

            // find interpolation for both the stacks of relative transformation
            interpA1 = quatInterp(vTransA1);
            interpA2 = quatInterp(vTransA2);

            Eigen::Quaternionf interpA1_inv = interpA1.inverse();
            Eigen::Quaternionf interpA2_inv = interpA2.inverse();

            // find dot product between mean and each quaternion and add it to dist
            for(vector< PixelFeature* >::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) { // for all training data of class c in Set

                // 1 generate realtive transformation for first offset vector
                Eigen::Quaternionf t1 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.first;

                if(! ( isnan(t1.w()) || isnan(t1.x()) || isnan(t1.y()) || isnan(t1.z()) ) ){
                    // find difference between quaternions
                    Eigen::Quaternionf diff = interpA1_inv * t1;

                    // convert diff into axis angle
                    Eigen::AngleAxisf diff_angleAxis(diff);

                    dist[c] += diff_angleAxis.angle();

                }

                // 2 generate realtive transformation for second offset vector
                Eigen::Quaternionf t2 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.second;

                if(! ( isnan(t2.w()) || isnan(t2.x()) || isnan(t2.y()) || isnan(t2.z()) ) ){
                    // find difference between quaternions
                    Eigen::Quaternionf diff = interpA2_inv * t2;

                    // convert diff into axis angle
                    Eigen::AngleAxisf diff_angleAxis(diff);

                    dist[c] += diff_angleAxis.angle();

                }

            }



            // For SetB
            vTransB1.clear();
            vTransB2.clear();

            vTransB1.reserve(SetB[c].size());
            vTransB2.reserve(SetB[c].size());

            for(vector< PixelFeature* >::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) { // for all training data of class c in SetA

                cv::RNG rng(cv::getCPUTickCount());
                float val = rng.uniform( 0.f, 1.f ); // generate a random number within [0,1]

                if( val > sample_pose || SetB[c].size()*sample_pose < 100.f ){

                    // 1 generate realtive transformation for first offset vector
                    Eigen::Quaternionf t1 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.first;

                    if(! ( isnan(t1.w()) || isnan(t1.x()) || isnan(t1.y()) || isnan(t1.z()) ) ){
                        // push quaternion into stack
                        vTransB1.push_back(t1);

                    }

                    // 2 generate realtive transformation for second offset vector
                    Eigen::Quaternionf t2 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.second;

                    if(! ( isnan(t2.w()) || isnan(t2.x()) || isnan(t2.y()) || isnan(t2.z()) ) ){
                        // push quaternion into stack
                        vTransB2.push_back(t2);
                    }
                }
            }

            // find interpolation for both the stacks of relative transformation
            interpB1 = quatInterp(vTransB1);
            interpB2 = quatInterp(vTransB2);

            Eigen::Quaternionf interpB1_inv = interpB1.inverse();
            Eigen::Quaternionf interpB2_inv = interpB2.inverse();

            // find dot product between mean and each quaternion and add it to dist
            for(vector< PixelFeature* >::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) { // for all training data of class c in Set

                // 1 generate realtive transformation for first offset vector
                Eigen::Quaternionf t1 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.first;

                if(! ( isnan(t1.w()) || isnan(t1.x()) || isnan(t1.y()) || isnan(t1.z()) ) ){
                    // find difference between quaternions
                    Eigen::Quaternionf diff = interpB1_inv * t1;

                    // convert diff into axis angle
                    Eigen::AngleAxisf diff_angleAxis(diff);

                    dist[c] += diff_angleAxis.angle();

                }

                // 2 generate realtive transformation for second offset vector
                Eigen::Quaternionf t2 = (*it)->dynFeature->transformationMatrixOQuery_at_current_node.second;

                if(! ( isnan(t2.w()) || isnan(t2.x()) || isnan(t2.y()) || isnan(t2.z()) ) ){
                    // find difference between quaternions
                    Eigen::Quaternionf diff = interpB2_inv * t2;

                    // convert diff into axis angle
                    Eigen::AngleAxisf diff_angleAxis(diff);

                    dist[c] += diff_angleAxis.angle();

                }

            }

        }// end if for class check
    }

    double Dist = 0;
    for(unsigned int c = 0; c < num_labels; ++c){

        Dist += dist[c];
    }

    return Dist;
}


// this code uses the class label!!!!
double CRTree::distMeanMC(const vector<vector< PixelFeature*> >& SetA, const vector<vector< PixelFeature*> >& SetB) {
    // calculating location entropy per class
    vector<double> meanAx(num_labels,0);
    vector<double> meanAy(num_labels,0);
    vector<double> meanAz(num_labels,0);
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            for(vector< PixelFeature*>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
                meanAx[c] += (*it)->statFeature->disVector.x;
                meanAy[c] += (*it)->statFeature->disVector.y;
                meanAz[c] += (*it)->statFeature->disVector.z;
            }
        }
    }

    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            meanAx[c] /= (double)SetA[c].size();
            meanAy[c] /= (double)SetA[c].size();
            meanAz[c] /= (double)SetA[c].size();
        }
    }

    vector<double> distA(num_labels,0);
    int non_empty_classesA = 0;
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            if (SetB[c].size() > 0)
                non_empty_classesA++;
            for(std::vector< PixelFeature*>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
                double tmp = (*it)->statFeature->disVector.x - meanAx[c];
                distA[c] += tmp*tmp;
                tmp = (*it)->statFeature->disVector.y - meanAy[c];
                distA[c] += tmp*tmp;
                tmp = (*it)->statFeature->disVector.z - meanAz[c];
                distA[c] += tmp*tmp;
            }
        }
    }

    vector<double> meanBx(num_labels,0);
    vector<double> meanBy(num_labels,0);
    vector<double> meanBz(num_labels,0);
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            for(vector< PixelFeature*>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
                meanBx[c] += (*it)->statFeature->disVector.x;
                meanBy[c] += (*it)->statFeature->disVector.y;
                meanBz[c] += (*it)->statFeature->disVector.z;

            }
        }
    }

    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            meanBx[c] /= (double)SetB[c].size();
            meanBy[c] /= (double)SetB[c].size();
            meanBz[c] /= (double)SetB[c].size();
        }
    }

    vector<double> distB(num_labels,0);
    int non_empty_classesB = 0;
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            if (SetB[c].size() > 0)
                non_empty_classesB++;

            for(std::vector< PixelFeature*>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
                double tmp = (*it)->statFeature->disVector.x - meanBx[c];
                distB[c] += tmp*tmp;
                tmp = (*it)->statFeature->disVector.y - meanBy[c];
                distB[c] += tmp*tmp;
                tmp = (*it)->statFeature->disVector.z - meanBz[c];
                distB[c] += tmp*tmp;
            }
        }
    }

    double Dist = 0;

    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            Dist += distA[c];
            Dist += distB[c];
        }
    }
    return Dist;
}

//double CRTree::distMeanMC_pose(const vector< vector< PixelFeature* > >& SetA, const vector<vector< PixelFeature* > >& SetB) {

//    // For setA

//    // calculating location entropy per class
//    vector<double> meanAx(num_labels,0);
//    vector<double> meanAy(num_labels,0);
//    vector<double> meanAz(num_labels,0);

//    vector<double> distA(num_labels,0);
//    int non_empty_classesA = 0;

//    for(unsigned int c = 0; c < num_labels; ++c) {

//        int weight = 0;

//        if(class_id[c] > 0) {
//            if (SetA[c].size() > 0)
//                non_empty_classesA++;
//            for(vector< PixelFeature* >::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
//                for(unsigned int p = 0; p < (*it)->alpha.size(); p++){

//                    // 1 generate affine transformation and transform disvector to the local coordinate
//                    float alpha1 = (*it)->alpha[p].first * 2.f * PI / 255.f;
//                    Eigen::Quaternionf Qnext1( cos(alpha1/2.f), 0.f, 0.f, sin(alpha1/2.f));
//                    Eigen::Vector3f disVecNext =  Eigen::Matrix3f(Qnext1) * (*it)->disVectorInQueryPixelCordinate.first;

//                    if(!(isnan(disVecNext[0]) ||isnan(disVecNext[1]) ||isnan(disVecNext[2]))){

//                        meanAx[c] += disVecNext[0];
//                        meanAy[c] += disVecNext[1];
//                        meanAz[c] += disVecNext[2];

//                        weight++;
//                    }

//                    // 2 generate affine transformation and transform disvector to the local coordinate
//                    float alpha2 = (*it)->alpha[p].second* 2.f * PI / 255.f;
//                    Eigen::Quaternionf Qnext2( cos(alpha2/2.f), 0.f, 0.f, sin(alpha2/2.f));
//                    disVecNext = Eigen::Matrix3f(Qnext2) * (*it)->disVectorInQueryPixelCordinate.second;

//                    if(!(isnan(disVecNext[0]) ||isnan(disVecNext[1]) ||isnan(disVecNext[2]))){

//                        meanAx[c] += disVecNext[0];
//                        meanAy[c] += disVecNext[1];
//                        meanAz[c] += disVecNext[2];

//                        weight++;
//                    }

//                }
//            }
//            meanAx[c] /= weight;
//            meanAy[c] /= weight;
//            meanAz[c] /= weight;

//            for(std::vector< PixelFeature*>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
//                double tmp = (*it)->disVector.x - meanAx[c];
//                distA[c] += tmp*tmp;
//                tmp = (*it)->disVector.y - meanAy[c];
//                distA[c] += tmp*tmp;
//                tmp = (*it)->disVector.z - meanAz[c];
//                distA[c] += tmp*tmp;
//            }

//        }
//    }

//    // For setB

//    // calculating location entropy per class
//    vector<double> meanBx(num_labels,0);
//    vector<double> meanBy(num_labels,0);
//    vector<double> meanBz(num_labels,0);

//    vector<double> distB(num_labels,0);
//    int non_empty_classesB = 0;

//    for(unsigned int c = 0; c < num_labels; ++c) {

//        int weight = 0;

//        if(class_id[c] > 0) {
//            if (SetB[c].size() > 0)
//                non_empty_classesB++;
//            for(vector< PixelFeature* >::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
//                for(unsigned int p = 0; p < (*it)->alpha.size(); p++){
//                    // 1 generate affine transformation and transform disvector to the local coordinate
//                    float alpha1 = (*it)->alpha[p].first* 2.f * PI / 255.f;
//                    Eigen::Quaternionf Qnext1( cos(alpha1/2.f), 0.f, 0.f, sin(alpha1/2.f));
//                    Eigen::Vector3f disVecNext = Eigen::Matrix3f(Qnext1) * (*it)->disVectorInQueryPixelCordinate.first;


//                    if(!(isnan(disVecNext[0]) ||isnan(disVecNext[1]) ||isnan(disVecNext[2]))){

//                        meanBx[c] += disVecNext[0];
//                        meanBy[c] += disVecNext[1];
//                        meanBz[c] += disVecNext[2];

//                        weight++;
//                    }


//                    // 2 generate affine transformation and transform disvector to the local coordinate
//                    float alpha2 = (*it)->alpha[p].second* 2.f * PI / 255.f;
//                    Eigen::Quaternionf Qnext2( cos(alpha2/2.f), 0.f, 0.f, sin(alpha2/2.f));
//                    disVecNext = Eigen::Matrix3f(Qnext2) * (*it)->disVectorInQueryPixelCordinate.second;


//                    if(!(isnan(disVecNext[0]) ||isnan(disVecNext[1]) ||isnan(disVecNext[2]))){

//                        meanBx[c] += disVecNext[0];
//                        meanBy[c] += disVecNext[1];
//                        meanBz[c] += disVecNext[2];

//                        weight++;
//                    }

//                }
//            }
//            meanBx[c] /= weight;
//            meanBy[c] /= weight;
//            meanBz[c] /= weight;

//            for(std::vector< PixelFeature*>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
//                double tmp = (*it)->disVector.x - meanBx[c];
//                distB[c] += tmp*tmp;
//                tmp = (*it)->disVector.y - meanBy[c];
//                distB[c] += tmp*tmp;
//                tmp = (*it)->disVector.z - meanBz[c];
//                distB[c] += tmp*tmp;
//            }

//        }
//    }



//    double Dist = 0;

//    for(unsigned int c = 0; c<num_labels; ++c) {
//        if(class_id[c]>0) {
//            Dist += distA[c];
//            Dist += distB[c];
//        }
//    }
//    return Dist;
//}

double CRTree::distMean(const vector<vector< PixelFeature*> >& SetA, const vector<vector< PixelFeature*> >& SetB) {
    // total location entropy (class-independent)
    double meanAx = 0;
    double meanAy = 0;
    double meanAz = 0;
    int countA = 0;
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            countA += SetA[c].size();
            for(vector< PixelFeature*>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
                meanAx += (*it)->statFeature->disVector.x;
                meanAy += (*it)->statFeature->disVector.y;
                meanAz += (*it)->statFeature->disVector.z;
            }
        }
    }


    meanAx /= (double)countA;
    meanAy /= (double)countA;
    meanAz /= (double)countA;

    double distA = 0;
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            for(std::vector< PixelFeature*>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
                double tmp = (*it)->statFeature->disVector.x - meanAx;
                distA += tmp*tmp;
                tmp = (*it)->statFeature->disVector.y - meanAy;
                distA += tmp*tmp;
                tmp = (*it)->statFeature->disVector.z - meanAz;
                distA += tmp*tmp;
            }
        }
    }

    double meanBx = 0;
    double meanBy = 0;
    double meanBz = 0;
    int countB = 0;
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            countB += SetB[c].size();
            for(vector< PixelFeature*>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
                meanBx += (*it)->statFeature->disVector.x;
                meanBy += (*it)->statFeature->disVector.y;
                meanBz += (*it)->statFeature->disVector.z;
            }
        }
    }

    meanBx /= (double)countB;
    meanBy /= (double)countB;
    meanBz /= (double)countB;

    double distB = 0;
    for(unsigned int c = 0; c<num_labels; ++c) {
        if(class_id[c]>0) {
            for(std::vector< PixelFeature*>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
                double tmp = (*it)->statFeature->disVector.x - meanBx;
                distB += tmp*tmp;
                tmp = (*it)->statFeature->disVector.y - meanBy;
                distB += tmp*tmp;
                tmp = (*it)->statFeature->disVector.z - meanBz;
                distB += tmp*tmp;
            }
        }
    }
    return distA + distB;
}


// optimization functions for class impurity

double CRTree::InfGain(const vector<vector< PixelFeature*> >& SetA, const vector<vector< PixelFeature*> >& SetB, const std::vector<float>& vRatio) {
    // get size of set A
    double sizeA = 0;
    vector<float> countA(SetA.size(),0);
    int count = 0;
    for(unsigned int i=0; i < SetA.size(); ++i) {
        sizeA += float(SetA[i].size())*vRatio[i];
        if(i>0 && class_id[i]!=class_id[i-1]) ++count;
        countA[count] += float(SetA[i].size())*vRatio[i];
    }

    double n_entropyA = 0;
    for(int i=0; i < count + 1; ++i) {
        double p = double( countA[i] ) / sizeA;
        if(p>0) n_entropyA += p*log(p);
    }

    // get size of set B
    double sizeB = 0;
    vector<float> countB(SetB.size(),0);
    count = 0;
    for(unsigned int i=0; i < SetB.size(); ++i) {
        sizeB += float(SetB[i].size())*vRatio[i];
        if(i>0 && class_id[i]!=class_id[i-1]) ++count;
        countB[count] += float(SetB[i].size())*vRatio[i];
    }

    double n_entropyB = 0;
    for(int i=0; i < count + 1; ++i) {
        double p = double( countB[i] ) / sizeB;
        if(p>0) n_entropyB += p*log(p);
    }

    return (sizeA*n_entropyA+sizeB*n_entropyB);
}

double CRTree::InfGainBG(const vector<vector< PixelFeature*> >& SetA, const vector<vector< PixelFeature*> >& SetB, const std::vector<float>& vRatio) {
    // get size of set A

    double sizeA = 0;
    vector<float> countA(SetA.size(),0);
    int count = 0;
    for(unsigned int i=0; i < SetA.size(); ++i) {
        if(i>0 && ((class_id[i]<=0 && class_id[i-1] >0) || (class_id[i] >0 && class_id[i-1] <=0) )) ++count;

        sizeA += float(SetA[i].size())*vRatio[i];
        countA[count] += float(SetA[i].size())*vRatio[i];
    }

    double n_entropyA = 0;
    for(int i=0; i < count + 1; ++i) {
        double p = double( countA[i] ) / sizeA;
        if(p>0) n_entropyA += p*log(p);
    }

    // get size of set B

    double sizeB = 0;
    vector<float> countB(SetB.size(),0);
    count = 0;
    for(unsigned int i=0; i < SetB.size(); ++i) {
        if(i>0 &&((class_id[i]<=0 && class_id[i-1] >0) || (class_id[i] >0 && class_id[i-1] <=0) )) ++count;

        sizeB += float(SetB[i].size())*vRatio[i];
        countB[count] += float(SetB[i].size())*vRatio[i];

    }

    double n_entropyB = 0;
    for(int i=0; i < count +1; ++i) {
        double p = double( countB[i] ) / sizeB;
        if(p>0) n_entropyB += p*log(p);
    }

    return (sizeA*n_entropyA+sizeB*n_entropyB);
}

