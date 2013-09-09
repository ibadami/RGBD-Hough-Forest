/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
//
// Modified: Nima Razavi, BIWI, ETH Zurich
// Email: nrazavi@vision.ee.ethz.ch
*/

// Modified by: Ishrat Badami, AIS, Uni-Bonn
// Email:       badami@vision.rwth-aachen.de

#pragma once

#define sprintf_s sprintf


#include <iostream>
#include <fstream>

#include "Surfel.h"
#include "Pixel.h"

// Auxilary structure
struct IntIndex {
    int val;
    unsigned int index;
    bool operator<(const IntIndex& a) const { return val<a.val; }
};

struct DynamicFeature{
    
    DynamicFeature(){}
    
    std::pair< Eigen::Vector3f, Eigen::Vector3f > disVectorInQueryPixelCordinate;
    std::pair< Eigen::Vector3f, Eigen::Vector3f > tempDisVectorInQueryPixelCordinate;
    std::pair< Eigen::Quaternionf, Eigen::Quaternionf > pointPairTransformation;
    
    std::pair< Eigen::Matrix3f, Eigen::Matrix3f > firstQuerytoCameraTransformation;
    
    std::pair< Eigen::Quaternionf, Eigen::Quaternionf > transformationMatrixOQuery_at_current_node;
    
    std::vector< std::pair < float, float > > alpha;
    
};


// Structure for the leafs
struct LeafNode {
    // Constructors
    LeafNode() {}

    // IO functions
    const void show(int delay, int width, int height, int* class_id);
    const void print() const {
        std::cout << "Leaf " << vCenter.size() << " ";
        for(unsigned int c = 0; c<vCenter.size(); ++c)
            std::cout << vCenter[c].size() << " "  << vPrLabel[c] << " ";
        std::cout << std::endl;
    }
    int depth;
    int parent;
    float cL; // what proportion of the entries at this leaf is from foreground
    int idL; // leaf id
    //    float fL; //occurrence frequency
    //    float eL;// emprical probability of when a pixel is matched to this cluster, it belons to fg
    std::vector<int> nOcc;
    std::vector<float> vLabelDistrib;

    // Probability of foreground
    std::vector<float> vPrLabel;

    // Vectors from object center to training pixeles
    std::vector<std::vector<cv::Point3f> > vCenter;
    std::vector<std::vector<float> > vCenterWeights;
    std::vector<std::vector<int> > vID;
    std::vector<std::vector<cv::Point3f > > bbSize3D;
    std::vector<std::vector<std::pair< Eigen::Quaternionf, Eigen::Quaternionf > > > vPose;
    std::vector<std::vector<std::pair< Eigen::Vector3f, Eigen::Vector3f > > > QdisVector;

    std::vector < std::vector< std::vector< std::pair< float, float > > > > alpha;

};

// Structure for internal Nodes
struct InternalNode {
    // Constructors
    InternalNode() {}

    // Copy Constructor
    InternalNode(const InternalNode& arg){
        parent = arg.parent;
        leftChild = arg.leftChild;
        rightChild = arg.rightChild;
        idN = arg.idN;
        depth = arg.depth;
        data.resize(arg.data.size());
        for (unsigned int dNr=0; dNr < arg.data.size(); dNr++)
            data[dNr] = arg.data[dNr];
        isLeaf = arg.isLeaf;
    }

    // relative node Ids
    int parent; // parent id, if this node is root, the parent will be -1
    int leftChild; // stores the left child id, if leaf stores the leaf id
    int rightChild;// strores the right child id, if leaf is set to -1

    //internal data
    int idN;//node id

    int depth;

    //	the data inside each not
    std::vector<int> data;// x1 y1 x2 y2 channel threshold
    bool isLeaf;// if leaf is set to 1 otherwise to 0, the id of the leaf is stored at the left child

};

struct HNode {
    HNode() {}

    // explicit copy constructor
    HNode(const HNode& arg){
        id = arg.id;
        parent = arg.parent;
        leftChild = arg.leftChild;
        rightChild = arg.rightChild;
        subclasses = arg.subclasses;
        linkage = arg.linkage;
    }

    bool isLeaf(){
        return ((leftChild < 0) && (rightChild < 0));
    }

    int id;
    int parent;// stores the id of the parent node: if root -1
    int leftChild; // stores the id of the left child, if leaf -1
    int rightChild;// stores the id of the right child, if leaf -1
    float linkage;
    std::vector<int> subclasses; // stores the id of the subclasses which are under this node,
};

class CRTree {
public:
    // Constructors
    CRTree(const char* filename, bool& success);
    CRTree(int min_s, int max_d, int l, cv::RNG* pRNG) : min_samples(min_s), max_depth(max_d), num_leaf(0), num_nodes(1), num_labels(l), cvRNG(pRNG) {

        nodes.resize(int(num_nodes));
        nodes[0].isLeaf = false;
        nodes[0].idN = 0; // the id is set to zero for the root
        nodes[0].leftChild = -1;
        nodes[0].rightChild = -1;
        nodes[0].parent = -1;
        nodes[0].data.resize(6,0);
        nodes[0].depth = 0;

        //initializing the leafs
        leafs.resize(0);
        // class structure
        class_id = new int[num_labels];
    }
    ~CRTree() {}//clearLeaves(); clearNodes();

    // Set/Get functions
    unsigned int GetDepth() const { return max_depth; }
    unsigned int GetNumLabels() const { return num_labels; }
    void setClassId( const std::vector< int >& id ) {
        for( unsigned int i = 0;i < num_labels; ++i ) class_id[ i ] = id[ i ];
    }
    void setObjectSize(std::pair<float, float> objectSize){

        class_size.resize(num_labels-1);
        for( unsigned int i = 0; i < num_labels; ++i ){

            if(class_id > 0){
                class_size[i].first = objectSize.first;
                class_size[i].second = objectSize.second;
            }
        }
    }

    void getClassId( std::vector< int >& id ) const {
        id.resize(num_labels);
        for( unsigned int i = 0; i < num_labels; ++i ) id[ i ] = class_id[ i ];
    }
    float GetScale() {return scale;}
    void SetScale(const float tscale) {scale = tscale;}

    int getNumLeaf(){return num_leaf;}
    int getNumNodes(){return num_nodes;}
    LeafNode* getLeaf(int leaf_id = 0){return &leafs[leaf_id];}
    InternalNode* getNode(int node_id = 0){return &nodes[node_id];}

    void setTrainingMode(int mode){training_mode = mode;}

    bool GetHierarchy( std::vector< HNode > &h ){
        if ( (hierarchy.size() == 0) ) { // check if the hierarchy is set at all(hierarchy == NULL) ||
            return false;
        }
        h = hierarchy;
        return true;
    }

    // Regression
    int regression(const std::vector<cv::Mat> &vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, cv::Point &pt, float &scale) const;

    // Training
    void growTree( const Parameters& param,  const CRPixel& TrData, int samples, int trNr, std::vector< std::vector< int > > numbers );

    // IO functions
    bool saveTree(const char* filename) const;
    bool loadHierarchy(const char* filename);
    std::vector< std::vector<DynamicFeature*> > dynFeatureSet;

private:

    // Private functions for training
    void grow(const Parameters& param, const vector< vector< PixelFeature*> >& TrainSet, vector<vector< DynamicFeature*> >& dynFeatures, const vector<vector<int> >& TrainIDs, int node, unsigned int depth, int samples, vector<float>& vRatio, int trNr) ;

    int getStatSet(const std::vector<std::vector< PixelFeature*> >& TrainSet, int* stat);

    void makeLeaf(const std::vector<std::vector< PixelFeature*> >& TrainSet, const std::vector<std::vector< DynamicFeature*> >& dynFeatures, const std::vector<std::vector< int> >& TrainIDs, std::vector<float>& vRatio, int node);

    bool optimizeTest(const Parameters& param, vector<vector< PixelFeature*> >& SetA, vector<vector< PixelFeature*> >& SetB, vector<vector< DynamicFeature*> >& dynA, vector<vector< DynamicFeature*> >& dynB, vector<vector<int> >& idA, vector<vector<int> >& idB , const vector<vector<  PixelFeature*> >& TrainSet, vector<vector<  DynamicFeature*> >& dynFeatures, const vector<vector<int> >& TrainIDs , int* test, unsigned int iter, unsigned int measure_mode,const std::vector<float>& vRatio, int node);

    void generateTest(const Parameters& p, int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c);

    void evaluateTest( std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector< std::vector< PixelFeature*> >& TrainSet, std::vector<std::vector< DynamicFeature*> >& dynFeatures, int node, bool addTransformation, bool addPoseMeasure);

    void split(vector< std::vector< PixelFeature* > >& SetA, vector< std::vector< PixelFeature* > >& SetB, vector< std::vector< DynamicFeature* > >& dynA, vector< std::vector< DynamicFeature* > >& dynB, vector< std::vector< int > >& idA, vector< std::vector< int > >& idB, const vector< std::vector< PixelFeature* > >& TrainSet, vector< std::vector< DynamicFeature* > >& dynFeatures, const vector< std::vector< int > >& TrainIDs, const vector< vector< IntIndex > >& valSet, int t);

    double measureSet( const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB,  const std::vector<std::vector< DynamicFeature*> >& dynA, const std::vector<std::vector< DynamicFeature*> >& dynB, unsigned int mode, const std::vector<float>& vRatio, bool addPoseMeasure) {
 
        if ( mode == 0 || mode == -1 ) {

            if ( training_mode == 0 ){ // two class information gain
                return InfGain( SetA, SetB, vRatio );

            }else if( training_mode == 1 ){// multiclass infGain with background
                return InfGainBG( SetA, SetB, vRatio ) + InfGain( SetA, SetB, vRatio) / double( SetA.size() );

            }else if( training_mode == 2 ){ // multiclass infGain without background
                return InfGain( SetA, SetB, vRatio );

            }else{
                std::cerr << " there is no method associated with the training mode: " << training_mode << std::endl;
                return -1;

            }
        }else {

            // check if pose measure is true
            if( addPoseMeasure ){
                if( mode == 1 )
                    // calculate pose measure
                    return -orientationMeanMC( SetA, SetB , dynA, dynB);
                else{

                    if ( training_mode == 2 || training_mode == 0 ){
                        return -distMean( SetA, SetB );
                    }else{
                        return -distMeanMC( SetA, SetB );
                    }
                }

            }else{

                if ( training_mode == 2 || training_mode == 0 ){
                    return -distMean( SetA, SetB );
                }else{
                    return -distMeanMC( SetA, SetB );
                }

            }

        }// end of if else for mode
    }

    double distMean(const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB);

    double distMeanMC(const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB);

    double orientationMeanMC(const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB, const std::vector<std::vector< DynamicFeature*> >& dynA, const std::vector<std::vector< DynamicFeature*> >& dynB);

    double distMeanMC_pose(const vector< vector< PixelFeature* > >& SetA, const vector<vector< PixelFeature* > >& SetB) ;

    double InfGain(const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB, const std::vector<float>& vRatio);

    double InfGainBG(const std::vector<std::vector< PixelFeature*> >& SetA, const std::vector<std::vector< PixelFeature*> >& SetB, const std::vector<float>& vRatio);


    // Data structure

    // tree table
    // 2^(max_depth+1)-1 x 7 matrix as vector
    // column: leafindex x1 y1 x2 y2 channel thres
    // if node is not a leaf, leaf=-1
    //int* treetable;

    // stop growing when number of Pixeles is less than min_samples
    unsigned int min_samples;

    // depth of the tree: 0-max_depth
    unsigned int max_depth;

    // number of nodes: 2^(max_depth+1)-1
    unsigned int num_nodes;

    // number of leafs
    unsigned int num_leaf;

    // number of labels
    unsigned int num_labels;

    // classes
    int* class_id;
    std::vector<std::pair<float, float> > class_size;

    int training_mode;// 1 for multi-class detection

    // scale of the training data with respect to some reference
    float scale;
    //leafs as vector
    std::vector<LeafNode> leafs;

    // internalNodes as vector
    std::vector<InternalNode> nodes;// the first element of this is the root

    // hierarchy as vector
    std::vector<HNode> hierarchy;
    cv::RNG *cvRNG;
};

inline int CRTree::regression(const std::vector<cv::Mat> &vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, cv::Point &pt, float &scale) const {
    // pointer to the current node first set to the root node
    //InternalNode* pnode = &nodes[0];

    int node = 0;
    int p1,  p2;
    bool test ;


    cv::Point pt1,pt2;
    // Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
    while(!nodes[node].isLeaf) {
        // binary test 0 - left, 1 - right
        // Note that x, y are changed since the Pixeles are given as matrix and not as image
        // p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false

        // pointer to channel
        //cv::Mat ptC = vImg[nodes[node].data[4]];

        pt1.x = std::max(0, int(pt.x + nodes[node].data[0]*scale));
        pt1.x = std::min(pt1.x, vImg[0].cols-1);

        pt1.y = std::max(0, int(pt.y + nodes[node].data[1]*scale));
        pt1.y = std::min(pt1.y, vImg[0].rows-1);

        pt2.x = std::max(0, int(pt.x + nodes[node].data[2]*scale));
        pt2.x = std::min(pt2.x, vImg[0].cols-1);

        pt2.y = std::max(0, int(pt.y + nodes[node].data[3]*scale));
        pt2.y = std::min(pt2.y, vImg[0].rows-1);


        // get pixel values
        if(nodes[node].data[4] < vImg.size() && nodes[node].data[4] != 7 && nodes[node].data[4] != 15 && nodes[node].data[4] != 24 ){
            p1 = vImg[nodes[node].data[4]].at< unsigned char >(pt1);
            p2 = vImg[nodes[node].data[4]].at< unsigned char >(pt2);
            // test
            test = ( p1 - p2 ) >= nodes[node].data[5];
        }else if(nodes[node].data[4] == 7 || nodes[node].data[4] == 15 || nodes[node].data[4] == 24){
            p1 = vImg[nodes[node].data[4]].at< unsigned short >(pt1);
            p2 = vImg[nodes[node].data[4]].at< unsigned short >(pt2);
            // test
            test = ( p1 - p2 ) >= nodes[node].data[5];

        }else{

            SurfelFeature sf;
            Surfel::computeSurfel(normals, cv::Point2f(pt1.x, pt1.y), cv::Point2f(pt2.x, pt2.y), cv::Point2f(vImg[0].cols/2.f, vImg[0].rows/2.f), sf, vImg[7].at<unsigned short>(pt1)/1000.f, vImg[7].at<unsigned short>(pt2)/1000.f  );
            test = sf.fVector[nodes[node].data[4] - vImg.size()] >= nodes[node].data[5];

        }

        // next node is at the left or the right child depending on test
        if(isnan(test)){
            return -1;
            cout<< "reached here"<< endl ;
            break;
        }

        if (test)
            node = nodes[node].rightChild;
        else
            node = nodes[node].leftChild;
    }
    if(nodes[node].leftChild > num_leaf)
        cout<<"something is wrong"<<endl;

    return nodes[node].leftChild;
}

inline void CRTree::generateTest(const Parameters& p, int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c) {
    //	cv::Point pt1, pt2;

    float scale_factor = 0.8f;

    test[ 0 ] = cvRNG->operator ()( max_w * scale_factor ) - max_w * scale_factor / 2.0f;
    test[ 1 ] = cvRNG->operator ()( max_h * scale_factor ) - max_h * scale_factor / 2.0f;
    test[ 2 ] = cvRNG->operator ()( max_w * scale_factor ) - max_w * scale_factor / 2.0f;
    test[ 3 ] = cvRNG->operator ()( max_h * scale_factor ) - max_h * scale_factor / 2.0f;

    if( p.addSurfel && p.addIntensity )
        test[ 4 ] = cvRNG->operator ()( max_c + 4 );  //max_c  + 4 dimension for surfel feature
    else if(!p.addSurfel && p.addIntensity)
        test[ 4 ] = cvRNG->operator ()( max_c ) ;
    else if( p.addSurfel && !p.addIntensity )
        test[ 4 ] = cvRNG->operator()(max_c + 4 -1) +1;
    else
         test[ 4 ] = cvRNG->operator()( max_c - 1) + 1;

}

inline int CRTree::getStatSet(const std::vector<std::vector< PixelFeature*> >& TrainSet, int* stat){

    int count = 0;
    for( unsigned int l = 0; l < TrainSet.size(); ++l ) {
        if( TrainSet[l].size() > 0 )
            stat[ count++ ] = l;
    }
    return count;
}
