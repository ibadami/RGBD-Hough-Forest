/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "Tree.h"
#include "Trainer.h"

#include <vector>

class CRForest {
public:
    // Constructors
    CRForest(int trees = 0, bool doSkip=true) {
        vTrees.resize(trees);
        do_skip = doSkip;

    }
    ~CRForest() {
        for(std::vector<CRTree*>::iterator it = vTrees.begin(); it != vTrees.end(); ++it) delete *it;
        vTrees.clear();
    }

    // Set/Get functions
    void SetTrees(int n) {vTrees.resize(n);}
    int GetSize() const {return vTrees.size();}
    unsigned int GetDepth() const {return vTrees[0]->GetDepth();}
    unsigned int GetNumLabels() const {return vTrees[0]->GetNumLabels();}
    void GetClassID(std::vector<std::vector<int> >& id) const;
    LeafNode* getLeaf(int treeId, int leafId) const {return vTrees[treeId]->getLeaf(leafId);}
    InternalNode* getNode(int treeId, int nodeId) const {return vTrees[treeId]->getNode(nodeId);}
    bool GetHierarchy(std::vector<HNode>& hierarchy) const{
        return vTrees[0]->GetHierarchy(hierarchy);
    }
    void SetTrainingLabelsForDetection(std::vector<int>& class_selector);
    void GetTrainingLabelsForDetection(std::vector<int>& class_selector);

    // Regression
    void regression(std::vector<int>& result, const std::vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, cv::Point &pt, float &scale) const;
    void regression(std::vector<const LeafNode*>& result, std::vector<unsigned int>& trID, uchar** ptFCh, int stepImg, CvRNG* pRNG, double thresh ,float scale_tree = -1.0f) const;

    // Training
    void trainForest(const Parameters& p, rawData& data, int min_s,  int samples ) ;

    // IO functions
    void saveForest(string filename, unsigned int offset = 0);
    bool loadForest(string filename, unsigned int offset = 0);
    void loadHierarchy(const char* hierarchy, unsigned int offset=0);

    // Trees
    std::vector<CRTree*> vTrees;

    // training labels to use for detection
    std::vector<int>  use_labels;

    // skipping training
    bool do_skip;

    // decide what kind of training procedures to take
    int training_mode;// the normal information gain
    // the training mode=0 does the InfGain over all classes
    // the mode 1 transforms all the positive class_ids into different labels and does multi-class training with InfGain/nlabels + InfGainBG
    // the mode 3 also transforms all the positive class ids into one label and does all-against-background training with InfGainBG
};

// Matching
inline void CRForest::regression(std::vector<int>& result, const std::vector<cv::Mat>& vImg, const pcl::PointCloud<pcl::Normal>::Ptr& normals, cv::Point &pt, float &scale) const {
    result.resize( vTrees.size() );
    for(int i=0; i<(int)vTrees.size(); ++i) {
        result[i] = vTrees[i]->regression(vImg, normals, pt, scale);
    }
}

//Training
inline void CRForest::
trainForest(const Parameters& p, rawData& data, int min_s,  int samples ) {

    // Init random generator
    time_t t = time( NULL );
    int seed = ( int )(t/double( p.off_tree + 1 ) );
    cv::RNG pRNG = cv::RNG( seed );

    // Init training data
    CRPixel TrData( &pRNG );
    TrData.setClasses( p.nlabels );
    // Extract training patches

    CRForestTraining::extract_Pixels( data, p, TrData, &pRNG);


#pragma omp parallel for
    for( int i = p.off_tree; i < ( int )vTrees.size(); ++i ){

        CRTree* Trees = new CRTree( min_s, p.treedepth, TrData.vRPixels.size(), &pRNG );
        Trees->setClassId( p.class_structure );
        Trees->SetScale( p.scale_tree );
        Trees->setTrainingMode( p.training_mode );
        Trees->setObjectSize( p.objectSize );
        Trees->growTree( p, TrData, samples, i );

        char buffer[ 200 ];

        sprintf_s( buffer, "%s%03d.txt", (p.treepath + "/treetable").c_str(), i );
        Trees->saveTree( buffer);

    }
}

// IO Functions
inline void CRForest::saveForest( string filename, unsigned int offset ) {
    char buffer[ 200 ];
    for( unsigned int i = offset ; i < vTrees.size(); ++i ) {
        sprintf_s( buffer, "%s%03d.txt", filename.c_str(), i );
        vTrees[ i ]->saveTree( buffer );
    }
}

inline bool CRForest::loadForest( string filename, unsigned int offset ) {

    char buffer[ 200 ];
    bool final_success = true;
    //    bool success = true;
    vector<bool>success(vTrees.size());

#pragma omp parallel for private(buffer)
    for(unsigned int i = offset; i < vTrees.size(); ++i ) {
        sprintf_s( buffer, "%s%03d.txt", (filename + "/treetable").c_str(), i );
        bool s;
        vTrees[ i-offset ] = new CRTree( buffer, s );
        success[ i-offset ] = s;
        //        success = s;

    }

    for(int i = 0; i < vTrees.size(); i++ )
        if(success[i]!=true)
            final_success = false;
    return final_success;
    //    return success;
}

inline void CRForest::loadHierarchy(const char* hierarchy, unsigned int offset){
    //char buffer[400];
    int cccc =0;
    for (unsigned int i = offset; i < vTrees.size(); ++i,++cccc){
        if(!(vTrees[cccc]->loadHierarchy(hierarchy))){
            std::cerr<< "failed to load the hierarchy: " << hierarchy << std::endl;
        }else{
            std::cout<< "loaded the hierarchy: " << hierarchy << std::endl;
        }
    }
}

// Get/Set functions
inline void CRForest::GetClassID( std::vector< std::vector< int > >& id ) const {
    id.resize( vTrees.size() );
    for( unsigned int i = 0; i < vTrees.size(); ++i ) {
        vTrees[ i ]->getClassId( id[ i ]);
    }
}

inline void CRForest::SetTrainingLabelsForDetection(std::vector<int>& class_selector){
    int nlabels = GetNumLabels();
    if (class_selector.size() == 1 && class_selector[0]==-1){
        use_labels.resize(nlabels);
        std::cout<< nlabels << " labels used for detections:" << std::endl;
        for ( int i=0; i < nlabels; i++){
            use_labels[i] = 1;

        }
    }else{
        if ((unsigned int)(nlabels)!= class_selector.size()){
            std::cerr<< "nlabels: " << nlabels << " class_selector.size(): " << class_selector.size() << std::endl;
            std::cerr<< "CRForest.h: the number of labels does not match the number of elements in the class_selector" << std::endl;
            return;
        }
        use_labels.resize(class_selector.size());
        for (unsigned int i=0; i< class_selector.size(); i++){
            use_labels[i] = class_selector[i];
        }
    }

}

inline void CRForest::GetTrainingLabelsForDetection(std::vector<int>& class_selector){
    class_selector.resize(use_labels.size());
    for (unsigned int i=0; i< use_labels.size(); i++)
        class_selector[i] = use_labels[i];
}
