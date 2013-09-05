# change paths if necessary
#INCLUDES = -I/usr/include/pcl-1.6 -I/usr/include/vtk-5.8 -I/usr/include/eigen3
#LIBDIRS  = -L/usr/lib

INCLUDES = -I/home/badami/usr/include/pcl-1.6 -I/home/badami/usr/include/vtk-5.8 -I/home/badami/usr/include/eigen3 -I/home/badami/usr/include
LIBDIRS  = -L/home/badami/usr/lib -L/home/badami/usr/lib/vtk-5.8
LIBS     = -lboost_thread -lvtkfreetype -lvtkRendering -lvtkFiltering -lopencv_core -lpcl_common -lpcl_visualization -lvtkCommon -lopencv_highgui -lpcl_search -lpcl_kdtree -lpcl_features -lopencv_imgproc -lopencv_calib3d -lpcl_surface -lpcl_segmentation -lpcl_filters

OPT      = -O3 -Wno-deprecated-declarations -Wno-deprecated -Wno-unused-result -mssse3 -DEIGEN_NO_DEBUG -fopenmp
CXX      = ccache g++
OBJS     = CRForest-Detector.o Pixel.o HoG.o CRForestDetector.o CRTree.o Candidate.o Surfel.o utils.o CRForestTraining.o
			
all: CRForest-Detector
	echo all: make complete
%.o:%.cpp
	$(CXX) -c $(INCLUDES) $+ $(OPT)
	
CRForest-Detector: $(OBJS)
	$(CXX) -o $@ $+ $(LIBDIRS) $(LIBS)  $(OPT)

clean:
	rm -f *.o *~ CRForest-Detector

.PHONY: all clean
