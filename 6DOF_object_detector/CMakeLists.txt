cmake_minimum_required ( VERSION 2.8 )

# The name of our project is "6DOF_object_detector". 
project( 6DOF_object_detector ) 

# cpp flags
SET( CMAKE_CXX_FLAGS "-O3 -Wno-deprecated-declarations -Wno-deprecated -Wno-unused-result -mssse3 -DEIGEN_NO_DEBUG -fopenmp" )

FIND_PACKAGE( PCL 1.6 REQUIRED )
FIND_PACKAGE( OpenCV 2.4.5 REQUIRED )

# header dir for project 
SET ( INCLUDE_DIR ${6DOF_object_detector_SOURCE_DIR}/include/ )

# header files *.h
FILE ( GLOB HEADERS ${6DOF_object_detector_SOURCE_DIR}/include/*.h )

# source files *.c
FILE ( GLOB SRCS ${6DOF_object_detector_SOURCE_DIR}/src/*.cpp )

# set all include directories
INCLUDE_DIRECTORIES ( ${INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${INCLUDE_EIGEN_DIRS} )

# set all libraries
SET ( LIBS ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SEARCH_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_FILTERS_LIBRARIES} ${OpenCV_LIBS} )

# define the executable 
ADD_EXECUTABLE ( 6DOF_object_detector ${SRCS} ${HEADERS} )

# link libraries with executable
TARGET_LINK_LIBRARIES ( 6DOF_object_detector ${LIBS} )

# install binary to set destination
INSTALL_TARGETS(/home/badami/Codes/RGBD_Hough_Forest/6DOF_object_detector 6DOF_object_detector)
