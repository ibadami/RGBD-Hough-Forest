#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct objectConfig{

   std::string name;
//   struct instances;
//   struct pitches;
//   struct pose;
   std::vector< std::vector< std::vector< std::string > > > filenames;
};

inline std::string convertInt2string(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

inline void loadConfig(const char* filename, std::vector<objectConfig> &objects){

    char buffer[1000];
    float num_objects;
    std::string objFile;

//    std::string extension  = '.png';


    std::ifstream in(filename);
    if( in.is_open() ) {

        in >> num_objects;
        objects.resize(num_objects);

        // read config file of each object
        for( int i = 0; i < num_objects; i++){

            // name of an object files
            in >> objFile;
            std::ifstream obj( objFile.c_str() );
            if( in.is_open() ){

                // name of an object
                obj.getline(buffer,1000);
                obj.getline(buffer,1000);
                objects[i].name = buffer;

                // number of instances
                obj.getline(buffer,1000);
                float num_instance;
                obj >> num_instance;
                obj.getline(buffer,1000);

                objects[i].filenames.resize(num_instance);

                // for each instance read pitch and pose
                for( int j = 0; j < num_instance; j++){

                    std::string instanceName = objects[ i ].name;
                    instanceName += "_";
                    instanceName += convertInt2string( j+1 );

                    obj.getline(buffer,1000);
                    obj.getline(buffer,1000);
                    obj.getline(buffer,1000);

                    // number of pitch per instance
                    float num_pitch;
                    obj >> num_pitch;
//                    obj.getline(buffer,1000);


                    objects[i].filenames[j].resize(num_pitch);
                    // for each pitch angle
                    for( int k = 0; k < num_pitch; k++){

                        float pitchIndex;
                        obj >> pitchIndex;
                        if( pitchIndex == k+1 ){

                            float num_poses;
                            obj >> num_poses;
                            obj.getline(buffer,1000);
                            objects[i].filenames[ j ][ k ].resize(num_poses);

                            // generate file name for each pose
                            for( int m = 0; m < num_poses; m++){

                                std::string poseName;
                                poseName = objects[ i ].name;
                                poseName += '_';
                                poseName += convertInt2string( j+1 );
                                poseName += '_';
                                poseName += convertInt2string( k+1 );
                                poseName += '_';
                                poseName += convertInt2string( m+1 );
                                std::string filename = objects[i].name + '/' +instanceName + '/' + poseName;

                                objects[i].filenames[j][k][m] = filename;

                            } // end for loop for each pose angle

                        }// end if statment

                    }// end for loop for each pitch angles

                }// end for loop for instances

            }// end if for object config file

        }// end for loop for objects

    }// end of if for config_all.txt

}// end of function

inline void readColors(std::string& filename, std::vector<cv::Point3i>& colors){

    int total_colors = 0;
    std::ifstream in(filename.c_str());
    if(in.is_open()){
        in >> total_colors;
        colors.resize(total_colors);
        for(int i = 0; i < total_colors; i++){

            in >> colors[i].x;
            in >> colors[i].y;
            in >> colors[i].z;

        }
    }
}



inline float readPose(std::string filename){

    float pose = -1;
    std::ifstream in(filename.c_str());
    if(in.is_open())
        in >> pose;
    return pose;

}


inline void readTable_wood(std::string filename, std::vector<std::string>& table_wood_image){

    char buffer[1000];
    float size;
    std::ifstream in(filename.c_str());
    if(in.is_open()){
        in >> size;
        table_wood_image.resize(size);
        in.getline(buffer, 1000);
        for(int i = 0; i < size; i ++){
        in.getline(buffer, 1000);
        table_wood_image[i] = buffer;
        }
    }
}

inline void readIntensity_gradient(std::string filename, std::vector<std::string>& intensity_gradient_images){

    char buffer[1000];
    float size;
    std::ifstream in(filename.c_str());
    if(in.is_open()){
        in >> size;
        intensity_gradient_images.resize(size);
        in.getline(buffer, 1000);
        for(int i = 0; i < size; i ++){
        in.getline(buffer, 1000);
        intensity_gradient_images[i] = buffer;
        }
    }



}

#endif // CONFIG_H
