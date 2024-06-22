#include <iostream>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>

#include "../../calibrate.h"



int main(int argc, char** argv){
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;
    stereo_output_par_t stereo_par;


    std::vector<cv::String> imagesL, imagesR;
    std::string pathL = "../../../../Fotoset/FEDOR/L";
    std::string pathR = "../../../../Fotoset/FEDOR/R";

    int checkerboard_c = 9;                 // Число ключевых точек по столбцам
    int checkerboard_r = 6;                 // Число ключевых точек по строкам
    float square_size = 20.1;               // Размер квадрата калибровочной доски в мм
    std::string params_filename = "test_calib.yml";



    cv::FileStorage stereo_fs;
    if (stereo_fs.open(params_filename, cv::FileStorage::READ)){
        stereo_par = read_stereo_params(params_filename);
    } else {
        cout << "Stereo calibration procedure is running..." << endl;
        calibrate_with_mono(imagesL,imagesR, pathL, pathR, mono_parL, mono_parR, stereo_par,  checkerboard_c, checkerboard_r, square_size);
    }


    print_stereo_camera_parameters(stereo_par);
    write_stereo_params(params_filename,stereo_par);

    return 0;
}
