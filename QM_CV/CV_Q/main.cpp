#include <iostream>
#include <opencv2/opencv.hpp>
//#include "opencv2/core.hpp"
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>

#include "calibrate.h"
#include "disparity.h"
#include "three_dimensional_proc.h"


using namespace std;
using namespace cv;
using namespace cuda;


/// Объявление переменных (временно)
/// \todo Убрать глобальные переменные, сделать их передачу как параметров
cv::Mat rectifiedLeft, rectifiedRight, imageLeft, imageRight;
cv::Mat disparity;
int click_counter = 0;

int mode = 1; // Смена алгоритмов Stereo


/// Объявление структур для хранения параметров калибровки
stereo_output_par_t stereo_par;

struct MouseCallbackData {
    //std::vector<KeyPoint> keypointsLeft;
    vector<vector<int>> leftKey;
    std::vector<Point2f> rightKey;
    cv::Mat disparity;
    cv::Mat depth;
    vector<vector<double>> points3D;
    double focalLenght;
    double baseline;
};


void onMouseClick(int event, int x, int y, int flags, void* userdata){
    cv::Mat* coords = static_cast<cv::Mat*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN){
        cv::Point clickPoint(x,y);
        std::cout << third_coords(imageLeft, imageRight, clickPoint, *coords) << std::endl;
    }
}


int main(int argc, char** argv) {
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    MouseCallbackData callbackData;

    std::vector<cv::String> imagesL, imagesR;   // Переменные, содержащие названия изображений в датасетах
    std::string pathL, pathR;                   // Переменные, содержащие пути к датасетам
    unsigned int num_set = 6;                   // Номер текущего используемого калибровочного датасета
    unsigned int num_set_stereo = 7;            // Номер текущего используемого тестового датасета
    int checkerboard_c;                         // Число ключевых точек по столбцам
    int checkerboard_r;                         // Число ключевых точек по строкам
    float square_size = 20.1;                   // Размер квадрата калибровочной доски в мм
    std::string name;                           // Наименование датасета (и yml-файла)
    bool isCalibrate = false;                   // Флаг принудительной калибровки

    // Выбор датасета для калибровки
    switch (num_set){
    case 0:
        pathL = "../../Fotoset/T_rep/left";
        pathR = "../../Fotoset/T_rep/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "0";
        break;
    case 1:
        pathL = "../../Fotoset/lab_set/left";
        pathR = "../../Fotoset/lab_set/right";
        checkerboard_c = 7;
        checkerboard_r = 4;
        name = "1";
        break;
    case 2:
        checkerboard_c = 9;
        checkerboard_r = 6;
        pathL = "../../Fotoset/basler_2_png/left";
        pathR = "../../Fotoset/basler_2_png/right";
        name = "2";
        break;
    case 3:
        pathL = "../../Fotoset/dataset_res/left";
        pathR = "../../Fotoset/dataset_res/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "3";
        break;
    case 4:
        pathL = "../../Fotoset/basler_festo/left";
        pathR = "../../Fotoset/basler_festo/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "4";
        break;
    case 5:
        pathL = "../../Fotoset/pairs/left";
        pathR = "../../Fotoset/pairs/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "5";
        break;
    case 6:
        pathL = "../../Fotoset/FEDOR/L";
        pathR = "../../Fotoset/FEDOR/R";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "6";
        break;
    default:
        pathL = "../../Fotoset/basler_festo/left";
        pathR = "../../Fotoset/basler_festo/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "7";
        break;
    }

    // Калибровка стереопары
    cv::FileStorage stereo_fs;
    if (stereo_fs.open("../../Calibration_parameters(stereo)/A" + name + "_stereo_camera_parameters.yml", cv::FileStorage::READ) && (!isCalibrate)){
        if (stereo_fs.isOpened()){
            stereo_fs["cameraMatrixL"]              >> stereo_par.cameraM1;
            stereo_fs["cameraMatrixR"]              >> stereo_par.cameraM2;
            stereo_fs["DistorsionCoeffsL"]          >> stereo_par.distCoeffs1;
            stereo_fs["DistorsionCoeffsR"]          >> stereo_par.distCoeffs2;
            stereo_fs["RotationMatrix"]             >> stereo_par.R;
            stereo_fs["TranslationMatrix"]          >> stereo_par.T;
            stereo_fs["EssentialMatrix"]            >> stereo_par.E;
            stereo_fs["FundamentalMatrix"]          >> stereo_par.F;
            stereo_fs["VectorOfRotationVectors"]    >> stereo_par.rvecs;
            stereo_fs["VectorOfTranslationVectors"] >> stereo_par.tvecs;
            stereo_fs["PerViewErrors"]              >> stereo_par.perViewErrors;
            stereo_fs["RMS"]                        >> stereo_par.RMS;
            stereo_fs.release();
          }
    } else {
        cout << "Stereo calibration procedure is running..." << endl;
        calibrate_with_mono(imagesL,imagesR, pathL, pathR, name, checkerboard_c, checkerboard_r,square_size, mono_parL, mono_parR, stereo_par);
    }

    // Вывод параметров стереопары
    print_stereo_camera_parameters(stereo_par);

    // Загрузка тестовых левого и правого изображений
    //cv::Mat imageLeft, imageRight;
    switch(num_set_stereo){
    case 0:
        imageLeft = cv::imread("../../Fotoset/Stereo/tests/left/L88.png");
        imageRight = cv::imread("../../Fotoset/Stereo/tests/right/R88.png");
        break;
    case 1:
        imageLeft = cv::imread("../../Fotoset/Stereo/tests/tele/left/R87.png");
        imageRight = cv::imread("../../Fotoset/Stereo/tests/tele/right/L87.png");
        break;
    case 2:
        imageLeft = cv::imread("../../Fotoset/Stereo/tests/any/view0.png");
        imageRight = cv::imread("../../Fotoset/Stereo/tests/any/view1.png");
        break;
    case 3:
        imageLeft = cv::imread("../../Fotoset/lab_set/left/left_var9_7.png");
        imageRight = cv::imread("../../Fotoset/lab_set/right/right_var9_7.png");
        break;
    case 4:
        imageLeft = cv::imread("../../Fotoset/Stereo/basler_festo/stereo_test/left/L1.png");
        imageRight = cv::imread("../../Fotoset/Stereo/basler_festo/stereo_test/right/R1.png");
        break;
    case 5:
        imageLeft = cv::imread("../../Fotoset/basler_2_png/test/left/LIm1.png");
        imageRight = cv::imread("../../Fotoset/basler_2_png/test/right/RIm1.png");
        break;
    case 6:
        imageLeft = cv::imread("../../Fotoset/Stereo/tests/fish/photo_left.png");
        imageRight = cv::imread("../../Fotoset/Stereo/tests/fish/photo_right.png");
        break;
    case 7:
        imageLeft = cv::imread("../../Fotoset/FEDOR/L_27_4_37.jpg");
        imageRight = cv::imread("../../Fotoset/FEDOR/R_27_4_37.jpg");
        break;
    default:
        break;
    }


    cv::Mat pointsAll;
    point3d_finder(imageLeft, imageRight, pointsAll);

    //cv::cvtColor(imageLeft, imageLeft, cv::COLOR_GRAY2BGR);

    cv::imshow("Res", imageLeft);
    cv::setMouseCallback("Res", onMouseClick, &pointsAll);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}








