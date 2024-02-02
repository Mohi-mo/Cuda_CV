#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>
//#include <opencv2/cudaarithm.hpp>

#include <opencv2/imgproc/imgproc.hpp>
//#include <QApplication>
//#include <QMouseEvent>
//#include <QLabel>

using namespace std;
using namespace cv;
using namespace cuda;

#include "calibrate.h"

// @todo сделать триангуляцию точек на стереоизображение
//          протестировать триангуляцию на
//          автоматизировать алгоритм нахождения ключевых точек на изображении


// Проверка работоспособности CUDA
void CUDA_work_check(){
    printShortCudaDeviceInfo(getDevice());
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
}


int main(int argc, char** argv) {
    // Калибровка через захват видео
    /*
    cv::VideoCapture leftCamera(0);
    cv::VideoCapture rightCamera(1);

    if (!leftCamera.isOpened() || !rightCamera.isOpened()) {
        std::cerr << "Error: Could not open cameras." << endl;
        return -1;
    }

    // Код вставить сюда

    leftCamera.release();
    rightCamera.release();
    */

    // Калибровка камер через заготоволенный датасет
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    std::vector<cv::String> imagesL, imagesR;
    //string pathL = "../../../Fotoset/Left";
    //string pathR = "../../../Fotoset/Right";
    //string pathL = "../../Fotoset/lab_set/left";
    //string pathR = "../../Fotoset/lab_set/right";
    string pathL = "../../Fotoset/T_rep/left";
    string pathR = "../../Fotoset/T_rep/right";

    calibrate_camera(imagesL, pathL,  mono_parL);
    calibrate_camera(imagesR, pathR,  mono_parR);

    // Вывод параметров камер
    cout << "\n\n\t Left camera\n";
    cout << "cameraMatrix: "        << mono_parL.cameraMatrix     << endl;
    //cout << "distCoeffs: "         << mono_parL.distCoeffs       << endl;
    //cout << "Per view errors: "     << mono_parL.perViewErrors    << endl;
    //cout << "STD Intrinsics: "      << mono_parL.stdDevIntrinsics << endl;
    //cout << "STD Extrinsics: "      << mono_parL.stdDevExtrinsics << endl;
    //cout << "Rotation vector: "     << mono_parL.rvecs            << endl;
    //cout << "Translation vector: "  << mono_parL.tvecs            << endl;
    cout << "RMS: "                 << mono_parL.RMS              << endl;

    cout << "\n\n\t Right camera\n";
    cout << "cameraMatrix: "        << mono_parR.cameraMatrix     << endl;
    //cout << "distCoeffs: "         << mono_parR.distCoeffs       << endl;
    //cout << "Per view errors: "     << mono_parR.perViewErrors    << endl;
    //cout << "STD Intrinsics: "      << mono_parR.stdDevIntrinsics << endl;
    //cout << "STD Extrinsics: "      << mono_parR.stdDevExtrinsics << endl;
    //cout << "Rotation vector: "     << mono_parR.rvecs            << endl;
    //cout << "Translation vector: "  << mono_parR.tvecs            << endl;
    cout << "RMS: "                 << mono_parR.RMS              << endl;


    // Калибровка стереопары
    //vector<String> images1,images2;
    cam_par_output_t o_par;

    calibrate_stereo(imagesL, imagesR, pathL, pathR, o_par);

    // Вывод параметров стереопары
    cout << "\n\n\t\t Both cameras\n";
    //cout << "cameraMatrix L: "                << o_par.cameraM1       << endl;
    //cout << "cameraMatrix R: "                << o_par.cameraM2       << endl;
    cout << "Distorsion coeffs L: "           << o_par.distCoeffs1    << endl;
    cout << "Distorsion coeffs R: "           << o_par.distCoeffs2    << endl;
    //cout << "Rotation matrix: "               << o_par.R              << endl;
    cout << "Translation matrix: "            << o_par.T              << endl;
    //cout << "Essential matrix: "              << o_par.E              << endl;
    //cout << "Fundamental matrix: "            << o_par.F              << endl;
    //cout << "Vector of rotation vectors: "    << o_par.rvecs          << endl;
    //cout << "Vector of translation vectors: " << o_par.tvecs          << endl;
    //cout << "Per view errors: "               << o_par.perViewErrors  << endl;
    cout << "RMS: "                           << o_par.RMS            << endl;


    // Загрузка калибровочных данных камер
    cv::Mat cameraMatrixL = o_par.cameraM1;
    cv::Mat cameraMatrixR = o_par.cameraM2;
    cv::Mat distCoeffsL = o_par.distCoeffs1;
    cv::Mat distCoeffsR = o_par.distCoeffs2;
    cv::Mat R = o_par.R;
    cv::Mat T = o_par.T;

    // Загрузка левого и правого изображений
    cv::Mat imageLeft = cv::imread("../../Fotoset/Stereo/tests/left/L88.png");
    cv::Mat imageRight = cv::imread("../../Fotoset/Stereo/tests/right/R88.png");
    //cv::Mat imageLeft = cv::imread("../../Fotoset/Stereo/tests/tele/left/R87.png");
    //cv::Mat imageRight = cv::imread("../../Fotoset/Stereo/tests/tele/right/L87.png");
    //cv::Mat imageLeft = cv::imread("../../Fotoset/Stereo/tests/any/view0.png");
    //cv::Mat imageRight = cv::imread("../../Fotoset/Stereo/tests/any/view1.png");

    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageLeft, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRight, grayImageRight, cv::COLOR_BGR2GRAY);

    cv::Mat Q, R1, R2, P1, P2;
    cv::stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,
                      grayImageLeft.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);


    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1,
                                imageLeft.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2,
                                imageRight.size(), CV_32FC1, mapRx, mapRy);

    cv::Mat rectifiedLeft, rectifiedRight;
    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR);
    /*
    CV_WRAP static Ptr<StereoSGBM> create(int minDisparity = 0, int numDisparities = 16, int blockSize = 3,
                                          int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
                                          int preFilterCap = 0, int uniquenessRatio = 0,
                                          int speckleWindowSize = 0, int speckleRange = 0,
                                          int mode = StereoSGBM::MODE_SGBM);
    */

    // Вычисление карты диспарантности
    cv::Mat disparityMap, disparity;
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, 16*5, 5, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            StereoSGBM::MODE_SGBM);
    stereo->compute(rectifiedLeft, rectifiedRight, disparityMap);
    disparityMap.convertTo(disparity,CV_32F, 1.0);
    disparity = (disparity/16.0f - 0.0)/((float)16*5);

    // Визуализация карты диспарантности
    cv::imshow("Disparity Map", disparity);
    cv::waitKey(0);

    // -----------------------------------------------------------------------------
/*
    // Триангуляция и сопоставление 2D и 3D точек
    cv::Mat image3D;
    cv::reprojectImageTo3D(disparityMap, image3D, Q, false, -1);

    // Ключевые точки на левом изображении (пример)
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageLeft, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRight, grayImageRight, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(grayImageLeft, keypoints, imageLeft.cols*imageLeft.rows, 0.01, 10);
    cout << "Worked"<<endl;

    int i = 0;
    // Сопоставление 2D и 3D точек
    for (const cv::Point2f& keypoint : keypoints) {
        int x = static_cast<int>(keypoint.x);
        int y = static_cast<int>(keypoint.y);

        // 3D координаты точки
        cv::Vec3f point3D = image3D.at<cv::Vec3f>(y, x);
        std::cout << i++ << ") 2D Point: (" << x << ", " << y << "), 3D Point: (" << point3D[0] << ", " << point3D[1] << ", " << point3D[2] << ")" << std::endl;
    }
*/

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}









