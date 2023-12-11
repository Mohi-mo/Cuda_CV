#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <QApplication>
#include <QMouseEvent>
#include <QLabel>

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

// Преобразование изображений (смещение, повоторот, устранение искажений)
// rectifyImage(cam_par_output_t& output_parameters, OutputArray image, cv::Mat image2)
void rectifyImage(cv::Mat image1, cv::Mat image2, cv::Mat& combinedImage, cam_par_output_t& output_parameters){
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validROI[2];

    Size imageSize = image1.size();

    cv::stereoRectify(output_parameters.cameraM1, output_parameters.distCoeffs1, output_parameters.cameraM2, output_parameters.distCoeffs2,
                      imageSize, output_parameters.R, output_parameters.T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validROI[0], &validROI[1]);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(output_parameters.cameraM1, output_parameters.distCoeffs1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(output_parameters.cameraM2, output_parameters.distCoeffs2, R2, P2, imageSize, CV_32FC1, map2x, map2y);

    cv::Mat rectified1, rectified2;
    cv::remap(image1, rectified1, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(image2, rectified2, map2x, map2y, cv::INTER_LINEAR);

    cv::Rect roi1 = validROI[0];
    cv::Rect roi2 = validROI[1];

    cv::Mat cropped1 = rectified1(roi1);
    cv::Mat cropped2 = rectified2(roi2);

    //cv::Mat combinedImage;
    cv::hconcat(cropped1, cropped2, combinedImage);  // Или vconcat
    cv::imshow("Combined", combinedImage);
}


int main(int argc, char** argv) {
    /* Калибровка через захват видео
    cv::VideoCapture leftCamera(0);
    cv::VideoCapture rightCamera(1);

    if (!leftCamera.isOpened() || !rightCamera.isOpened()) {
        std::cerr << "Error: Could not open cameras." << std::endl;
        return -1;
    }

    // Код вставить сюда

    leftCamera.release();
    rightCamera.release();
    */

    // Калибровка камер через заготоволенный датасет
    Mat cameraMatrixL,distCoeffsL,RL,TL;
    Mat cameraMatrixR,distCoeffsR,RR,TR;

    vector<String> imagesL;
    //string pathL = "../../../Fotoset/Stereo/left";
    string pathL = "../../../Fotoset/D2/";

    vector<String> imagesR;
    string pathR = "../../../Fotoset/J2/";

    calibrate_camera(imagesL, pathL, cameraMatrixL, distCoeffsL, RL, TL);
    calibrate_camera(imagesR, pathR, cameraMatrixR, distCoeffsR, RR, TR);

    // Вывод параметров камер
    cout << "\t Left camera\n";
    cout << "cameraMatrix : " << cameraMatrixL  << endl;
    cout << "distCoeffs : " << distCoeffsL      << endl;
    cout << "Rotation vector : " << RL          << endl;
    cout << "Translation vector : " << TL       << endl;

    cout << "\t Right camera\n";
    cout << "cameraMatrix : " << cameraMatrixR  << endl;
    cout << "distCoeffs : " << distCoeffsR      << endl;
    cout << "Rotation vector : " << RR          << endl;
    cout << "Translation vector : " << TR       << endl;

    // Калибровка стереопары
    vector<String> images1,images2;
    //string path1 = "../../../Fotoset/Stereo/test/left";
    //string path2 = "../../../Fotoset/Stereo/test/right";

    cam_par_output_t o_par;
    calibrate_stereo(images1, images1, pathL, pathR, o_par);

    // Вывод параметров стереопары
    cout << "\n\n\t\t Both cameras\n";
    cout << "RMS Error: \t"              << o_par.RMS    << "\n" << endl;
    cout << "Rotation Matrix (R): \t"    << o_par.R      << "\n" << endl;
    cout << "Translation Vector (T): \t" << o_par.T      << "\n" << endl;


    cv::Mat imResult;
    //cv::Mat im1 = cv::imread("../../../Fotoset/Stereo/tests/left/R88.bmp");
    //cv::Mat im2 = cv::imread( "../../../Fotoset/Stereo/tests/right/L88.bmp");
    cv::Mat im1 = cv::imread("../../../Fotoset/D2/camera0_0.png");
    cv::Mat im2 = cv::imread("../../../Fotoset/J2/camera1_0.png");

    rectifyImage(im1, im2, imResult, o_par);
    //triangulation();
    return 0;
}









