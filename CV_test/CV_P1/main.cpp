#include <iostream>
#include <opencv2/opencv.hpp>
//#include "opencv2/core.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>

//#include <QApplication>
//#include <QMouseEvent>
//#include <QLabel>

#include "calibrate.h"

using namespace std;
using namespace cv;
using namespace cuda;


// Stereo create params
int minDisparity = 0;
int numDisparities = 1;
int blockSize = 3;
int preFilterType = 1; // P1
int preFilterSize = 1; // P2
int disp12MaxDiff = 18;
int preFilterCap = 0;
int uniquenessRatio = 1;
int speckleWindowSize = 0;
int speckleRange = 0;
int mode = StereoSGBM::MODE_SGBM;

cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();


static void on_trackbar1( int, void* )
{
  stereo->setNumDisparities(numDisparities*16);
  numDisparities = numDisparities*16;
}

static void on_trackbar2( int, void* )
{
  stereo->setBlockSize(blockSize*2+5);
  blockSize = blockSize*2+5;
}

static void on_trackbar3( int, void* )
{
  stereo->setPreFilterType(preFilterType);
}

static void on_trackbar4( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;
}

static void on_trackbar5( int, void* )
{
  stereo->setPreFilterCap(preFilterCap);
}

static void on_trackbar6( int, void* )
{
  stereo->setUniquenessRatio(uniquenessRatio);
}

static void on_trackbar7( int, void* )
{
  stereo->setSpeckleRange(speckleRange);
}

static void on_trackbar8( int, void* )
{
  stereo->setSpeckleWindowSize(speckleWindowSize*2);
  speckleWindowSize = speckleWindowSize*2;
}

static void on_trackbar9( int, void* )
{
  stereo->setDisp12MaxDiff(disp12MaxDiff);
}

static void on_trackbar10( int, void* )
{
  stereo->setMinDisparity(minDisparity);
}


// Проверка работоспособности CUDA
void CUDA_work_check(){
    printShortCudaDeviceInfo(getDevice());
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
}

// Захват видео
int cameraRecv(cv::Mat InputLeftIm, cv::Mat InputRigthIm){
    cv::VideoCapture leftCamera(0);
    cv::VideoCapture rightCamera(1);

    if (!leftCamera.isOpened() || !rightCamera.isOpened()) {
        std::cerr << "Error: Could not open cameras." << endl;
        return -1;
    }

    /*
     * leftCamera >> imagesL;
     * rigthCamera >> imagesL;
     *
     * cv::cvtColor(imagesL, imgL_gray, cv::COLOR_BGR2GRAY);
     * cv::cvtColor(imagesR, imgR_gray, cv::COLOR_BGR2GRAY);
     *
     *
    */

    leftCamera.release();
    rightCamera.release();
    return 0;
}

// Отображение параметров камеры
void print_mono_camera_parameters(std::string name, mono_output_par_t mono_struct){
    cout << "\n\n\t" << name << "---------------------------------" << endl;
    cout << "cameraMatrix: "        << mono_struct.cameraMatrix     << endl;
    cout << "distCoeffs: "         << mono_struct.distCoeffs        << endl;
    //cout << "Per view errors: "     << mono_struct.perViewErrors    << endl;
    //cout << "STD Intrinsics: "      << mono_struct.stdDevIntrinsics << endl;
    //cout << "STD Extrinsics: "      << mono_struct.stdDevExtrinsics << endl;
    //cout << "Rotation vector: "     << mono_struct.rvecs            << endl;
    //cout << "Translation vector: "  << mono_struct.tvecs            << endl;
    cout << "RMS: "                 << mono_struct.RMS              << endl;
}

// Отображение параметров стерео камеры
void print_stereo_camera_parameters(stereo_output_par_t stereo_struct){
    cout << "\n\n\t\t Both cameras" << "------------------------------------" << endl;
    cout << "cameraMatrix L: "                << stereo_struct.cameraM1       << endl;
    cout << "cameraMatrix R: "                << stereo_struct.cameraM2       << endl;
    cout << "Distorsion coeffs L: "           << stereo_struct.distCoeffs1    << endl;
    cout << "Distorsion coeffs R: "           << stereo_struct.distCoeffs2    << endl;
    cout << "Rotation matrix: "               << stereo_struct.R              << endl;
    cout << "Translation matrix: "            << stereo_struct.T              << endl;
    cout << "Essential matrix: "              << stereo_struct.E              << endl;
    cout << "Fundamental matrix: "            << stereo_struct.F              << endl;
    //cout << "Vector of rotation vectors: "    << stereo_struct.rvecs          << endl;
    //cout << "Vector of translation vectors: " << stereo_struct.tvecs          << endl;
    //cout << "Per view errors: "               << stereo_struct.perViewErrors  << endl;
    cout << "RMS: "                           << stereo_struct.RMS            << endl;
}


int main(int argc, char** argv) {
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    std::vector<cv::String> imagesL, imagesR;
    string pathL, pathR;
    int num_set = 1;
    int checkerboard_c;
    int checkerboard_r;

    if (num_set == 0){
        // 4x7 - lab_set
        // 6x9 - T-rep
        pathL = "../../Fotoset/T_rep/left";
        pathR = "../../Fotoset/T_rep/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
    } else if (num_set == 1) {
        pathL = "../../Fotoset/lab_set/left";
        pathR = "../../Fotoset/lab_set/right";
        checkerboard_c = 7;
        checkerboard_r = 4;
    } else if (num_set == 2) {
        //pathL = "../../../Fotoset/Left";
        //pathR = "../../../Fotoset/Right";
    }

    // Предзагрузка параметров калибровки камер
    cv::FileStorage fs;
    if (fs.open("left_camera_parameters.yml", cv::FileStorage::READ)){
        if (fs.isOpened()){
            fs["cameraMatrix"] >> mono_parL.cameraMatrix;
            fs["distCoeffs"] >> mono_parL.distCoeffs;
            fs["PerViewErrors"] >> mono_parL.perViewErrors;
            fs["STDIntrinsics"] >> mono_parL.stdDevIntrinsics;
            fs["STDExtrinsics"] >> mono_parL.stdDevExtrinsics;
            fs["RotationVector"] >> mono_parL.rvecs;
            fs["TranslationVector"] >> mono_parL.tvecs;
            fs["RMS"] >> mono_parL.RMS;
            fs.release();
          }
      } else {
        cout << "Left calibration procedure is running..." << endl;
        calibrate_camera(imagesL, pathL, "left", checkerboard_c,checkerboard_r, mono_parL);
    }

    if (fs.open("right_camera_parameters.yml", cv::FileStorage::READ)){
        if (fs.isOpened()){
          fs["cameraMatrix"] >> mono_parR.cameraMatrix;
          fs["distCoeffs"] >> mono_parR.distCoeffs;
          fs["PerViewErrors"] >> mono_parR.perViewErrors;
          fs["STDIntrinsics"] >> mono_parR.stdDevIntrinsics;
          fs["STDExtrinsics"] >> mono_parR.stdDevExtrinsics;
          fs["RotationVector"] >> mono_parR.rvecs;
          fs["TranslationVector"] >> mono_parR.tvecs;
          fs["RMS"] >> mono_parR.RMS;
          fs.release();
        }
    } else {
        cout << "Right calibration procedure is running..." << endl;
        calibrate_camera(imagesR, pathR, "right", checkerboard_c,checkerboard_r, mono_parR);
    }

    // Show cameras parameters
    print_mono_camera_parameters("Left_camera", mono_parL);
    print_mono_camera_parameters("Right_camera", mono_parR);

    // Калибровка стереопары
    stereo_output_par_t stereo_par;
    cv::FileStorage stereo_fs;
    if (stereo_fs.open("stereo_camera_parameters.yml", cv::FileStorage::READ)){
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
        calibrate_stereo(imagesL, imagesR, pathL, pathR, checkerboard_c,checkerboard_r, stereo_par);
    }

    // Вывод параметров стереопары
    print_stereo_camera_parameters(stereo_par);

    // Загрузка калибровочных данных камер
    cv::Mat cameraMatrixL = stereo_par.cameraM1;
    cv::Mat cameraMatrixR = stereo_par.cameraM2;
    cv::Mat distCoeffsL = stereo_par.distCoeffs1;
    cv::Mat distCoeffsR = stereo_par.distCoeffs2;
    cv::Mat R = stereo_par.R;
    cv::Mat T = stereo_par.T;

    // Загрузка левого и правого изображений
    cv::Mat imageLeft = cv::imread("../../Fotoset/Stereo/tests/left/L88.png");
    cv::Mat imageRight = cv::imread("../../Fotoset/Stereo/tests/right/R88.png");
    //cv::Mat imageLeft = cv::imread("../../Fotoset/Stereo/tests/tele/left/R87.png");
    //cv::Mat imageRight = cv::imread("../../Fotoset/Stereo/tests/tele/right/L87.png");
    //cv::Mat imageLeft = cv::imread("../../Fotoset/Stereo/tests/any/view0.png");
    //cv::Mat imageRight = cv::imread("../../Fotoset/Stereo/tests/any/view1.png");

    cv::imshow("Left image", imageLeft);
    cv::imshow("Right image", imageRight);
    cv::waitKey(0);

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
                                grayImageRight.size(), CV_32FC1, mapRx, mapRy);

    cv::imshow("Gray left image", grayImageLeft);
    cv::imshow("Gray right image", grayImageRight);
    cv::waitKey(0);

    cv::Mat rectifiedLeft, rectifiedRight;
    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

    cv::imshow("Rectified left image", rectifiedLeft);
    cv::imshow("Rectified right image", rectifiedRight);
    cv::waitKey(0);

    /*
    // Рассчёт карты диспарантности на CUDA
    // Подготовка данных на GPU
    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedLeft);
    gpuImageRight.upload(rectifiedRight);

    // Создание объекта для вычисления диспарантности на GPU
    cv::Ptr<cv::cuda::StereoBM> stereo = cv::cuda::createStereoBM(numDisparities, blockSize);

    // Вычисление карты диспарантности на GPU
    cv::cuda::GpuMat gpuDisparityMap;
    stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

    // Скачивание результата с GPU
    cv::Mat disparityMap;
    gpuDisparityMap.download(disparityMap);

    cv::Mat disparity;
    disparityMap.convertTo(disparity,CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

    // Визуализация карты диспарантности
    cv::imshow("Disparity Map", disparity);  // Деление на 16 для приведения к масштабу
    */


    // Вычисление карты диспарантности
    /*
     *
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
                                                            preFilterType, preFilterSize, disp12MaxDiff,
                                                            preFilterCap, uniquenessRatio,
                                                            speckleWindowSize, speckleRange,
                                                            mode);

    stereo->compute(rectifiedLeft, rectifiedRight, disparityMap);
    disparityMap.convertTo(disparity,CV_32F, 1.0);
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

    cv::imshow("Rectified image", rectifiedLeft);
    cv::waitKey(0);

    // Визуализация карты диспаSрантности
    cv::imshow("Disparity Map", disparity);
    cv::waitKey(0);
    */

    // -----------------------------------------------------------------------------

    cv::Mat disparity, disparityMap;
    // Creating a named window to be linked to the trackbars
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity",600,600);

    // Creating trackbars to dynamically update the StereoBM parameters
    cv::createTrackbar("numDisparities", "disparity", &numDisparities, 18, on_trackbar1);
    cv::createTrackbar("blockSize", "disparity", &blockSize, 50, on_trackbar2);
    cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
    cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
    cv::createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
    cv::createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar6);
    cv::createTrackbar("speckleRange", "disparity", &speckleRange, 100, on_trackbar7);
    cv::createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 25, on_trackbar8);
    cv::createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar9);
    cv::createTrackbar("minDisparity", "disparity", &minDisparity, 25, on_trackbar10);

    while (true){
        stereo->compute(rectifiedLeft, rectifiedRight, disparityMap);
        disparityMap.convertTo(disparity,CV_32F, 1.0);
        disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

        cv::imshow("Disparity Map", disparity);
        if (cv::waitKey(1) == 27) break;
    }

    // Триангуляция и сопоставление 2D и 3D точек
    cv::Mat image3D;
    cv::reprojectImageTo3D(disparityMap, image3D, Q, true, -1);

    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(rectifiedLeft, keypoints, imageLeft.cols*imageLeft.rows, 0.01, 10);

    std::vector<cv::Point2f> point2D;
    cv::Vec3f point3D;

    int i = 0;
    // Сопоставление 2D и 3D точек
    for (const cv::Point2f& keypoint : keypoints) {
        int x = static_cast<int>(keypoint.x);
        int y = static_cast<int>(keypoint.y);

        // 3D координаты точки
        point3D = image3D.at<cv::Vec3f>(y, x);
        //std::cout << i++ << ") 2D Point: (" << x << ", " << y << "), 3D Point: (" << point3D[0] << ", " << point3D[1] << ", " << point3D[2] << ")" << std::endl;
        cv::projectPoints(point3D, stereo_par.R, stereo_par.T, stereo_par.cameraM1, stereo_par.distCoeffs1, point2D);

        for (const auto& point : point2D){
            cv::circle(rectifiedLeft, point, 5, Scalar(0,255,255), -1, LINE_8);
        }
    }

    cv::imshow("3D points on image", rectifiedLeft);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}








