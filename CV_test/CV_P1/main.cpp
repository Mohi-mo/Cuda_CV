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


mono_output_par_t mono_parL;

cv::Mat rectifiedLeft, rectifiedRight;
stereo_output_par_t stereo_par;
cv::Vec3f point3D;
std::vector<cv::Point2f> point2D;

// Stereo create params
/*
int minDisparity = 0;  // 40
int numDisparities = 28; // or 2 - 15 for CUDA
int blockSize = 3; // 0
int preFilterType = 0; // P1
int preFilterSize = 8; // P2
int disp12MaxDiff = 25;
int preFilterCap = 40;
int uniquenessRatio = 1;
int speckleWindowSize = 0;
int speckleRange = 0;
int mode = StereoSGBM::MODE_SGBM; //  MODE_SGBM
*/

int minDisparity = 0;  // 40
int numDisparities = 16; // or 2 - 15 for CUDA
int blockSize = 3; // 0
int P1_ = 1;
int P2_ = 1;
int disp12MaxDiff = 25;
int preFilterCap = 40;
int uniquenessRatio = 1;
int speckleWindowSize = 0;
int speckleRange = 0;
int mode = StereoSGBM::MODE_SGBM; //  MODE_SGBM


//cv::Ptr<cv::cuda::StereoSGM> stereo = cv::cuda::createStereoSGM();

cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();


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

/*
// Закомментить при использовании алгоритмов CUDA
static void on_trackbar3( int, void* )
{
  stereo->setPreFilterType(preFilterType);
}

static void on_trackbar4( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;
}
*/
// ------------------------------------------------
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

// CUDA SGM features
static void on_trackbar3(int, void*){
    stereo->setP1(P1_);
}

static void on_trackbar4(int, void*){
    stereo->setP2(P2_);
}
/*
*/

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


void onMouseClick(int event, int x, int y, int flags, void* userdata){
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Mat image3D = *static_cast<cv::Mat*>(userdata);
        point3D = image3D.at<cv::Vec3f>(y, x);

        cv::circle(rectifiedLeft, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(rectifiedLeft, cv::Point(point3D[0], point3D[1]), 5, cv::Scalar(255, 0, 0), -1);

        std::cout << "Clicked 2D Point: (" << x << ", " << y << ")" << std::endl;
        std::cout << "3D Point: (" << point3D[0] << ", " << point3D[1] << ", " << point3D[2] << ")" << std::endl;
        cv::imshow("3D points on image", rectifiedLeft);
    }
}

void stereo_depth_map(cv::Mat rectified_image_left, cv::Mat rectified_image_right,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                      cv::Mat T, cv::Mat &disparity){

    double local_max, local_min;
    cv::Mat depthMap, coloredDepthMap, disparityMap;

    double fx_pix = cameraMatrixLeft.at<double>(0, 0);
    double fy_pix = cameraMatrixLeft.at<double>(1, 1);

    double fx_pixR = cameraMatrixRight.at<double>(0, 0);
    double fy_pixR = cameraMatrixRight.at<double>(1, 1);

    float focalLengthL = (fx_pix+fy_pix)/2;
    float focalLengthR = (fx_pixR+fy_pixR)/2;
    float focalLength = (focalLengthL + focalLengthR)/2;

    double baseline = std::abs(T.at<double>(0));

    cout << "Baseline: " << baseline <<endl;
    cout << "Focal length: " << focalLength << endl;

    while (true){
        stereo->compute(rectifiedLeft, rectifiedRight, disparityMap);

        disparityMap.convertTo(disparity,CV_32F,1.0f);
        disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

        cv::imshow("Disparity Map", disparity);
        cv::minMaxIdx(disparity, &local_min, &local_max);

        depthMap = focalLength * baseline / (disparity);
        //cv::normalize(depthMap, depthMap, local_min, local_max, cv::NORM_MINMAX);
        depthMap.convertTo(depthMap, CV_8U);

        cv::applyColorMap(depthMap, coloredDepthMap, cv::COLORMAP_JET);
        cv::imshow("Depth Map", coloredDepthMap);

        if (cv::waitKey(1) == 27) break;
    }
    cout << "Min disparity [" << local_min << "]; Max Disparity [" << local_max << "]" << endl;
}

void cuda_stereo_depth_map(){
    // Рассчёт карты диспарантности на CUDA
    // Подготовка данных на GPU
    //cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    //gpuImageLeft.upload(rectifiedLeft);
    //gpuImageRight.upload(rectifiedRight);

    /*
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
}

int main(int argc, char** argv) {
    //mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    std::vector<cv::String> imagesL, imagesR;
    string pathL, pathR;
    int num_set = 1;
    int checkerboard_c;
    int checkerboard_r;
    std::string name;

    if (num_set == 0){
        pathL = "../../Fotoset/T_rep/left";
        pathR = "../../Fotoset/T_rep/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "0";
    } else if (num_set == 1) {
        pathL = "../../Fotoset/lab_set/left";
        pathR = "../../Fotoset/lab_set/right";
        checkerboard_c = 7;
        checkerboard_r = 4;
        name = "1";
    } else if (num_set == 2) {
        //pathL = "../../../Fotoset/Left";
        //pathR = "../../../Fotoset/Right";
        name = "2";
    } else if (num_set == 3){
        pathL = "../../Fotoset/dataset_res/left";
        pathR = "../../Fotoset/dataset_res/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "3";
    }


    // Предзагрузка параметров калибровки камер
    cv::FileStorage fs;
    if (fs.open(name + "_left_camera_parameters.yml", cv::FileStorage::READ)){
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
        calibrate_camera(imagesL, pathL, name+ "_left", checkerboard_c,checkerboard_r, mono_parL);
    }

    if (fs.open(name + "_right_camera_parameters.yml", cv::FileStorage::READ)){
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
        calibrate_camera(imagesR, pathR, name + "_right", checkerboard_c,checkerboard_r, mono_parR);
    }

    // Show cameras parameters
    print_mono_camera_parameters("Left_camera", mono_parL);
    print_mono_camera_parameters("Right_camera", mono_parR);

    // Калибровка стереопары
    //stereo_output_par_t stereo_par;
    cv::FileStorage stereo_fs;
    if (stereo_fs.open(name + "_stereo_camera_parameters.yml", cv::FileStorage::READ)){
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
        calibrate_stereo(imagesL, imagesR, pathL, pathR, name, checkerboard_c,checkerboard_r, stereo_par);
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

    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    cv::imshow("Rectified left image", rectifiedLeft);
    cv::imshow("Rectified right image", rectifiedRight);

    // Creating a named window to be linked to the trackbars
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity",800,600);

    // Creating trackbars to dynamically update the StereoBM parameters
    cv::createTrackbar("numDisparities", "disparity", &numDisparities, 64, on_trackbar1);
    cv::createTrackbar("blockSize", "disparity", &blockSize, 50, on_trackbar2);
    //cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
    //cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
    cv::createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
    cv::createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar6);
    cv::createTrackbar("speckleRange", "disparity", &speckleRange, 100, on_trackbar7);
    cv::createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 25, on_trackbar8);
    cv::createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar9);
    cv::createTrackbar("minDisparity", "disparity", &minDisparity, 25, on_trackbar10);
    cv::createTrackbar("P1", "disparity", &P1_, 200, on_trackbar3);     // CUDA features
    cv::createTrackbar("P2", "disparity", &P2_, 200, on_trackbar4);     // CUDA features

    cv::Mat image3D, disparity;
    stereo_depth_map(rectifiedLeft, rectifiedRight, stereo_par.cameraM1, stereo_par.cameraM2, stereo_par.T, disparity);

    cv::reprojectImageTo3D(disparity, image3D, Q, false, -1);

    cv::cvtColor(rectifiedLeft, rectifiedLeft, cv::COLOR_GRAY2BGR);

    cv::imshow("3D points on image", rectifiedLeft);
    cv::setMouseCallback("3D points on image", onMouseClick, &image3D);
    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}








