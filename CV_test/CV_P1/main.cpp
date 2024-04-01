#include <iostream>
#include <opencv2/opencv.hpp>
//#include "opencv2/core.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>

#include "calibrate.h"
#include "triangulate.h"


using namespace std;
using namespace cv;
using namespace cuda;


cv::Mat rectifiedLeft, rectifiedRight;
cv::Mat mapLx, mapLy, mapRx, mapRy;
cv::Mat P1, P2;
cv::Mat points3D;

stereo_output_par_t stereo_par;
stereo_match_t SGBM_par;

//cv::Ptr<cv::cuda::StereoSGM> stereo = cv::cuda::createStereoSGM();
cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();

void on_trackbar1( int, void* )
{
  stereo->setNumDisparities(SGBM_par.numDisparities*16);
  SGBM_par.numDisparities = SGBM_par.numDisparities*16;
}

void on_trackbar2( int, void* )
{
  stereo->setBlockSize(SGBM_par.blockSize*2+5);
  SGBM_par.blockSize = SGBM_par.blockSize*2+5;
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
void on_trackbar5( int, void* )
{
  stereo->setPreFilterCap(SGBM_par.preFilterCap);
}

void on_trackbar6( int, void* )
{
  stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
}

void on_trackbar7( int, void* )
{
  stereo->setSpeckleRange(SGBM_par.speckleRange);
}

void on_trackbar8( int, void* )
{
  stereo->setSpeckleWindowSize(SGBM_par.speckleWindowSize*2);
  SGBM_par.speckleWindowSize = SGBM_par.speckleWindowSize*2;
}

void on_trackbar9( int, void* )
{
  stereo->setDisp12MaxDiff(SGBM_par.disp12MaxDiff);
}

void on_trackbar10( int, void* )
{
  stereo->setMinDisparity(SGBM_par.minDisparity);
}

// CUDA SGM features
void on_trackbar3(int, void*){
    stereo->setP1(SGBM_par.P1_);
}

void on_trackbar4(int, void*){
    stereo->setP2(SGBM_par.P2_);
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

/*
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
*/

void onMouseClick(int event, int x, int y, int flags, void* userdata){

    std::vector<cv::Point2f> point2DL, point2DR;

    if (event == cv::EVENT_LBUTTONDOWN) {
        point2DL.push_back(cv::Point2f(x,y));
        point2DR.push_back(cv::Point2f(mapRx.at<float>(y,x), mapRy.at<float>(y,x)));

        std::cout << "2D Points left: " << point2DL << " and right: " << point2DR << std::endl;
        cv::triangulatePoints(P1, P2, point2DL, point2DR, points3D);

        for(int i = 0; i< points3D.cols; i++){
            cv::Point3f point3D(points3D.at<float>(0, i) / points3D.at<float>(3, i),
                                points3D.at<float>(1, i) / points3D.at<float>(3, i),
                                points3D.at<float>(2, i) / points3D.at<float>(3, i));
            std::cout << "3D Point: " << point3D << std::endl;

            points3D.at<float>(0, i) = point3D.x;
            points3D.at<float>(1, i) = point3D.y;
            points3D.at<float>(2, i) = point3D.z;
        }

        cv::circle(rectifiedLeft, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        //cv::circle(rectifiedRight, cv::Point(point2DR[0], point2DR[1]), 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(rectifiedLeft, cv::Point(points3D.at<float>(0,0), points3D.at<float>(1,0)), 5, cv::Scalar(255, 0, 0), -1);

        cv::imshow("3D points on image", rectifiedLeft);
    }
}

int main(int argc, char** argv) {
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    std::vector<cv::String> imagesL, imagesR;
    string pathL, pathR;
    int num_set = 1;
    int checkerboard_c;
    int checkerboard_r;
    std::string name;
    bool calibrate = false;


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
    if (stereo_fs.open(name + "_stereo_camera_parameters.yml", cv::FileStorage::READ) && (!calibrate)){
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

    //cv::Mat Q, R1, R2, P1, P2;
    cv::Mat Q, R1, R2;
    cv::stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,
                      grayImageLeft.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);


    //cv::Mat mapLx, mapLy, mapRx, mapRy;
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
    cv::createTrackbar("numDisparities", "disparity", &SGBM_par.numDisparities, 64, on_trackbar1);
    cv::createTrackbar("blockSize", "disparity", &SGBM_par.blockSize, 50, on_trackbar2);
    //cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
    //cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
    cv::createTrackbar("preFilterCap", "disparity", &SGBM_par.preFilterCap, 62, on_trackbar5);
    cv::createTrackbar("uniquenessRatio", "disparity", &SGBM_par.uniquenessRatio, 100, on_trackbar6);
    cv::createTrackbar("speckleRange", "disparity", &SGBM_par.speckleRange, 100, on_trackbar7);
    cv::createTrackbar("speckleWindowSize", "disparity", &SGBM_par.speckleWindowSize, 25, on_trackbar8);
    cv::createTrackbar("disp12MaxDiff", "disparity", &SGBM_par.disp12MaxDiff, 25, on_trackbar9);
    cv::createTrackbar("minDisparity", "disparity", &SGBM_par.minDisparity, 25, on_trackbar10);
    cv::createTrackbar("P1", "disparity", &SGBM_par.P1_, 200, on_trackbar3);     // CUDA features
    cv::createTrackbar("P2", "disparity", &SGBM_par.P2_, 200, on_trackbar4);     // CUDA features

    cv::Mat image3D, disparity;
    stereo_depth_map(rectifiedLeft, rectifiedRight, stereo_par.cameraM1, stereo_par.cameraM2, stereo_par.T,
                     disparity, SGBM_par.numDisparities, SGBM_par.minDisparity, stereo);

    //cv::reprojectImageTo3D(disparity, image3D, Q, false, -1);

    cv::cvtColor(rectifiedLeft, rectifiedLeft, cv::COLOR_GRAY2BGR);

    cv::imshow("3D points on image", rectifiedLeft);
    cv::setMouseCallback("3D points on image", onMouseClick, &image3D);
    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}








