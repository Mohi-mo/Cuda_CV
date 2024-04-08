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
#include "triangulate.h"


using namespace std;
using namespace cv;
using namespace cuda;


/// Объявление переменных (временно)
/// \todo Убрать глобальные переменные, сделать их передачу как параметров
cv::Mat rectifiedLeft, rectifiedRight;
//cv::Mat mapLx, mapLy, mapRx, mapRy, Q;
//cv::Mat P1, P2;
cv::Mat point3D;
cv::Mat image3D;
cv::Mat disparity;

/// Объявление структур для хранения параметров калибровки
stereo_output_par_t stereo_par;
stereo_match_t SGBM_par;


/// Создание объектов для алгоритмов рассчёта карты диспарантности
cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
//cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();

/// Создание объектов для алгоритмов рассчёта карты диспарантности с использованием CUDA
//cv::Ptr<cv::cuda::StereoSGM> stereo = cv::cuda::createStereoSGM();
//cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo = cv::cuda::createStereoBeliefPropagation();


/// Определение функций, задающих параметры алгоритмов рассчёта карты диспарантности
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

void on_trackbar3( int, void* )
{
  stereo->setPreFilterCap(SGBM_par.preFilterCap);
}

void on_trackbar4( int, void* )
{
  stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
}

void on_trackbar5( int, void* )
{
  stereo->setSpeckleRange(SGBM_par.speckleRange);
}

void on_trackbar6( int, void* )
{
  stereo->setSpeckleWindowSize(SGBM_par.speckleWindowSize*2);
  SGBM_par.speckleWindowSize = SGBM_par.speckleWindowSize*2;
}

void on_trackbar7( int, void* )
{
  stereo->setDisp12MaxDiff(SGBM_par.disp12MaxDiff);
}

void on_trackbar8( int, void* )
{
  stereo->setMinDisparity(SGBM_par.minDisparity);
}

void on_trackbar9(int, void*){
    stereo->setP1(SGBM_par.P1_);
}

void on_trackbar10(int, void*){
    stereo->setP2(SGBM_par.P2_);
}

/*
static void on_trackbar9( int, void* )
{
  stereo->setPreFilterType(preFilterType);
}

static void on_trackbar10( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;
}
*/


/// Функция для считывания кадров в режиме реального времени (нуждается в доработке)
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


/// Функция обратного вызова по клику мышкой
void onMouseClick(int event, int x, int y, int flags, void* userdata){

    // Переменные для хранения координат 2Д точек левого и правого изображений
    std::vector<cv::Point2f> point2DL, point2DR;

    if (event == cv::EVENT_LBUTTONDOWN) {
        point2DL.push_back(cv::Point2f(x,y));

        //cv::reprojectImageTo3D(disparity, image3D, Q, false, -1);
        //point2DR.push_back(cv::Point2f(mapRx.at<float>(y,x), mapRy.at<float>(y,x)));

        //std::cout << " and right: " << point2DR << std::endl;

        //std::cout << "3D Image: " << image3D << std::endl;
        //point2DR.push_back(cv::Point2f(image3D.at<float>(y,x), mapRy.at<float>(y,x)));
        //cv::triangulatePoints(P1, P2, point2DL, point2DR);
        /*
        for(int i = 0; i< image3D.cols; i++){
            cv::Point3f point3D(points3D.at<float>(0, i) / points3D.at<float>(3, i),
                                points3D.at<float>(1, i) / points3D.at<float>(3, i),
                                points3D.at<float>(2, i) / points3D.at<float>(3, i));
            std::cout << "3D Point: " << point3D << std::endl;

            points3D.at<float>(0, i) = point3D.x;
            points3D.at<float>(1, i) = point3D.y;
            points3D.at<float>(2, i) = point3D.z;
        }
        */

        // Вывод в консоль значений координат 2д точки левого изображенияя
        std::cout << "2D Points left: " << point2DL<< std::endl;

        // Проекция 3D точки на 2D изображение левой камеры
        std::vector<cv::Point3f> points3D(1, cv::Point3f(image3D.at<cv::Vec3f>(y, x)));
        std::vector<cv::Point2f> projectedPoints;

        cv::projectPoints(points3D, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F),
                          stereo_par.cameraM1, stereo_par.distCoeffs1, projectedPoints);

        std::cout << "3D Point: " << points3D << std::endl;

        // Отображение 3D точки на левом и правом изображении
        cv::circle(rectifiedLeft, projectedPoints[0], 5, cv::Scalar(255, 0, 0), -1);
        cv::circle(rectifiedLeft, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        //cv::circle(rectifiedRight, cv::Point(newpoint2DR[0].x, newpoint2DR[0].y), 5, cv::Scalar(0, 255, 0), -1);
        //cv::circle(rectifiedLeft, cv::Point(points3D.at<float>(0,0), points3D.at<float>(1,0)), 5, cv::Scalar(255, 0, 0), -1);

        cv::imshow("3D points on image", rectifiedLeft);
        cv::imshow("Points on right image: ", rectifiedRight);
    }
}


int main(int argc, char** argv) {
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    std::vector<cv::String> imagesL, imagesR;   // Переменные, содержащие пути к датасетам
    std::string pathL, pathR;                   // Переменные, содержащие пути к датасетам
    unsigned int num_set = 1;                   // Номер текущего используемого калибровочного датасета
    unsigned int num_set_stereo = 5;            // Номер текущего используемого тестового датасета
    int checkerboard_c;                         // Число ключевых точек по столбцам
    int checkerboard_r;                         // Число ключевых точек по строкам
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
    default:
        pathL = "../../Fotoset/basler_festo/left";
        pathR = "../../Fotoset/basler_festo/right";
        checkerboard_c = 9;
        checkerboard_r = 6;
        name = "4";
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
        calibrate_with_mono(imagesL,imagesR, pathL, pathR, name, checkerboard_c, checkerboard_r, mono_parL, mono_parR, stereo_par);
    }

    // Вывод параметров стереопары
    print_stereo_camera_parameters(stereo_par);

    // Загрузка тестовых левого и правого изображений
    cv::Mat imageLeft, imageRight;
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
    default:
        break;
    }

    // Перевод изображений из цветного формата в монохромный
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageLeft, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRight, grayImageRight, cv::COLOR_BGR2GRAY);

    // Стереоректификация изображений
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Mat Q, R1, R2, P1, P2;
    //cv::Mat R1, R2;
    cv::stereoRectify(stereo_par.cameraM1, stereo_par.distCoeffs1, stereo_par.cameraM2, stereo_par.distCoeffs2,
                      cv::Size(grayImageLeft.cols, grayImageLeft.rows), stereo_par.R, stereo_par.T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY);


    double baseline = Q.at<double>(3,2);
    double focalLength = Q.at<double>(2,3);

    std::cout << "Baseline: " << baseline << std::endl;
    std::cout << "Focal length: " << focalLength << std::endl;

    // Нахождение параметров ректификации и устранения искажений для левого изображения
    //cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(stereo_par.cameraM1, stereo_par.distCoeffs1, R1, P1,
                                cv::Size(imageLeft.cols, imageLeft.rows), CV_32FC1, mapLx, mapLy);

    // Нахождение параметров ректификации и устранения искажений для правого изображения
    cv::initUndistortRectifyMap(stereo_par.cameraM2, stereo_par.distCoeffs2, R2, P2,
                                cv::Size(imageLeft.cols, imageLeft.rows), CV_32FC1, mapRx, mapRy);

    // Ректификация и устранение ускажений с использованием полученных параметров
    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // Вывод ректифицированных изображений
    cv::imshow("Rectified left image", rectifiedLeft);
    cv::imshow("Rectified right image", rectifiedRight);

    // Создание окна с ползунками для методов вычисления карты диспарантности
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity",800,600);

    // Создание функций обратной связи для обновления параметров
    cv::createTrackbar("numDisparities", "disparity", &SGBM_par.numDisparities, 64, on_trackbar1);
    cv::createTrackbar("blockSize", "disparity", &SGBM_par.blockSize, 50, on_trackbar2);
    cv::createTrackbar("preFilterCap", "disparity", &SGBM_par.preFilterCap, 62, on_trackbar3);
    cv::createTrackbar("uniquenessRatio", "disparity", &SGBM_par.uniquenessRatio, 100, on_trackbar4);
    cv::createTrackbar("speckleRange", "disparity", &SGBM_par.speckleRange, 100, on_trackbar5);
    cv::createTrackbar("speckleWindowSize", "disparity", &SGBM_par.speckleWindowSize, 25, on_trackbar6);
    cv::createTrackbar("disp12MaxDiff", "disparity", &SGBM_par.disp12MaxDiff, 25, on_trackbar7);
    cv::createTrackbar("minDisparity", "disparity", &SGBM_par.minDisparity, 25, on_trackbar8);
    cv::createTrackbar("P1", "disparity", &SGBM_par.P1_, 200, on_trackbar9);
    cv::createTrackbar("P2", "disparity", &SGBM_par.P2_, 200, on_trackbar10);

    //cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
    //cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);



    // Вычисление карты диспарантности
    stereo_depth_map(rectifiedLeft, rectifiedRight, P1, P2, stereo_par.T,
                     disparity, SGBM_par.numDisparities, SGBM_par.minDisparity, stereo);

    // Вычисление карты диспарантности с использованием CUDA
    //cuda_stereo_depth_map(rectifiedLeft, rectifiedRight, P1, P2, stereo_par.T,
    //                     disparity, SGBM_par.numDisparities, SGBM_par.minDisparity, stereo);

    // Нахождение 3д точек
    reprojectImageTo3D(disparity, point3D, Q, false, -1);

    // Поиск ключевых точек на левом и правом изображениях
    Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> keypointsLeft, keypointsRight;
    detector->detect(rectifiedLeft, keypointsLeft);
    detector->detect(rectifiedRight, keypointsRight);

    // Сопоставление ключевых точек
    Ptr<DescriptorExtractor> extractor = ORB::create();
    Mat descriptorsLeft, descriptorsRight;
    extractor->compute(rectifiedLeft, keypointsLeft, descriptorsLeft);
    extractor->compute(rectifiedRight, keypointsRight, descriptorsRight);

    // Использование BFMatcher для сопоставления дескрипторов
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Отображение сопоставленных точек
    Mat matchedImage;
    drawMatches(rectifiedLeft, keypointsLeft, rectifiedRight, keypointsRight, matches, matchedImage);

    // Вывод результатов
    cv::imshow("Matched Points", matchedImage);

    // Смена цветовой схемы изображений для отрисовки цветных точек
    cv::cvtColor(rectifiedLeft, rectifiedLeft, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectifiedRight, rectifiedRight, cv::COLOR_GRAY2BGR);

    // Вывод левого изображения и установка функции обратной связи для обработки кликов мыши
    cv::imshow("3D points on image", rectifiedLeft);
    cv::setMouseCallback("3D points on image", onMouseClick);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}








