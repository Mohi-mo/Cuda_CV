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
cv::Mat disparity;

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

stereo_match_t SGBM_par;

float min_y = 10000.0;
float max_y = -10000.0;
float min_x =  10000.0;
float max_x = -10000.0;

/// Создание объектов для алгоритмов рассчёта карты диспарантности
cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
//cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();

/// Создание объектов для алгоритмов рассчёта карты диспарантности с использованием CUDA
//cv::Ptr<cv::cuda::StereoSGM> stereo = cv::cuda::createStereoSGM();
//cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo = cv::cuda::createStereoBeliefPropagation();


/// Определение функции, задающей параметры алгоритмов рассчёта карты диспарантности
void on_trackbar(int, void*){
    stereo->setNumDisparities(SGBM_par.numDisparities*16);
    stereo->setBlockSize(SGBM_par.blockSize*2+5);
    stereo->setMinDisparity(SGBM_par.minDisparity);
    stereo->setPreFilterCap(SGBM_par.preFilterCap);
    stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
    stereo->setSpeckleWindowSize(SGBM_par.speckleWindowSize);
    stereo->setSpeckleRange(SGBM_par.speckleRange);
    stereo->setDisp12MaxDiff(SGBM_par.disp12MaxDiff);


    /// Вычисление карты диспарантности
    stereo_depth_map(rectifiedLeft, rectifiedRight, disparity, stereo);

    //stereo.estimateRecommendedParams(rectifiedLeft.cols, rectifiedLeft.rows, SGBM_par.numDisparities);
    //int ndisp, iters, levels;
    //cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(rectifiedLeft.cols, rectifiedLeft.rows, ndisp, iters, levels);

    //std::cout << "Num disparities: "    << ndisp << "\n"
    //            <<"Iters: "             << iters << "\n"
    //            <<"Levels: "            << levels << std::endl;

    /// Вычисление карты диспарантности с использованием CUDA
    //cuda_stereo_depth_map(rectifiedLeft, rectifiedRight, P1, P2, stereo_par.T,
    //                     disparity, SGBM_par.numDisparities, SGBM_par.minDisparity, stereo);

}

/// Функция обратного вызова по клику мышкой
void onMouseClick(int event, int x, int y, int flags, void* userdata) {
    MouseCallbackData* data = static_cast<MouseCallbackData*>(userdata);
    vector<vector<int>>& cordsL = data->leftKey;
    vector<vector<double>>& d3 = data->points3D;

    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Point2f clickPoint(x, y);
        double minDistance = std::numeric_limits<double>::max();
        int closestIdx = -1;
        double distance = 0;

        for (size_t i = 0; i < cordsL.size(); i++) {
            double dx = clickPoint.x - cordsL[i][1];
            double dy = clickPoint.y - cordsL[i][0];
            distance = std::sqrt(dx * dx + dy * dy);
            if (distance < minDistance) {
                minDistance = distance;
                closestIdx = i;
            }
        }
        //std::cout << "Min distance: " << minDistance << " distance: " << distance << std::endl;
        if (closestIdx != -1) {
            std::cout << closestIdx << ") X: " << cordsL[closestIdx][1] << " Y: " << cordsL[closestIdx][0] << " Z: " << d3[closestIdx][2]<< std::endl;
            cv::circle(rectifiedLeft, cv::Point(cordsL[closestIdx][1], cordsL[closestIdx][0]), 2, cv::Scalar(255, 255, 255), -1);
            cv::putText(rectifiedLeft, std::to_string(closestIdx), cv::Point(cordsL[closestIdx][1], cordsL[closestIdx][0]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            cv::imshow("3D points on image", rectifiedLeft);
        }
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

    // Перевод изображений из цветного формата в монохромный
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageLeft, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRight, grayImageRight, cv::COLOR_BGR2GRAY);

    // Стереоректификация изображений
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Mat Q, R1, R2, P1, P2;

    cv::stereoRectify(stereo_par.cameraM1, stereo_par.distCoeffs1, stereo_par.cameraM2, stereo_par.distCoeffs2,
                      cv::Size(grayImageLeft.cols, grayImageLeft.rows), stereo_par.R, stereo_par.T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY);

    // Нахождение параметров ректификации и устранения искажений для левого изображения
    cv::initUndistortRectifyMap(stereo_par.cameraM1, stereo_par.distCoeffs1, R1, P1,
                                cv::Size(imageLeft.cols, imageLeft.rows), CV_32FC1, mapLx, mapLy);

    // Нахождение параметров ректификации и устранения искажений для правого изображения
    cv::initUndistortRectifyMap(stereo_par.cameraM2, stereo_par.distCoeffs2, R2, P2,
                                cv::Size(imageLeft.cols, imageLeft.rows), CV_32FC1, mapRx, mapRy);

    // Ректификация и устранение искажений с использованием полученных параметров
    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // Вывод ректифицированных изображений
    //cv::imshow("Rectified left image", rectifiedLeft);
    //cv::imshow("Rectified right image", rectifiedRight);

    // Создание окна с ползунками для настройки методов вычисления карты диспарантности
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity",800,600);

    //cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
    //cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);

    // Создание функций обратной связи для обновления параметров
    cv::createTrackbar("numDisparities", "disparity", &SGBM_par.numDisparities, 64, on_trackbar);
    cv::createTrackbar("blockSize", "disparity", &SGBM_par.blockSize, 15, on_trackbar);
    cv::createTrackbar("preFilterCap", "disparity", &SGBM_par.preFilterCap, 62, on_trackbar);
    cv::createTrackbar("uniquenessRatio", "disparity", &SGBM_par.uniquenessRatio, 50, on_trackbar);
    cv::createTrackbar("speckleRange", "disparity", &SGBM_par.speckleRange, 100, on_trackbar);
    cv::createTrackbar("speckleWindowSize", "disparity", &SGBM_par.speckleWindowSize, 25, on_trackbar);
    cv::createTrackbar("disp12MaxDiff", "disparity", &SGBM_par.disp12MaxDiff, 25, on_trackbar);
    cv::createTrackbar("minDisparity", "disparity", &SGBM_par.minDisparity, 25, on_trackbar);
    cv::createTrackbar("P1", "disparity", &SGBM_par.P1_, 10, on_trackbar);
    cv::createTrackbar("P2", "disparity", &SGBM_par.P2_, 10, on_trackbar);
    cv::waitKey(0);


    // Нахождение 3д точек
    cv::Mat pointsAll;
    cv::reprojectImageTo3D(disparity, pointsAll, Q, true, CV_32F);

    callbackData.depth = Q;
    callbackData.disparity = disparity;

    vector<double> limit_outlierArea {-8.0e3, -8.0e3, 250, 8.0e3, 8.0e3, 15.20e3};

    vector<vector<int>> vu;             // 2D координаты точки на изображении
    vector<vector<double>> xyz;         // 3D координаты точки на пространсве
    vector<vector<int>> rgb;            // цвет 3D точки

    for(int v = 0; v < pointsAll.rows; v++)
    {
        for(int u = 0; u < pointsAll.cols; u++)
        {

            cv::Vec3f xyz3D = pointsAll.at<cv::Vec3f>(v, u);


            if( xyz3D[0] < limit_outlierArea[0] ) continue;
            if( xyz3D[1] < limit_outlierArea[1] ) continue;
            if( xyz3D[2] < limit_outlierArea[2] ) continue;

            if( xyz3D[0] > limit_outlierArea[3] ) continue;
            if( xyz3D[1] > limit_outlierArea[4] ) continue;
            if( xyz3D[2] > limit_outlierArea[5] ) continue;


            vu.push_back({v, u});
            xyz.push_back(vector<double> ({xyz3D[0], xyz3D[1], xyz3D[2]}));
        }
    }

    callbackData.leftKey = vu;

    for (int i = 0; i < vu.size(); i++){
        int v = vu[i][0];
        int u = vu[i][1];
        cv::circle(rectifiedLeft, cv::Point(u, v), 3, cv::Scalar(20, 255, 0), -1);

        double x = xyz[i][0];
        double y = xyz[i][1];
        double z = xyz[i][2];
        callbackData.points3D.push_back(xyz[i]);

        if (i < 5){
            std::cout << std::endl;
            std::cout << "2D coords: [" << u << ", " << v << "] in pix" << std::endl;
            std::cout << "3D coords: [" << x << ", " << y << ", "<< z << "] in mm" << std::endl;
        }
    }


    //std::cout << xyz.at<cv::Vec3f>(0,0) << std::endl;

    /*
    vector<KeyPoint> keypointsLeft, keypointsRight;
    Mat descriptorsLeft, descriptorsRight;

    // Поиск ключевых точек на левом и правом изображениях
    Ptr<FeatureDetector> detector = ORB::create();
    detector->detect(rectifiedLeft, keypointsLeft);
    detector->detect(rectifiedRight, keypointsRight);

    // Сопоставление ключевых точек
    Ptr<DescriptorExtractor> extractor = ORB::create();
    extractor->compute(rectifiedLeft, keypointsLeft, descriptorsLeft);
    extractor->compute(rectifiedRight, keypointsRight, descriptorsRight);

    //Ptr<SIFT> sift = SIFT::create();
    //sift->detectAndCompute(rectifiedLeft, noArray(), keypointsLeft, descriptorsLeft);
    //sift->detectAndCompute(rectifiedRight, noArray(), keypointsRight, descriptorsRight);

    // Использование BFMatcher для сопоставления дескрипторов
    BFMatcher matcher(NORM_HAMMING);
    //BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Отображение сопоставленных точек
    Mat matchedImage;
    drawMatches(rectifiedLeft, keypointsLeft, rectifiedRight, keypointsRight, matches, matchedImage);

    // Смена цветовой схемы изображений для отрисовки цветных точек
    cv::cvtColor(rectifiedLeft, rectifiedLeft, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectifiedRight, rectifiedRight, cv::COLOR_GRAY2BGR);


    // Вывод результатов
    cv::imshow("Matched Points", matchedImage);


    std::vector<Point2f> x, y;
    for (size_t i = 0; i < matches.size(); i++){
        callbackData.leftKey.push_back(keypointsLeft[matches[i].queryIdx].pt);
        callbackData.rightKey.push_back(keypointsRight[matches[i].trainIdx].pt);

        callbackData.leftKey.push_back(keypointsLeft[matches[i].queryIdx].pt);
        callbackData.rightKey.push_back(keypointsRight[matches[i].trainIdx].pt);    
    }
    */


    // Вывод левого изображения и установка функции обратной связи для обработки кликов мыши
    cv::imshow("3D points on image", rectifiedLeft);
    cv::setMouseCallback("3D points on image", onMouseClick, &callbackData);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}








