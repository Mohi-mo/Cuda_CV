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
cv::Mat point3D;
cv::Mat image3D;
cv::Mat disparity;;

/// Объявление структур для хранения параметров калибровки
stereo_output_par_t stereo_par;
struct MouseCallbackData {
    //std::vector<KeyPoint> keypointsLeft;
    std::vector<Point2f> leftKey;
    std::vector<Point2f> rightKey;
    cv::Mat disparity;
    cv::Mat depth;
    double focalLenght;
    double baseline;
};

stereo_match_t SGBM_par;

float min_y = 10000.0;
float max_y = -10000.0;
float min_x =  10000.0;
float max_x = -10000.0;

/// Создание объектов для алгоритмов рассчёта карты диспарантности
//cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();

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
    std::vector<cv::Point2f>& cordsL = data->leftKey;
    std::vector<cv::Point2f>& cordsR = data->rightKey;
    cv::Mat& disparity = data->disparity;

    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Point2f clickPoint(x, y);
        double minDistance = std::numeric_limits<double>::max();
        int closestIdx = -1;
        double distance = 0;

        for (size_t i = 0; i < 500; i++) {
            double dx = clickPoint.x - cordsL[i].x;
            double dy = clickPoint.y - cordsL[i].y;
            distance = std::sqrt(dx * dx + dy * dy);
            if (distance < minDistance) {
                minDistance = distance;
                closestIdx = i;
            }
        }
        //std::cout << "Min distance: " << minDistance << " distance: " << distance << std::endl;
        if (closestIdx != -1) {
            //double disparityValue = disparity.at<double>(cordsL[closestIdx]);
            //double depth = (data->baseline * data->focalLenght) / (disparityValue + 0.0001);

            cv::circle(rectifiedLeft, (cordsL[closestIdx], cordsL[closestIdx]), 3, cv::Scalar(0, 0, 0), -1);
            std::cout << "X: " << cordsL[closestIdx].x << " Y: " << cordsL[closestIdx].y << " Z: " << data->disparity.at<double>(cordsL[closestIdx]) << std::endl;

            cv::imshow("3D points on image", rectifiedLeft);
        }
    }
}


int main(int argc, char** argv) {
    mono_output_par_t mono_parL;
    mono_output_par_t mono_parR;

    MouseCallbackData callbackData;
    //callbackData.disparity;

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

    double fx_pixL = stereo_par.cameraM1.at<double>(0, 0);
    double fy_pixL = stereo_par.cameraM1.at<double>(1, 1);

    //double fx_pixR = stereo_par.cameraM2.at<double>(0, 0);
    //double fy_pixR = stereo_par.cameraM2.at<double>(1, 1);

    float focalLengthL = sqrt(fx_pixL*fx_pixL+fy_pixL*fy_pixL);
    float focalLength = focalLengthL;

    double baseline = std::abs(stereo_par.T.at<double>(0));

    callbackData.baseline = baseline;
    callbackData.focalLenght = focalLength;

    std::cout << std::endl;
    std::cout << "Baseline length: " << baseline << std::endl;
    std::cout << "Focal length: " << focalLength << std::endl;

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
    cv::createTrackbar("preFilterCap", "disparity", &SGBM_par.preFilterCap, 1, on_trackbar);
    cv::createTrackbar("uniquenessRatio", "disparity", &SGBM_par.uniquenessRatio, 50, on_trackbar);
    cv::createTrackbar("speckleRange", "disparity", &SGBM_par.speckleRange, 100, on_trackbar);
    cv::createTrackbar("speckleWindowSize", "disparity", &SGBM_par.speckleWindowSize, 25, on_trackbar);
    cv::createTrackbar("disp12MaxDiff", "disparity", &SGBM_par.disp12MaxDiff, 25, on_trackbar);
    cv::createTrackbar("minDisparity", "disparity", &SGBM_par.minDisparity, 25, on_trackbar);
    cv::createTrackbar("P1", "disparity", &SGBM_par.P1_, 10, on_trackbar);
    cv::createTrackbar("P2", "disparity", &SGBM_par.P2_, 10, on_trackbar);
    cv::waitKey(0);

    cv::Mat maxInColumns = cv::Mat::zeros(disparity.cols, 1, CV_8UC1);
    int matMax = INT_MIN;
    int matMin = INT_MAX;
    for (int i = 0; i < disparity.cols; i++)
    {
        int max = INT_MIN;
        int min = INT_MAX;
        for (int j = 0; j < disparity.rows; j++)
        {
        int val = (int)(disparity.at<uchar>(j, i));
            if (val > max)
                max = val;
            if (val < min)
                min = val;
        }

        maxInColumns.at<uchar>(i, 0) = max;
        if (max > matMax)
            matMax = max;
        if (min < matMin)
            matMin = min;
    }

    int map_width = 320;
    int map_height = 240;

    cv::Mat points;
    cv::reprojectImageTo3D(maxInColumns, points, Q);

    callbackData.disparity = maxInColumns;
    cv::Mat xy_projection = cv::Mat::zeros(map_height, map_width, CV_8UC1);

    for(int i = 0; i< points.rows; i++){
        float cur_y = -points.at<cv::Vec3f>(i, 0)[0];
        float cur_x = points.at<cv::Vec3f>(i, 0)[1];

        if (!isinf(cur_y)) {
            min_y = std::min(cur_y, min_y);
            max_y = std::max(cur_y, max_y);
        }

        if (!isinf(cur_x)){
            max_x = std::max(cur_x, max_x);
            min_x = std::min(cur_x, min_x);
        }

        int xx = (int)((cur_x)) + (int)(map_width/2); // zero point is in the middle of the map
        int yy = map_height - (int)((cur_y-min_y));     // zero point is at the bottom of the map

        // If the point fits on our 2D map - let's draw it!
        if (xx < map_width && xx >= 0 && yy < map_height && yy >= 0){
            xy_projection.at<uchar>(yy, xx) = maxInColumns.at<uchar>(i, 0); //maximized_line.at<uchar>(i, 0);
        }
    }

    // Нахождение 3д точек
    //reprojectImageTo3D(disparity, point3D, Q, false, -1);

    vector<KeyPoint> keypointsLeft, keypointsRight;
    Mat descriptorsLeft, descriptorsRight;

    // Поиск ключевых точек на левом и правом изображениях
    Ptr<FeatureDetector> detector = ORB::create();
    detector->detect(disparity, keypointsLeft);
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

        cv::circle(rectifiedLeft, (callbackData.leftKey[i]), 5, cv::Scalar(150, 150, 150), -1);
    }

    cv::Mat depth = (focalLength * baseline) / (disparity);
    callbackData.depth = depth;

    // Вывод левого изображения и установка функции обратной связи для обработки кликов мыши
    cv::imshow("3D points on image", rectifiedLeft);
    cv::setMouseCallback("3D points on image", onMouseClick, &callbackData);

    /*
*/
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}








