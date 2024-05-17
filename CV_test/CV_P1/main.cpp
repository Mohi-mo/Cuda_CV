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


using namespace std;
using namespace cv;
using namespace cuda;


/// Объявление переменных (временно)
/// \todo Убрать глобальные переменные, сделать их передачу как параметров
cv::Mat rectifiedLeft, rectifiedRight;
cv::Mat disparity;
int click_counter = 0;

int mode = 1; // Смена алгоритмов Stereo

float min_y = 10000.0;
float max_y = -10000.0;
float min_x =  10000.0;
float max_x = -10000.0;


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


#define CUDA //NO_CUDA

#ifdef CUDA
    cuda_bm_t cuda_bm_par;
    cuda_sgm_t cuda_sgm_par;
    cuda_bp_t cuda_bp_par;
    cuda_csbp_t cuda_csbp_par;


    /// Определение функции, задающей параметры алгоритмов рассчёта карты диспарантности
    void on_trackbar(int, void*){

        /// Вычисление карты диспарантности с использованием CUDA
        if (mode == 0){

            cv::Ptr<cv::cuda::StereoBM> stereo = cv::cuda::createStereoBM();
            stereo->setNumDisparities(cuda_bm_par.numDisparities);

            cuda_stereo_d_map(rectifiedLeft, rectifiedRight, disparity, stereo);

        } else if (mode == 1){ // cudaSGM

            cv::Ptr<cv::cuda::StereoSGM> stereo = cv::cuda::createStereoSGM();
            stereo->setNumDisparities(cuda_sgm_par.numDisparities*16);
            stereo->setBlockSize(cuda_sgm_par.blockSize);

            cuda_stereo_d_map(rectifiedLeft, rectifiedRight, disparity, stereo);

        } else if (mode == 2){ // cudaBP

            cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo = cv::cuda::createStereoBeliefPropagation();
            stereo->setNumDisparities(cuda_bp_par.numDisparities);
            stereo->setBlockSize(cuda_bp_par.blockSize);
            stereo->setNumIters(cuda_bp_par.numIters);
            stereo->setNumLevels(cuda_bp_par.numLevels);

            cuda_stereo_d_map(rectifiedLeft, rectifiedRight, disparity, stereo);
            //cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(rectifiedLeft.cols, rectifiedLeft.rows, ndisp, iters, levels);

        } else if (mode == 3) {

            cv::Ptr<cv::cuda::StereoConstantSpaceBP> stereo = cv::cuda::createStereoConstantSpaceBP();
            stereo->setNumDisparities(cuda_csbp_par.numDisparities);
            stereo->setBlockSize(cuda_csbp_par.blockSize);
            stereo->setNumIters(cuda_csbp_par.numIters);
            stereo->setNumLevels(cuda_csbp_par.numLevels);
            stereo->setSpeckleRange(cuda_csbp_par.speckleRange);
            stereo->setSpeckleWindowSize(cuda_csbp_par.speckleWindowSize);
            stereo->setDisp12MaxDiff(cuda_csbp_par.disp12MaxDiff);

            cuda_stereo_d_map(rectifiedLeft, rectifiedRight, disparity, stereo);

        }
    }
#endif

#ifdef NO_CUDA
    /// Объявление структуры для хранения параметров BM / SGBM
    stereo_sgbm_t SGBM_par;
    stereo_bm_t bm_par;


    /// Определение функции, задающей параметры алгоритмов рассчёта карты диспарантности
    void on_trackbar(int, void*){

        if (mode_ == 0) { // BM

            cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();  // Создание объекта для алгоритмов рассчёта карты диспарантности
            stereo->setPreFilterCap(bm_par.preFilterCap);
            stereo->setPreFilterSize(bm_par.preFilterSize);
            stereo->setPreFilterType(bm_par.preFilterType);
            //stereo->setROI1(bm_par.ROI1);
            //stereo->setROI2(bm_par.ROI2);
            stereo->setSmallerBlockSize(bm_par.blockSize+5);
            stereo->setTextureThreshold(bm_par.getTextureThreshhold);
            stereo->setNumDisparities(bm_par.numDisparities*16);
            stereo->setUniquenessRatio(bm_par.uniquenessRatio);

            /// Вычисление карты диспарантности
            stereo_d_map(rectifiedLeft, rectifiedRight, disparity, stereo);

        } else {

            // SGBM
            cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
            int cn = rectifiedLeft.channels();
            stereo->setNumDisparities(SGBM_par.numDisparities*64);
            stereo->setBlockSize(SGBM_par.blockSize);
            stereo->setMinDisparity(SGBM_par.minDisparity);
            stereo->setPreFilterCap(SGBM_par.preFilterCap);
            stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
            stereo->setSpeckleWindowSize(SGBM_par.speckleWindowSize);
            stereo->setSpeckleRange(SGBM_par.speckleRange);
            stereo->setDisp12MaxDiff(SGBM_par.disp12MaxDiff);
            stereo->setP1(8*cn*SGBM_par.P1_*SGBM_par.P1_);
            stereo->setP1(32*cn*SGBM_par.P2_*SGBM_par.P2_);

            /// Вычисление карты диспарантности
            stereo_d_map(rectifiedLeft, rectifiedRight, disparity, stereo);
        }

    }
#endif


/// Функция обратного вызова по клику мышкой
void onMouseClick(int event, int x, int y, int flags, void* userdata) {
    MouseCallbackData* data = static_cast<MouseCallbackData*>(userdata);
    vector<vector<int>>& cordsL = data->leftKey;
    vector<vector<double>>& d3 = data->points3D;

    if (event == cv::EVENT_LBUTTONDOWN) {

        std::cout << click_counter++ << std::endl;

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
        double prev_x = 0, prev_y = 0, prev_z = 0;
        double curr_x, curr_y , curr_z = 0;
        double dist;

        //std::cout << "Min distance: " << minDistance << " distance: " << distance << std::endl;
        if (closestIdx != -1) {
            std::cout << closestIdx << ") X: " << cordsL[closestIdx][1] << " Y: " << cordsL[closestIdx][0] << " Z: " << d3[closestIdx][2]/1000.0<< std::endl;
            cv::circle(rectifiedLeft, cv::Point(cordsL[closestIdx][1], cordsL[closestIdx][0]), 2, cv::Scalar(255, 255, 255), -1);
            cv::putText(rectifiedLeft, std::to_string(closestIdx), cv::Point(cordsL[closestIdx][1], cordsL[closestIdx][0]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

            //curr_x = d3[closestIdx][0]/1000.0;
            //curr_y = d3[closestIdx][1]/1000.0;
            //curr_z = d3[closestIdx][2]/1000.0;

            /*
            if (click_counter == 2){
                dist = sqrt((prev_x - curr_x)*(prev_x - curr_x) + (prev_y - curr_y)*(prev_y - curr_y) + (prev_z - curr_z)*(prev_z - curr_z));
                std::cout << "Point dist: " << dist << std::endl;
                prev_x = curr_x;
                prev_y = curr_y;
                //prev_z = curr_z;
                std::cout << "Point X: "<<  prev_x << " " << prev_x << std::endl;
                std::cout << "Point Y: "<<  prev_y << " " << prev_y << std::endl;
                std::cout << "Point Z: "<<  prev_z << " " << prev_z << std::endl;
                click_counter = 0;
            }
            */
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

    #ifdef NO_CUDA
    // Создание функций обратной связи для обновления параметров


    if (mode_ == 0){ // BM

        cv::createTrackbar("Prefilter cap", "disparity", &bm_par.preFilterCap, 50, on_trackbar);
        cv::createTrackbar("Prefilter size", "disparity", &bm_par.preFilterSize, 255, on_trackbar);
        cv::createTrackbar("Prefilter type", "disparity", &bm_par.preFilterType, 2, on_trackbar);
        //cv::createTrackbar("ROI 1", "disparity", &bm_par.ROI1, 100, on_trackbar);
        //cv::createTrackbar("ROI 2", "disparity", &bm_par.ROI2, 100, on_trackbar);
        cv::createTrackbar("Smaller block size", "disparity", &bm_par.blockSize, 25, on_trackbar);
        cv::createTrackbar("Texture threshold", "disparity", &bm_par.getTextureThreshhold, 100, on_trackbar);
        cv::createTrackbar("Num disparities", "disparity", &bm_par.numDisparities, 100, on_trackbar);

    } else { // SGBM

        cv::createTrackbar("numDisparities", "disparity", &SGBM_par.numDisparities, 64, on_trackbar);
        cv::createTrackbar("blockSize", "disparity", &SGBM_par.blockSize, 15, on_trackbar);
        cv::createTrackbar("preFilterCap", "disparity", &SGBM_par.preFilterCap, 62, on_trackbar);
        cv::createTrackbar("uniquenessRatio", "disparity", &SGBM_par.uniquenessRatio, 50, on_trackbar);
        cv::createTrackbar("speckleRange", "disparity", &SGBM_par.speckleRange, 100, on_trackbar);
        cv::createTrackbar("speckleWindowSize", "disparity", &SGBM_par.speckleWindowSize, 500, on_trackbar);
        cv::createTrackbar("disp12MaxDiff", "disparity", &SGBM_par.disp12MaxDiff, 25, on_trackbar);
        cv::createTrackbar("minDisparity", "disparity", &SGBM_par.minDisparity, 25, on_trackbar);
        cv::createTrackbar("P1", "disparity", &SGBM_par.P1_, 500, on_trackbar);
        cv::createTrackbar("P2", "disparity", &SGBM_par.P2_, 500, on_trackbar);

    }

    #endif

    #ifdef CUDA

        if (mode == 0){

            cv::createTrackbar("numDisparities", "disparity", &cuda_bm_par.numDisparities, 500, on_trackbar);

        } else if (mode == 1){

            cv::createTrackbar("numDisparities", "disparity", &cuda_sgm_par.numDisparities, 500, on_trackbar);
            cv::createTrackbar("Block size", "disparity", &cuda_sgm_par.blockSize, 15, on_trackbar);

        } else if (mode == 2){

            cv::createTrackbar("numDisparities", "disparity", &cuda_bp_par.numDisparities, 256, on_trackbar);
            cv::createTrackbar("Block size", "disparity", &cuda_bp_par.blockSize, 15, on_trackbar);
            cv::createTrackbar("Num iters", "disparity", &cuda_bp_par.numIters, 15, on_trackbar);
            cv::createTrackbar("Num levels", "disparity", &cuda_bp_par.numLevels, 9, on_trackbar);

        } else {

        }

    #endif

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
    }


    //std::cout << xyz.at<cv::Vec3f>(0,0) << std::endl;

    vector<KeyPoint> keypointsLeft, keypointsRight;
    Mat descriptorsLeft, descriptorsRight;

    // Поиск ключевых точек на левом и правом изображениях
    //Ptr<FeatureDetector> detector = ORB::create();
    //detector->detect(rectifiedLeft, keypointsLeft);
    //detector->detect(rectifiedRight, keypointsRight);

    // Сопоставление ключевых точек
    //Ptr<DescriptorExtractor> extractor = ORB::create();
    //extractor->compute(rectifiedLeft, keypointsLeft, descriptorsLeft);
    //extractor->compute(rectifiedRight, keypointsRight, descriptorsRight);

    //Ptr<SIFT> sift = SIFT::create();
    //sift->detectAndCompute(rectifiedLeft, noArray(), keypointsLeft, descriptorsLeft);
    //sift->detectAndCompute(rectifiedRight, noArray(), keypointsRight, descriptorsRight);

    // Использование BFMatcher для сопоставления дескрипторов
    //BFMatcher matcher(NORM_HAMMING);
    //BFMatcher matcher(NORM_L2);
    //vector<DMatch> matches;
    //matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Отображение сопоставленных точек
    //Mat matchedImage;
    //drawMatches(rectifiedLeft, keypointsLeft, rectifiedRight, keypointsRight, matches, matchedImage);

    // Смена цветовой схемы изображений для отрисовки цветных точек
    cv::cvtColor(rectifiedLeft, rectifiedLeft, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectifiedRight, rectifiedRight, cv::COLOR_GRAY2BGR);


    // Вывод результатов
    //cv::imshow("Matched Points", matchedImage);


    // Вывод левого изображения и установка функции обратной связи для обработки кликов мыши
    cv::imshow("3D points on image", rectifiedLeft);
    cv::setMouseCallback("3D points on image", onMouseClick, &callbackData);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}








