#include "triangulate.h"


/// Создание объектов для алгоритмов рассчёта карты диспарантности
cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
//cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
stereo_match_t SGBM_par;
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

void on_trackbar3( int, void* )
{
  stereo->setPreFilterCap(SGBM_par.preFilterCap);
}

void on_trackbar4( int, void* )
{
  stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
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

/// Рассчёт карты диспарантности с использованием алгоритма SGBM
void stereo_depth_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                      cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity/*,
                      cv::Ptr<cv::StereoSGBM> stereo*/){

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

    std::cout << "Baseline length: " << baseline << std::endl;
    std::cout << "Focal length: " << focalLength << std::endl;

    while (true){
        stereo->compute(rectifiedImageLeft, rectifiedImageRight, disparityMap);

        disparityMap.convertTo(disparity,CV_32F,1.0f);
        disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

        cv::imshow("Disparity Map", disparity);

        cv::normalize(disparity, depthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::applyColorMap(depthMap, coloredDepthMap, cv::COLORMAP_JET);
        cv::imshow("Depth Map", coloredDepthMap);

        if (cv::waitKey(1) == 27) break;
    }
}


/// Рассчёт карты диспарантности с использованием CUDA SGM
void cuda_stereo_depth_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,
                           cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                           cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity/*,
                           cv::Ptr<cv::cuda::StereoSGM> stereo*/){

    cv::Mat depthMap, coloredDepthMap, disparityMap;
    // Подготовка данных на GPU
    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedImLeft);
    gpuImageRight.upload(rectifiedImRight);

    cv::cuda::GpuMat gpuDisparityMap;

    while (true){
        stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

        // Скачивание результата с GPU
        gpuDisparityMap.download(disparityMap);

        disparityMap.convertTo(disparity,CV_32F, 1.0);
        disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

        // Визуализация карты диспарантности
        cv::imshow("Disparity Map", disparity);

        cv::normalize(disparityMap, depthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //depthMap = focalLength * baseline / (disparity);
        //cv::normalize(depthMap, depthMap, local_min, local_max, cv::NORM_MINMAX);
        //disparity.convertTo(depthMap, CV_8UC1, 255*(local_max-local_min));

        cv::applyColorMap(depthMap, coloredDepthMap, cv::COLORMAP_JET);
        cv::imshow("Depth Map", coloredDepthMap);

        if (cv::waitKey(1) == 27) break;
    }
}
