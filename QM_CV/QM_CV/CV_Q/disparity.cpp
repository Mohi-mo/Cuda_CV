#include "disparity.h"


/// ====================================
/// Рассчёт карты диспаратности без CUDA
/// ====================================
void stereo_d_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight, cv::Mat &disparity,
                      cv::Ptr<cv::StereoSGBM> &stereo){

    cv::Mat coloredDispMap, disparityMap;
    stereo->compute(rectifiedImageLeft, rectifiedImageRight, disparityMap);

    disparityMap.convertTo(disparity,CV_32F,0.0625f);
    //disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);

    cv::Mat filtered;
    cv::medianBlur(disparity, filtered, 11);

    cv::imshow("Colored disparity Map", coloredDispMap);
    cv::imshow("Filtered disparity Map", filtered);

    //disparity = coloredDispMap;
    disparity = filtered;

    cv::applyColorMap(filtered, filtered, cv::COLORMAP_JET);
    cv::imshow("Colorized filter Map", filtered);
}

void stereo_d_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight, cv::Mat &disparity,
                      cv::Ptr<cv::StereoBM> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    stereo->compute(rectifiedImageLeft, rectifiedImageRight, disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    //disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    cv::medianBlur(disparity, disparity, 5);

    //cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    //cv::imshow("Colored disparity Map", coloredDispMap);
}


/// =================================================
/// Функции поиска карт диспаратности с помощью CUDA
/// =================================================
void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,  cv::Mat &disparity,
                           cv::Ptr<cv::cuda::StereoBM> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    // Подготовка данных на GPU
    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedImLeft);
    gpuImageRight.upload(rectifiedImRight);

    cv::cuda::GpuMat gpuDisparityMap;

    stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

    gpuDisparityMap.download(disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    cv::imshow("Colored disparity Map", coloredDispMap);
}

/// Рассчёт карты диспарантности с использованием CUDA SGM
void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,  cv::Mat &disparity,
                           cv::Ptr<cv::cuda::StereoSGM> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    // Подготовка данных на GPU
    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedImLeft);
    gpuImageRight.upload(rectifiedImRight);

    cv::cuda::GpuMat gpuDisparityMap;

    stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

    gpuDisparityMap.download(disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    cv::medianBlur(disparity, disparity, 5);

    //cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    //cv::imshow("Colored disparity Map", coloredDispMap);
}


void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity,
                       cv::Ptr<cv::cuda::StereoBeliefPropagation> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedImLeft);
    gpuImageRight.upload(rectifiedImRight);

    cv::cuda::GpuMat gpuDisparityMap;

    stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

    gpuDisparityMap.download(disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    //cv::medianBlur(disparity, disparity, 5);

    //cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    //cv::imshow("Colored disparity Map", coloredDispMap);

}

void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,  cv::Mat &disparity,
                           cv::Ptr<cv::cuda::StereoConstantSpaceBP> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    // Подготовка данных на GPU
    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedImLeft);
    gpuImageRight.upload(rectifiedImRight);

    cv::cuda::GpuMat gpuDisparityMap;

    //while (true){
    stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

    // Скачивание результата с GPU
    gpuDisparityMap.download(disparityMap);

    //stereo->compute(rectifiedImLeft, rectifiedImRight, disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    disparity = (disparity/16.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    //cv::medianBlur(disparity, disparity, 5);

    //cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    //cv::imshow("Colored disparity Map", coloredDispMap);
}
