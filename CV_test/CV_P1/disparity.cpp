#include "disparity.h"


/// Рассчёт карты диспарантности с использованием алгоритма SGBM
void stereo_d_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight, cv::Mat &disparity,
                      cv::Ptr<cv::StereoSGBM> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    //while (true){
    stereo->compute(rectifiedImageLeft, rectifiedImageRight, disparityMap);

    disparityMap.convertTo(disparity,CV_32F,0.0625f);
    //disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    //cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    cv::imshow("Colored disparity Map", coloredDispMap);

    disparity = coloredDispMap;
        //if (cv::waitKey(0) != 27){

        //}
    //}
}

void stereo_d_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight, cv::Mat &disparity,
                      cv::Ptr<cv::StereoBM> &stereo){

    cv::Mat coloredDispMap, disparityMap;

    //while (true){
    stereo->compute(rectifiedImageLeft, rectifiedImageRight, disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    cv::medianBlur(disparity, disparity, 5);

    cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    cv::imshow("Colored disparity Map", coloredDispMap);

    //disparity = coloredDispMap;
        //if (cv::waitKey(0) != 27){

        //}
    //}
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

    //while (true){
    stereo->compute(gpuImageLeft, gpuImageRight, gpuDisparityMap);

    // Скачивание результата с GPU
    gpuDisparityMap.download(disparityMap);

    //stereo->compute(rectifiedImLeft, rectifiedImRight, disparityMap);

    disparityMap.convertTo(disparity,CV_32F,1.0f);
    disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

    //cv::medianBlur(disparity, disparity, 5);

    cv::imshow("Disparity Map", disparity);

    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
    cv::imshow("Colored disparity Map", coloredDispMap);
    /*

    //    if (cv::waitKey(1) == 27) break;
    //}
    */
}


void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity,
                       cv::Ptr<cv::cuda::StereoBeliefPropagation> &stereo){

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

        disparityMap.convertTo(disparity,CV_32F,1.0f);
        disparity = (disparity/32.0f - (double)stereo->getMinDisparity())/((double)stereo->getNumDisparities());

        //cv::medianBlur(disparity, disparity, 5);

        cv::imshow("Disparity Map", disparity);

        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::applyColorMap(disparity, coloredDispMap, cv::COLORMAP_JET);
        cv::imshow("Colored disparity Map", coloredDispMap);

    //    if (cv::waitKey(1) == 27) break;
    //}
}
