#include "triangulate.h"

void stereo_depth_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                      cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity,
                      cv::Ptr<cv::StereoSGBM> stereo){

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

void stereo_depth_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                      cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity,
                      cv::Ptr<cv::StereoBM> stereo){

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

        cv::normalize(disparity, depthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(depthMap, coloredDepthMap, cv::COLORMAP_JET);

        cv::imshow("Disparity Map", disparity);
        cv::imshow("Depth Map", coloredDepthMap);

        if (cv::waitKey(1) == 27) break;
    }
}



// Рассчёт карты диспарантности на CUDA
void cuda_stereo_depth_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,
                           cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                           cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity,
                           cv::Ptr<cv::cuda::StereoSGM> stereo){

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



// Рассчёт карты диспарантности на CUDA Stereo belief propagation
void cuda_stereo_depth_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,
                           cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                           cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity,
                           cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo){

    cv::Mat depthMap, coloredDepthMap, disparityMap;
    // Подготовка данных на GPU

    cv::cuda::GpuMat gpuImageLeft, gpuImageRight;
    gpuImageLeft.upload(rectifiedImLeft);
    gpuImageRight.upload(rectifiedImRight);

    cv::cuda::GpuMat gpuDisparityMap;

   // while (true){
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

        //if (cv::waitKey(1) == 27) break;
    //}
}

