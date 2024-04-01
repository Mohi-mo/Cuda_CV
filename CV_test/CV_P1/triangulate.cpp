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
        cv::minMaxLoc(disparity, &local_min, &local_max);

        //depthMap = focalLength * baseline / (disparity);
        //cv::normalize(depthMap, depthMap, local_min, local_max, cv::NORM_MINMAX);
        disparity.convertTo(depthMap, CV_8UC1, 255*(local_max-local_min));

        cv::applyColorMap(depthMap, coloredDepthMap, cv::COLORMAP_JET);
        cv::imshow("Depth Map", coloredDepthMap);

        if (cv::waitKey(1) == 27) break;
    }
    std::cout << "Min disparity [" << local_min << "]; Max Disparity [" << local_max << "]" << std::endl;
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
