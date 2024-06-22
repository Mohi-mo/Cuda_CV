/*!
 * \file
 * \brief Заголовочный файл с определением структур и функций 3д калибровки,
 *
Данный файл содержит определения структур и функций, используемых для вычисления
и вывода карт диспарантности и глубины.
*/
#pragma once
#ifndef DISPARITY_H
#define DISPARITY_H

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudastereo.hpp>


/// Структура, содержащая параметры объектов StereoSGBM / BM
typedef struct StereoSGBMstruct{
    int minDisparity = 0;  // 40
    int numDisparities = 4;
    int blockSize = 2; // 0
    int P1_ = 0;
    int P2_ = 0;
    int disp12MaxDiff = 0;
    int preFilterCap = 0;
    int uniquenessRatio = 3;
    int speckleWindowSize = 80;
    int speckleRange = 2;
}stereo_sgbm_t;

typedef struct StereoBMstruct{
    int preFilterCap = 31;
    int preFilterSize = 7;
    int preFilterType = cv::StereoBM::PREFILTER_XSOBEL;
    //cv::Rect ROI1(10, 200, 1000, 700);
    //cv::Rect ROI2(200, 250, 800, 600);
    int blockSize = 7;
    int getTextureThreshhold = 10;
    int uniquenessRatio = 15;
    int numDisparities = 17;
}stereo_bm_t;


typedef struct CudaBM_params {
    int numDisparities = 128; // Значение кратное 4-м
    int numIters = 5;
    int numLevels = 5;
    int msgType = CV_32F;
}cuda_bm_t;

typedef struct CudaSGM_params {
    int numDisparities = 16; // Значение кратное 16-м
    int blockSize = 7;
    int numLevels = 5;
    int numIters = 10;
}cuda_sgm_t;

typedef struct CudaBP_params {
    int numDisparities = 256; // Значение кратное 4-м
    int blockSize = 3;
    int numLevels = 5; // <9
    int numIters = 10; // ??
}cuda_bp_t;

typedef struct CudaCSBP_params {
    int numDisparities = 128;
    int blockSize = 3;
    int numIters = 20;
    int numLevels = 6;
    int speckleRange = 2;
    int speckleWindowSize = 100;
    int disp12MaxDiff = 0;
}cuda_csbp_t;


/*!
 *  \brief Функция для вычисления карты диспарантности методом SGBM
 *
 *  \param[in] rectifiedLeft Ректифицированное левое изображение
 *  \param[in] rectifiedRight Ректифицированное правое изображение
 *  \param[in] cameraMatrixLeft Матрица левой камеры
 *  \param[in] cameraMatrixRight Матрица правой камеры
 *  \param[in] T Матрица смещений, получаемая при калибровке
 *  \param[in] numDisparities Количество диспарантностей
 *  \param[in] minDisparity Минимальное значение диспарантности
 *  \param[in] stereo Объект StereoSGBM
 *
 *  \param[out] disparity Матрица значенией диспарантности
*/
void stereo_d_map(cv::Mat rectifiedLeft, cv::Mat rectifiedRight, cv::Mat &disparity, cv::Ptr<cv::StereoSGBM> &stereo);

void stereo_d_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight, cv::Mat &disparity, cv::Ptr<cv::StereoBM> &stereo);

/// ====
/// CUDA
/// ====
//void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity, cv::Ptr<cv::cuda::StereoBM> &stereo);

//void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity, cv::Ptr<cv::cuda::StereoSGM> &stereo);

//void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity, cv::Ptr<cv::cuda::StereoBeliefPropagation> &stereo);

//void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity, cv::Ptr<cv::cuda::StereoConstantSpaceBP> &stereo);

#endif // DISPARITY_H
