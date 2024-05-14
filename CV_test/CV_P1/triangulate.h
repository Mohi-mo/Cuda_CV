/*!
 * \file
 * \brief Заголовочный файл с определением структур и функций 3д калибровки,
 *
Данный файл содержит определения структур и функций, используемых для вычисления
и вывода карт диспарантности и глубины.
*/
#pragma once
#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>


/// Структура, содержащая параметры объектов StereoSGBM / BM
typedef struct StereoSGBMstruct{
    int minDisparity = 0;  // 40
    int numDisparities = 17;
    int blockSize = 2; // 0
    int P1_ = 0;
    int P2_ = 0;
    int disp12MaxDiff = 1;
    int preFilterCap = 45;
    int uniquenessRatio = 10;
    int speckleWindowSize = 25;
    int speckleRange = 10;
    int mode = cv::StereoSGBM::MODE_SGBM; //  MODE_SGBM
}stereo_sgbm_t;

typedef struct StereoBMstruct{
    int preFilterCap = 45;
    int preFilterSize = 9;
    int preFilterType = 0;
    //cv::Rect ROI1(10, 200, 1000, 700);
    //cv::Rect ROI2(200, 250, 800, 600);
    int blockSize = 7;
    int getTextureThreshhold = 30;
    int uniquenessRatio = 10;
    int numDisparities = 17;
}stereo_bm_t;

typedef struct CudaSGM_params {
    int numDisparities =16; // Значение кратное 4-м от 1
    int blockSize = 19;
    int numLevels = 14;
    int numIters = 6;
    int mode = cv::cuda::StereoSGM::MODE_SGBM;
}cuda_sgm_t;


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

// CUDA -------------------------------------------------------------------------------------------------------
void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity, cv::Ptr<cv::cuda::StereoSGM> &stereo);

void cuda_stereo_d_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight, cv::Mat &disparity, cv::Ptr<cv::cuda::StereoBeliefPropagation> &stereo);

#endif // TRIANGULATE_H
