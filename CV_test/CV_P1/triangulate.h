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
typedef struct StereoAlgorithmParams{
    int minDisparity = 1;  // 40
    int numDisparities = 15; // 16 or 1 for CUDA   27 - for not CUDA
    int blockSize = 2; // 0
    int P1_ = 0;
    int P2_ = 0;
    int disp12MaxDiff = 1;
    int preFilterCap = 45;
    int uniquenessRatio = 10;
    int speckleWindowSize = 25;
    int speckleRange = 2;
    int mode = cv::StereoSGBM::MODE_SGBM; //  MODE_SGBM
}stereo_match_t;

typedef struct CudaSGM_params {
    int numDisparity = 2;
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
void stereo_depth_map(cv::Mat rectifiedLeft, cv::Mat rectifiedRight, cv::Mat &disparity, cv::Ptr<cv::StereoSGBM> &stereo);

void stereo_depth_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight, cv::Mat &disparity, cv::Ptr<cv::StereoBM> &stereo);

// CUDA -------------------------------------------------------------------------------------------------------
void cuda_stereo_depth_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,
                           cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                           cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity/*,
                           cv::Ptr<cv::cuda::StereoSGM> stereo*/);


#endif // TRIANGULATE_H
