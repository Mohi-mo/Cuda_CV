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
    int minDisparity = 0;  // 40
    int numDisparities = 27; // or 1 for CUDA
    int blockSize = 3; // 0
    int P1_ = 10;
    int P2_ = 1;
    int disp12MaxDiff = 25;
    int preFilterCap = 40;
    int uniquenessRatio = 1;
    int speckleWindowSize = 0;
    int speckleRange = 0;
    int mode = cv::StereoSGBM::MODE_SGBM; //  MODE_SGBM
}stereo_match_t;


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
void stereo_depth_map(cv::Mat rectifiedLeft, cv::Mat rectifiedRight,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight, cv::Mat T,
                      cv::Mat &disparity, int numDisparities, int minDisparity, cv::Ptr<cv::StereoSGBM> stereo);

/// Функция для вычисления карты диспарантности методом BM
void stereo_depth_map(cv::Mat rectifiedImageLeft, cv::Mat rectifiedImageRight,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight, cv::Mat T,
                      cv::Mat &disparity, int numDisparities, int minDisparity, cv::Ptr<cv::StereoBM> stereo);


// CUDA -------------------------------------------------------------------------------------------------------
void cuda_stereo_depth_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,
                           cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                           cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity,
                           cv::Ptr<cv::cuda::StereoSGM> stereo);


void cuda_stereo_depth_map(cv::Mat rectifiedImLeft, cv::Mat rectifiedImRight,
                           cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                           cv::Mat T, cv::Mat &disparity, int numDisparities, int minDisparity,
                           cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo);


#endif // TRIANGULATE_H
