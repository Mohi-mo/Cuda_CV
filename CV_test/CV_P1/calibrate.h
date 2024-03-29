#pragma once
#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <iostream>
//#include <linux/delay.h>

#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>
//#include <opencv2/cudaarithm.hpp>


using namespace std;
using namespace cv;
using namespace cuda;


typedef struct MonoOutputParams{
  cv::Mat cameraMatrix;     // Матрица камеры
  cv::Mat distCoeffs;       // Вектор коэффициентов дисторсии
  cv::Mat rvecs;            // Кортеж векторов поворота для перехода из базиса объекта в базис камеры
  cv::Mat tvecs;            // Кортеж векторов смещения для перехода из базиса объекта в базис камеры
  cv::Mat stdDevIntrinsics; // Вектор оценок внутренних параметров камеры
  cv::Mat stdDevExtrinsics; // Вектор оценок внешних параметров камеры
  cv::Mat perViewErrors;    // Вектор среднеквадратической ошибки перепроецирования для каждого вида
  double RMS;               // Значение среднеквадратической ошибки перепроецирования
}mono_output_par_t;

typedef struct StereOutputParams{
    cv::Mat cameraM1;       // Матрица камеры 1
    cv::Mat cameraM2;       // Матрица камеры 2
    cv::Mat distCoeffs1;    // Вектор коэффициентов дисторсии камеры 1
    cv::Mat distCoeffs2;    // Вектор коэффициентов дисторсии камеры 2
    cv::Mat R;              // Матрица поворотов
    cv::Mat T;              // Вектор смещений
    cv::Mat E;              // Матрица существенных параметров
    cv::Mat F;              // Фундаментальная матрица
    cv::Mat rvecs;          // Кортеж векторов поворота для перехода из базиса объекта в базис камеры
    cv::Mat tvecs;          // Кортеж векторов смещения для перехода из базиса объекта в базис камеры
    cv::Mat perViewErrors;  // Вектор среднеквадратической ошибки перепроецирования для каждого вида
    double RMS;             // Значение среднеквадратической ошибки перепроецирования
}stereo_output_par_t;


// Функция калибровки одной камеры
void calibrate_camera(std::vector<cv::String> images, std::string path, std::string dataset_name,
                      int checkerboard_w, int checkerboard_h, mono_output_par_t &mono_out);

// Функция калибровки стереопары
void calibrate_stereo(std::vector<cv::String> im1, std::vector<cv::String> im2, std::string path1, std::string path2,
                      std::string dataset_name, int checkerboard_w, int checkerboard_h, stereo_output_par_t &outp_params);

#endif // CALLIBRATE_H
