#pragma once
#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <iostream>
#include <opencv2/opencv.hpp>
//#include "opencv2/core.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>


// Stereo create params
/*
int minDisparity = 0;  // 40
int numDisparities = 28; // or 2 - 15 for CUDA
int blockSize = 3; // 0
int preFilterType = 0; // P1
int preFilterSize = 8; // P2
int disp12MaxDiff = 25;
int preFilterCap = 40;
int uniquenessRatio = 1;
int speckleWindowSize = 0;
int speckleRange = 0;
int mode = StereoSGBM::MODE_SGBM; //  MODE_SGBM

int minDisparity = 0;  // 40
int numDisparities = 16; // or 2 - 15 for CUDA
int blockSize = 3; // 0
int P1_ = 1;
int P2_ = 1;
int disp12MaxDiff = 25;
int preFilterCap = 40;
int uniquenessRatio = 1;
int speckleWindowSize = 0;
int speckleRange = 0;
int mode = cv::StereoSGBM::MODE_SGBM; //  MODE_SGBM
*/

typedef struct StereoAlgorithmParams{
    int minDisparity = 0;  // 40
    int numDisparities = 24; // or 2 - 15 for CUDA
    int blockSize = 3; // 0
    int P1_ = 1;
    int P2_ = 1;
    int disp12MaxDiff = 25;
    int preFilterCap = 40;
    int uniquenessRatio = 1;
    int speckleWindowSize = 0;
    int speckleRange = 0;
    int mode = cv::StereoSGBM::MODE_SGBM; //  MODE_SGBM
}stereo_match_t;

void stereo_depth_map(cv::Mat rectified_image_left, cv::Mat rectified_image_right,
                      cv::Mat cameraMatrixLeft, cv::Mat cameraMatrixRight,
                      cv::Mat T, cv::Mat &disparity, int numDisparity, int minDisparity, cv::Ptr<cv::StereoSGBM> stereo);

void cuda_stereo_depth_map();

#endif // TRIANGULATE_H
