#pragma once
#ifndef THREE_DIMENSIONAL_PROC_H
#define THREE_DIMENSIONAL_PROC_H

#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

#include "disparity.h"
#include "calibrate.h"


void point3d_finder(cv::Mat imageL, cv::Mat imageR, cv::Mat &points23D);

cv::Vec3f third_coords(cv::Mat imageL, cv::Mat imageR, cv::Point xy, cv::Mat coords3d);

#endif // THREE_DIMENSIONAL_PROC_H
