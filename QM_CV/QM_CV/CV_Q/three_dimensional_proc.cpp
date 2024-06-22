#include "three_dimensional_proc.h"


std::vector<std::vector<double>> point3d_finder(cv::Mat imageL, cv::Mat imageR, stereo_output_par_t calib_par){

    // Стереоректификация изображений
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Mat Q, R1, R2, P1, P2;

    // Перевод изображений из цветного формата в монохромный
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageL, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageR, grayImageRight, cv::COLOR_BGR2GRAY);

    // Ректификация и устранение искажений
    cv::stereoRectify(calib_par.cameraM1, calib_par.distCoeffs1, calib_par.cameraM2, calib_par.distCoeffs2,
                      cv::Size(grayImageLeft.cols, grayImageLeft.rows), calib_par.R, calib_par.T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY);

    cv::initUndistortRectifyMap(calib_par.cameraM1, calib_par.distCoeffs1, R1, P1,
                                cv::Size(imageL.cols, imageL.rows), CV_32FC1, mapLx, mapLy);

    cv::initUndistortRectifyMap(calib_par.cameraM2, calib_par.distCoeffs2, R2, P2,
                                cv::Size(imageL.cols, imageL.rows), CV_32FC1, mapRx, mapRy);

    cv::Mat rectifiedLeft, rectifiedRight;

    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // Рассчёт карты диспаратности методом SGBM
    stereo_sgbm_t SGBM_par;
    cv::Mat disparity;

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    int cn = imageL.channels();
    stereo->setNumDisparities(SGBM_par.numDisparities*64);
    stereo->setBlockSize(SGBM_par.blockSize);
    stereo->setMinDisparity(SGBM_par.minDisparity);
    stereo->setPreFilterCap(SGBM_par.preFilterCap);
    stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
    stereo->setSpeckleWindowSize(SGBM_par.speckleWindowSize);
    stereo->setSpeckleRange(SGBM_par.speckleRange);
    stereo->setDisp12MaxDiff(SGBM_par.disp12MaxDiff);
    stereo->setP1(8*cn*SGBM_par.P1_*SGBM_par.P1_);
    stereo->setP1(32*cn*SGBM_par.P2_*SGBM_par.P2_);

    stereo_d_map(imageL, imageR, disparity, stereo);

    // Поиск 3д точек
    cv::Mat points23D;
    cv::reprojectImageTo3D(disparity, points23D, Q, true, CV_32F);

    std::vector<std::vector<double>> result;

    //std::vector<std::vector<int>> vu;             // 2D координаты точки на изображении
    //std::vector<std::vector<double>> xyz;         // 3D координаты точки в пространстве
    //std::vector<std::vector<int>> rgb;            // цвет 3D точки


    std::vector<double> limit_outlierArea {-8.0e3, -8.0e3, 250, 8.0e3, 8.0e3, 15.20e3};

    int valid_points = 0;


    for(int v = 0; v < points23D.rows; v++)
    {
        for(int u = 0; u < points23D.cols; u++)
        {
            cv::Vec3f xyz3D = points23D.at<cv::Vec3f>(v, u);

            if( xyz3D[0] < limit_outlierArea[0] ) continue;
            if( xyz3D[1] < limit_outlierArea[1] ) continue;
            if( xyz3D[2] < limit_outlierArea[2] ) continue;

            if( xyz3D[0] > limit_outlierArea[3] ) continue;
            if( xyz3D[1] > limit_outlierArea[4] ) continue;
            if( xyz3D[2] > limit_outlierArea[5] ) continue;

            //std::vector<double> pointData = {static_cast<double>(v), static_cast<double>(u), xyz3D[0], xyz3D[1], xyz3D[2]};
            //result.push_back(pointData);

            /*
            result.at<double>(valid_points,0) = v;
            result.at<double>(valid_points,1) = u;
            result.at<double>(valid_points,2) = xyz3D[0];
            result.at<double>(valid_points,3) = xyz3D[1];
            result.at<double>(valid_points,4) = xyz3D[2];
            */

            result.push_back({static_cast<double>(u), static_cast<double>(v), xyz3D[0], xyz3D[1], xyz3D[2]});

            valid_points++;
        }
    }
    return result;
}

cv::Vec3f third_coords(cv::Mat imageL, cv::Mat imageR, cv::Point xy, stereo_output_par_t calib_par) {

    // Перевод изображений из цветного формата в монохромный
    cv::Mat rectifiedLeft, rectifiedRight;
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageL, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageR, grayImageRight, cv::COLOR_BGR2GRAY);

    // Стереоректификация изображений
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Mat Q, R1, R2, P1, P2;

    cv::stereoRectify(calib_par.cameraM1, calib_par.distCoeffs1, calib_par.cameraM2, calib_par.distCoeffs2,
                      cv::Size(grayImageLeft.cols, grayImageLeft.rows), calib_par.R, calib_par.T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY);

    cv::initUndistortRectifyMap(calib_par.cameraM1, calib_par.distCoeffs1, R1, P1,
                                cv::Size(imageL.cols, imageL.rows), CV_32FC1, mapLx, mapLy);

    cv::initUndistortRectifyMap(calib_par.cameraM2, calib_par.distCoeffs2, R2, P2,
                                cv::Size(imageR.cols, imageR.rows), CV_32FC1, mapRx, mapRy);


    // Ректификация и устранение искажений
    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);


    // Рассчёт карты диспаратности методом SGBM
    stereo_sgbm_t SGBM_par;
    cv::Mat disparity;

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    int cn = imageL.channels();
    stereo->setNumDisparities(SGBM_par.numDisparities*64);
    stereo->setBlockSize(SGBM_par.blockSize);
    stereo->setMinDisparity(SGBM_par.minDisparity);
    stereo->setPreFilterCap(SGBM_par.preFilterCap);
    stereo->setUniquenessRatio(SGBM_par.uniquenessRatio);
    stereo->setSpeckleWindowSize(SGBM_par.speckleWindowSize);
    stereo->setSpeckleRange(SGBM_par.speckleRange);
    stereo->setDisp12MaxDiff(SGBM_par.disp12MaxDiff);
    stereo->setP1(8*cn*SGBM_par.P1_*SGBM_par.P1_);
    stereo->setP1(32*cn*SGBM_par.P2_*SGBM_par.P2_);

    stereo_d_map(imageL, imageR, disparity, stereo);


    cv::Mat coords3d;

    // Поиск 3д точек
    cv::reprojectImageTo3D(disparity, coords3d, Q, true, CV_32F);

    if (xy.x >= 0 && xy.x < coords3d.cols && xy.y >= 0 && xy.y < coords3d.rows ) {
        cv::Vec3f point3D = coords3d.at<cv::Vec3f>(xy);
        if (point3D[2] != 0 && point3D[2] < 10000) {
            return point3D;
        }
    }

    return cv::Vec3f(0, 0, 0);
}

void write_coords_file(std::vector<std::vector<double>> points23D,std::string filename){

    std::vector<double> limit_outlierArea {-8.0e3, -8.0e3, 250, 8.0e3, 8.0e3, 15.20e3};

    std::ofstream xyz_fs;
    xyz_fs.open(filename);

    int u,v;
    double x,y,z;

    if (xyz_fs.is_open()) {
        for(int i = 0; i < points23D.size(); i++)
        {

            v = points23D[i][0];
            u = points23D[i][1];

            x = points23D[i][2];
            y = points23D[i][3];
            z = points23D[i][4];

            xyz_fs << v <<"\t"<< u << "\t" << " \t" << x << "\t" << y << "\t" << z<< std::endl;
        }

        xyz_fs.close();

    } else {
        std::cerr << "Error while reading the " << filename << "." << std::endl;
    }
}
