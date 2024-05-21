#include "three_dimensional_proc.h"

void point3d_finder(cv::Mat imageL, cv::Mat imageR, cv::Mat &points23D){

    stereo_output_par_t calib_par;
    cv::FileStorage stereo_fs;
    if (stereo_fs.open("../../Calibration_parameters(stereo)/A6_stereo_camera_parameters.yml", cv::FileStorage::READ)){
        if (stereo_fs.isOpened()){
            stereo_fs["cameraMatrixL"]              >> calib_par.cameraM1;
            stereo_fs["cameraMatrixR"]              >> calib_par.cameraM2;
            stereo_fs["DistorsionCoeffsL"]          >> calib_par.distCoeffs1;
            stereo_fs["DistorsionCoeffsR"]          >> calib_par.distCoeffs2;
            stereo_fs["RotationMatrix"]             >> calib_par.R;
            stereo_fs["TranslationMatrix"]          >> calib_par.T;
            stereo_fs["EssentialMatrix"]            >> calib_par.E;
            stereo_fs["FundamentalMatrix"]          >> calib_par.F;
            stereo_fs["VectorOfRotationVectors"]    >> calib_par.rvecs;
            stereo_fs["VectorOfTranslationVectors"] >> calib_par.tvecs;
            stereo_fs["PerViewErrors"]              >> calib_par.perViewErrors;
            stereo_fs["RMS"]                        >> calib_par.RMS;
            stereo_fs.release();
          }
    } else {

    }

    // Перевод изображений из цветного формата в монохромный
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
                                cv::Size(imageL.cols, imageL.rows), CV_32FC1, mapRx, mapRy);

    cv::Mat rectifiedLeft, rectifiedRight;

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

    // Поиск 3д точек
    cv::reprojectImageTo3D(disparity, points23D, Q, true, CV_32F);

    std::vector<double> limit_outlierArea {-8.0e3, -8.0e3, 250, 8.0e3, 8.0e3, 15.20e3};

    //std::vector<std::vector<int>> vu;             // 2D координаты точки на изображении
    //std::vector<std::vector<double>> xyz;         // 3D координаты точки на пространсве
    //std::vector<std::vector<int>> rgb;            // цвет 3D точки


    // Запись координат найденных точек в файл
    std::string filename = "../../Output/3d_points.txt";
    //cv::FileStorage xyz_fs;
    std::ofstream xyz_fs;

    xyz_fs.open(filename);

    if (xyz_fs.is_open()) {
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

                //vu.push_back({v, u});
                //xyz.push_back(std::vector<double> ({xyz3D[0], xyz3D[1], xyz3D[2]}));


                xyz_fs << v <<"\t"<< u << "\t" << " \t" << xyz3D[0] << "\t" << xyz3D[1] << "\t" << xyz3D[2] << std::endl;
            }
        }

        xyz_fs.close();
        std::cout << "File " << filename << " was created." << std::endl;

    } else {
        std::cerr << "Error while reading the " << filename << "." << std::endl;
    }
}


cv::Vec3f third_coords(cv::Mat imageL, cv::Mat imageR, cv::Point xy, cv::Mat coords3d) {

    stereo_output_par_t calib_par;
    cv::FileStorage stereo_fs;
    if (stereo_fs.open("../../Calibration_parameters(stereo)/A6_stereo_camera_parameters.yml", cv::FileStorage::READ)){
        if (stereo_fs.isOpened()){
            stereo_fs["cameraMatrixL"]              >> calib_par.cameraM1;
            stereo_fs["cameraMatrixR"]              >> calib_par.cameraM2;
            stereo_fs["DistorsionCoeffsL"]          >> calib_par.distCoeffs1;
            stereo_fs["DistorsionCoeffsR"]          >> calib_par.distCoeffs2;
            stereo_fs["RotationMatrix"]             >> calib_par.R;
            stereo_fs["TranslationMatrix"]          >> calib_par.T;
            stereo_fs["EssentialMatrix"]            >> calib_par.E;
            stereo_fs["FundamentalMatrix"]          >> calib_par.F;
            stereo_fs["VectorOfRotationVectors"]    >> calib_par.rvecs;
            stereo_fs["VectorOfTranslationVectors"] >> calib_par.tvecs;
            stereo_fs["PerViewErrors"]              >> calib_par.perViewErrors;
            stereo_fs["RMS"]                        >> calib_par.RMS;
            stereo_fs.release();
          }
    } else {

    }

    // Перевод изображений из цветного формата в монохромный
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
                                cv::Size(imageL.cols, imageL.rows), CV_32FC1, mapRx, mapRy);

    cv::Mat rectifiedLeft, rectifiedRight;

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

    // Поиск 3д точек
    cv::reprojectImageTo3D(disparity, coords3d, Q, true, CV_32F);

    if (xy.x >= 0 && xy.x < coords3d.cols && xy.y >= 0 && xy.y < coords3d.rows) {
        cv::Vec3f point3D = coords3d.at<cv::Vec3f>(xy);
        if (point3D[2] != 0 && point3D[2] < 10000) {
            return point3D;
        }
    }

    return cv::Vec3f(0, 0, 0);
}
