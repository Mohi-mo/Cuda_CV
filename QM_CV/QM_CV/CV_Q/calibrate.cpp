#include "calibrate.h"


/// Функция общей калибровки
void calibrate_with_mono(std::vector<cv::String> imagesL,std::vector<cv::String> imagesR, std::string pathL,std::string pathR,
                         mono_output_par_t &mono_outL,mono_output_par_t &mono_outR, stereo_output_par_t &st_out,
                         int checkerboard_c, int checkerboard_r, float square_size)
{

    // Объявление вектора ключевых точек
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Объявление вектора для хранения координат 2д точек для каждого изображения шахматной доски
    std::vector<std::vector<cv::Point2f> > imgpointsL, imgpointsR;

    // Определение координат мировых 3д точек
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<checkerboard_r; i++)
    {
        for(int j{0}; j<checkerboard_c; j++)
            objp.push_back(cv::Point3f((float)j*square_size,(float)i*square_size,0.f));
    }

    // Загрузка путей изображений
    cv::glob(pathL, imagesL);
    cv::glob(pathR, imagesR);

    cv::Mat frameL, grayL, frameR, grayR;
    std::vector<cv::Point2f> corner_ptsL, corner_ptsR;

    bool successL, successR;

    for(int i{0}; i<imagesL.size(); i++)
    {
        frameL = cv::imread(imagesL[i]);
        cv::cvtColor(frameL,grayL,cv::COLOR_BGR2GRAY);

        frameR = cv::imread(imagesR[i]);
        cv::cvtColor(frameR,grayR,cv::COLOR_BGR2GRAY);

        // Нахождение углов шахматной доски
        // Если на изображении найдено нужное число углов success = true
        successL = cv::findChessboardCorners(grayL, cv::Size(checkerboard_c, checkerboard_r),
                                             corner_ptsL, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FILTER_QUADS);
        successR = cv::findChessboardCorners(grayR, cv::Size(checkerboard_c, checkerboard_r),
                                             corner_ptsR, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FILTER_QUADS);

        if(successL && successR)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1e-6);

            // Уточнение координат пикселей для заданных двумерных точек
            cv::cornerSubPix(grayL,corner_ptsL,cv::Size(11,11), cv::Size(-1,-1),criteria);
            cv::cornerSubPix(grayR,corner_ptsR,cv::Size(11,11), cv::Size(-1,-1),criteria);

            // Отображение обнаруженных угловых точек на шахматной доске
            cv::drawChessboardCorners(frameL, cv::Size(checkerboard_c, checkerboard_r), corner_ptsL, successL);
            cv::drawChessboardCorners(frameR, cv::Size(checkerboard_c, checkerboard_r), corner_ptsR, successL);

            objpoints.push_back(objp);
            imgpointsL.push_back(corner_ptsL);
            imgpointsR.push_back(corner_ptsR);

        }

        // Отображение последних кадров с отмеченными точками
        //cv::imshow("Left calib image", frameL);
        //cv::imshow("Right calib image", frameR);
        //cv::waitKey(0);
    }

    cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 10, DBL_EPSILON);
    int flags = 0;

    // Калибровка левой камеры
    mono_outL.RMS = cv::calibrateCamera(objpoints, imgpointsL, cv::Size(grayL.cols,grayL.rows),
                                        mono_outL.cameraMatrix, mono_outL.distCoeffs, mono_outL.rvecs, mono_outL.tvecs,
                                        mono_outL.stdDevIntrinsics, mono_outL.stdDevExtrinsics, mono_outL.perViewErrors, flags, criteria);

    // Калибровка правой камеры
    mono_outR.RMS = cv::calibrateCamera(objpoints, imgpointsR, cv::Size(grayL.cols,grayL.rows),
                                        mono_outR.cameraMatrix, mono_outR.distCoeffs, mono_outR.rvecs, mono_outR.tvecs,
                                        mono_outR.stdDevIntrinsics, mono_outR.stdDevExtrinsics, mono_outR.perViewErrors, flags, criteria);

    // Калибровка двух камер

    st_out.RMS = cv::stereoCalibrate(objpoints, imgpointsL, imgpointsR, st_out.cameraM1, st_out.distCoeffs1,
                                     st_out.cameraM2, st_out.distCoeffs2, cv::Size(grayL.cols,grayL.rows), st_out.R, st_out.T,
                                     st_out.E, st_out.F, st_out.perViewErrors, 0, criteria);
}


/// Функция калибровки одной камеры
void calibrate_camera(std::vector<cv::String> images, std::string path, mono_output_par_t &mono_out,
                      int checkerboard_c, int checkerboard_r, float square_size){

    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<checkerboard_r; i++)
    {
        for(int j{0}; j<checkerboard_c; j++)
            objp.push_back(cv::Point3f(j*square_size,i*square_size,0));
    }

    cv::glob(path, images);

    cv::Mat frame, gray;
    std::vector<cv::Point2f> corner_pts;

    bool success;

    // Looping over all the images in the directory
    for(int i{0}; i<images.size(); i++)
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true
        success = cv::findChessboardCorners(gray, cv::Size(checkerboard_c, checkerboard_r), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FILTER_QUADS);

        if(success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frame, cv::Size(checkerboard_c, checkerboard_r), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
    }

    cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, DBL_EPSILON);

    mono_out.RMS = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols),
                                       mono_out.cameraMatrix, mono_out.distCoeffs, mono_out.rvecs, mono_out.tvecs,
                                       mono_out.stdDevIntrinsics, mono_out.stdDevExtrinsics, mono_out.perViewErrors, 0, criteria);
}


/// Функция калибровки стереокамеры
void calibrate_stereo(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathL, std::string pathR,
                      stereo_output_par_t &stereo_output_params, int checkerboard_c, int checkerboard_r, float square_size){

    std::vector<std::vector<cv::Point3f> > objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints_left, imgpoints_right;

    cv::glob(pathL, imagesL);
    cv::glob(pathR, imagesR);

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    std::vector<cv::Point3f> objp;
    for(int i{0}; i< checkerboard_r; i++){
        for(int j{0}; j<checkerboard_c; j++)
            objp.push_back(cv::Point3f(j*square_size,i*square_size,0));
    }

    std::vector<cv::Mat> c1_images, c2_images;

    for(int i{0}; i<imagesL.size(); i++){
        cv::Mat im_1 = cv::imread(imagesL[i], 1);
        c1_images.push_back(im_1);

        cv::Mat im_2  = cv::imread(imagesR[i], 1);
        c2_images.push_back(im_2);
    }


    cv::Mat grayL, grayR;
    for (size_t i = 0; i < imagesL.size(); i++) {
        cv::cvtColor(c1_images[i], grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(c2_images[i], grayR, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners1, corners2;
        bool c_ret1 = cv::findChessboardCorners(grayL, cv::Size(checkerboard_c, checkerboard_r), corners1, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FILTER_QUADS);
        bool c_ret2 = cv::findChessboardCorners(grayR, cv::Size(checkerboard_c, checkerboard_r), corners2, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FILTER_QUADS);

        if (c_ret1 && c_ret2) {
            cv::cornerSubPix(grayL, corners1, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::cornerSubPix(grayR, corners2, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            cv::drawChessboardCorners(c1_images[i], cv::Size(checkerboard_c, checkerboard_r), corners1, c_ret1);

            cv::drawChessboardCorners(c2_images[i], cv::Size(checkerboard_c, checkerboard_r), corners2, c_ret2);

            objpoints.push_back(objp);
            imgpoints_left.push_back(corners1);
            imgpoints_right.push_back(corners2);

            //cv::imshow("img", c1_images[i]);
            //cv::imshow("img2", c2_images[i]);
        }
    }
    /*
    double stereoCalibrate( InputArrayOfArrays objectPoints,
                           InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                           InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                           InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                           Size imageSize, InputOutputArray R,InputOutputArray T, OutputArray E, OutputArray F,
                           OutputArray perViewErrors, int flags = CALIB_FIX_INTRINSIC,
                           TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );
*/


    stereo_output_params.RMS = cv::stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, stereo_output_params.cameraM1, stereo_output_params.distCoeffs1,
                                                   stereo_output_params.cameraM2, stereo_output_params.distCoeffs2, cv::Size(grayL.cols,grayL.rows), stereo_output_params.R, stereo_output_params.T,
                                                   stereo_output_params.E, stereo_output_params.F, stereo_output_params.perViewErrors, 0, criteria);

}

// Функция отображения параметров камеры
void print_mono_camera_parameters(std::string name, mono_output_par_t mono_struct){
    cout << "\n\n\t" << name << "---------------------------------" << endl;
    cout << "cameraMatrix: "        << mono_struct.cameraMatrix     << endl;
    cout << "distCoeffs: "         << mono_struct.distCoeffs        << endl;
    //cout << "Per view errors: "     << mono_struct.perViewErrors    << endl;
    //cout << "STD Intrinsics: "      << mono_struct.stdDevIntrinsics << endl;
    //cout << "STD Extrinsics: "      << mono_struct.stdDevExtrinsics << endl;
    //cout << "Rotation vector: "     << mono_struct.rvecs            << endl;
    //cout << "Translation vector: "  << mono_struct.tvecs            << endl;
    cout << "RMS: "                 << mono_struct.RMS              << endl;
}

// Функция отображения параметров стерео камеры
void print_stereo_camera_parameters(stereo_output_par_t stereo_struct){
    cout << "\n\n\t\t Both cameras" << "------------------------------------" << endl;
    cout << "cameraMatrix L: "                << stereo_struct.cameraM1       << endl;
    cout << "cameraMatrix R: "                << stereo_struct.cameraM2       << endl;
    cout << "Distorsion coeffs L: "           << stereo_struct.distCoeffs1    << endl;
    cout << "Distorsion coeffs R: "           << stereo_struct.distCoeffs2    << endl;
    cout << "Rotation matrix: "               << stereo_struct.R              << endl;
    cout << "Translation matrix: "            << stereo_struct.T              << endl;
    cout << "Essential matrix: "              << stereo_struct.E              << endl;
    cout << "Fundamental matrix: "            << stereo_struct.F              << endl;
    //cout << "Vector of rotation vectors: "    << stereo_struct.rvecs          << endl;
    //cout << "Vector of translation vectors: " << stereo_struct.tvecs          << endl;
    //cout << "Per view errors: "               << stereo_struct.perViewErrors  << endl;
    cout << "RMS: "                           << stereo_struct.RMS            << endl;
}

/// Функция чтения параметров калибровки одной камеры из файлов
mono_output_par_t read_mono_params(std::string filename){

    mono_output_par_t mono_params_struct = {};

    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "cameraMatrix" << mono_params_struct.cameraMatrix;
        fs << "distCoeffs" << mono_params_struct.distCoeffs;
        fs << "PerViewErrors" << mono_params_struct.perViewErrors;
        fs << "STDIntrinsics" << mono_params_struct.stdDevIntrinsics;
        fs << "STDExtrinsics" << mono_params_struct.stdDevExtrinsics;
        fs << "RotationVector" << mono_params_struct.rvecs;
        fs << "TranslationVector" << mono_params_struct.tvecs;
        fs << "RMS" << mono_params_struct.RMS;
        fs.release();
    } else {
        std::cerr << "Ошибка при открытии файла " << filename << std::endl;
    }
    return mono_params_struct;
}

/// Функция чтения параметров калибровки двух камер из файлов
stereo_output_par_t read_stereo_params(std::string filename){

    stereo_output_par_t stereo_params_struct = {};

    cv::FileStorage stereo_fs;
    if (stereo_fs.open(filename, cv::FileStorage::READ)){
        if (stereo_fs.isOpened()){
            stereo_fs["cameraMatrixL"]              >> stereo_params_struct.cameraM1;
            stereo_fs["cameraMatrixR"]              >> stereo_params_struct.cameraM2;
            stereo_fs["DistorsionCoeffsL"]          >> stereo_params_struct.distCoeffs1;
            stereo_fs["DistorsionCoeffsR"]          >> stereo_params_struct.distCoeffs2;
            stereo_fs["RotationMatrix"]             >> stereo_params_struct.R;
            stereo_fs["TranslationMatrix"]          >> stereo_params_struct.T;
            stereo_fs["EssentialMatrix"]            >> stereo_params_struct.E;
            stereo_fs["FundamentalMatrix"]          >> stereo_params_struct.F;
            stereo_fs["VectorOfRotationVectors"]    >> stereo_params_struct.rvecs;
            stereo_fs["VectorOfTranslationVectors"] >> stereo_params_struct.tvecs;
            stereo_fs["PerViewErrors"]              >> stereo_params_struct.perViewErrors;
            stereo_fs["RMS"]                        >> stereo_params_struct.RMS;
            stereo_fs.release();
        } else {
            std::cerr << "Ошибка при открытии файла " << filename << std::endl;
        }
    }
    return stereo_params_struct;
}

/// Функция для записи параметров калибровки одной камеры
void write_mono_params(std::string filename, mono_output_par_t mono_params_struct){

    //auto data_direct = std::filesystem::create_directory("../../Calibration_parameters(mono)");

    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "cameraMatrix"        << mono_params_struct.cameraMatrix;
        fs << "distCoeffs"          << mono_params_struct.distCoeffs;
        fs << "PerViewErrors"       << mono_params_struct.perViewErrors;
        fs << "STDIntrinsics"       << mono_params_struct.stdDevIntrinsics;
        fs << "STDExtrinsics"       << mono_params_struct.stdDevExtrinsics;
        fs << "RotationVector"      << mono_params_struct.rvecs;
        fs << "TranslationVector"   << mono_params_struct.tvecs;
        fs << "RMS"                 << mono_params_struct.RMS;
        fs.release();
    } else {
        std::cerr << "Ошибка при открытии файла " << filename << " для записи." << std::endl;
    }
}

/// Функция для записи параметров калибровки двух камер
void write_stereo_params(std::string filename, stereo_output_par_t stereo_params_struct){

    cv::FileStorage stereo_fs;
    stereo_fs.open(filename, cv::FileStorage::WRITE);
    if (stereo_fs.isOpened()) {
        stereo_fs << "cameraMatrixL"              << stereo_params_struct.cameraM1;
        stereo_fs << "cameraMatrixR"              << stereo_params_struct.cameraM2;
        stereo_fs << "DistorsionCoeffsL"          << stereo_params_struct.distCoeffs1;
        stereo_fs << "DistorsionCoeffsR"          << stereo_params_struct.distCoeffs2;
        stereo_fs << "RotationMatrix"             << stereo_params_struct.R;
        stereo_fs << "TranslationMatrix"          << stereo_params_struct.T;
        stereo_fs << "EssentialMatrix"            << stereo_params_struct.E;
        stereo_fs << "FundamentalMatrix"          << stereo_params_struct.F;
        stereo_fs << "VectorOfRotationVectors"    << stereo_params_struct.rvecs;
        stereo_fs << "VectorOfTranslationVectors" << stereo_params_struct.tvecs;
        stereo_fs << "PerViewErrors"              << stereo_params_struct.perViewErrors;
        stereo_fs << "RMS"                        << stereo_params_struct.RMS;
        stereo_fs.release();
    } else {
        std::cerr << "Ошибка при открытии файла " << filename << "." << std::endl;
    }
}
