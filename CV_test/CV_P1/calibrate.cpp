#include "calibrate.h"


void calibrate_with_mono(std::vector<cv::String> imagesL,std::vector<cv::String> imagesR,
                         std::string pathL,std::string pathR, std::string dataset_name,
                         int checkerboard_c, int checkerboard_r,
                         mono_output_par_t &mono_outL,mono_output_par_t &mono_outR, stereo_output_par_t &st_out)
{
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpointsL, imgpointsR;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<checkerboard_r; i++)
    {
        for(int j{0}; j<checkerboard_c; j++)
            objp.push_back(cv::Point3f(j,i,0));
    }

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

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true
        successL = cv::findChessboardCorners(grayL, cv::Size(checkerboard_c, checkerboard_r),
                                             corner_ptsL, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );
        successR = cv::findChessboardCorners(grayR, cv::Size(checkerboard_c, checkerboard_r),
                                             corner_ptsR, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

      if(successL && successR)
      {
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

        // refining pixel coordinates for given 2d points.
        cv::cornerSubPix(grayL,corner_ptsL,cv::Size(11,11), cv::Size(-1,-1),criteria);
        cv::cornerSubPix(grayR,corner_ptsR,cv::Size(11,11), cv::Size(-1,-1),criteria);

        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frameL, cv::Size(checkerboard_c, checkerboard_r), corner_ptsL, successL);
        cv::drawChessboardCorners(frameR, cv::Size(checkerboard_c, checkerboard_r), corner_ptsR, successL);

        objpoints.push_back(objp);
        imgpointsL.push_back(corner_ptsL);
        imgpointsR.push_back(corner_ptsR);
      }
      cv::imshow("Left calib image", frameL);
      cv::imshow("Right calib image", frameR);
    }

    //cv::destroyAllWindows();
    cv::TermCriteria criteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON);

    mono_outL.RMS = cv::calibrateCamera(objpoints, imgpointsL, cv::Size(grayL.rows,grayL.cols),
                                mono_outL.cameraMatrix, mono_outL.distCoeffs, mono_outL.rvecs, mono_outL.tvecs,
                                mono_outL.stdDevIntrinsics, mono_outL.stdDevExtrinsics, mono_outL.perViewErrors, 0, criteria);

    //mono_outL.cameraMatrix = cv::getOptimalNewCameraMatrix(mono_outL.cameraMatrix, mono_outL.distCoeffs, cv::Size(grayL.rows, grayL.cols),
    //                             1, grayL.size(),0);

    std::string filenameL = "../../Calibration_parameters(mono)/A" + dataset_name + "_left_" +"camera_parameters.yml";
    cv::FileStorage fs;

    fs.open(filenameL, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "cameraMatrix" << mono_outL.cameraMatrix;
        fs << "distCoeffs" << mono_outL.distCoeffs;
        fs << "PerViewErrors" << mono_outL.perViewErrors;
        fs << "STDIntrinsics" << mono_outL.stdDevIntrinsics;
        fs << "STDExtrinsics" << mono_outL.stdDevExtrinsics;
        fs << "RotationVector" << mono_outL.rvecs;
        fs << "TranslationVector" << mono_outL.tvecs;
        fs << "RMS" << mono_outL.RMS;
        fs.release();
        std::cout << "Файл " << filenameL << " был создан и записан." << std::endl;
    } else {
        std::cerr << "Ошибка при открытии файла " << filenameL << " для записи." << std::endl;
    }


    mono_outR.RMS = cv::calibrateCamera(objpoints, imgpointsR, cv::Size(grayL.rows,grayL.cols),
                                mono_outR.cameraMatrix, mono_outR.distCoeffs, mono_outR.rvecs, mono_outR.tvecs,
                                mono_outR.stdDevIntrinsics, mono_outR.stdDevExtrinsics, mono_outR.perViewErrors, 0, criteria);

    //mono_outR.cameraMatrix = cv::getOptimalNewCameraMatrix(mono_outR.cameraMatrix, mono_outR.distCoeffs, cv::Size(grayL.rows, grayL.cols),
    //                              1, grayL.size(),0);


    std::string filenameR = "../../Calibration_parameters(mono)/A" + dataset_name + "_right_" +"camera_parameters.yml";
    cv::FileStorage fs_0;

    fs.open(filenameR, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs_0 << "cameraMatrix" << mono_outR.cameraMatrix;
        fs_0 << "distCoeffs" << mono_outR.distCoeffs;
        fs_0 << "PerViewErrors" << mono_outR.perViewErrors;
        fs_0 << "STDIntrinsics" << mono_outR.stdDevIntrinsics;
        fs_0 << "STDExtrinsics" << mono_outR.stdDevExtrinsics;
        fs_0 << "RotationVector" << mono_outR.rvecs;
        fs_0 << "TranslationVector" << mono_outR.tvecs;
        fs_0 << "RMS" << mono_outR.RMS;
        fs_0.release();
        std::cout << "Файл " << filenameR << " был создан и записан." << std::endl;
    } else {
        std::cerr << "Ошибка при открытии файла " << filenameR << " для записи." << std::endl;
    }


    st_out.RMS = cv::stereoCalibrate(objpoints, imgpointsL, imgpointsR, mono_outL.cameraMatrix, st_out.distCoeffs1,
                        mono_outR.cameraMatrix, st_out.distCoeffs2, cv::Size(grayL.rows,grayL.cols), st_out.R, st_out.T,
                        st_out.E, st_out.F, st_out.rvecs, st_out.tvecs, st_out.perViewErrors,
                        cv::CALIB_FIX_INTRINSIC, criteria);

    // Writing updates
    st_out.cameraM1 = mono_outL.cameraMatrix;
    st_out.distCoeffs1 = mono_outL.distCoeffs;
    st_out.cameraM2 = mono_outR.cameraMatrix;
    st_out.distCoeffs2 = mono_outR.distCoeffs;

    std::string filename = "../../Calibration_parameters(stereo)/A" + dataset_name +"_stereo_camera_parameters.yml";
    cv::FileStorage stereo_fs;

    stereo_fs.open(filename, cv::FileStorage::WRITE);
    if (stereo_fs.isOpened()) {
        stereo_fs << "cameraMatrixL"              << st_out.cameraM1;
        stereo_fs << "cameraMatrixR"              << st_out.cameraM2;
        stereo_fs << "DistorsionCoeffsL"          << st_out.distCoeffs1;
        stereo_fs << "DistorsionCoeffsR"          << st_out.distCoeffs2;
        stereo_fs << "RotationMatrix"             << st_out.R;
        stereo_fs << "TranslationMatrix"          << st_out.T;
        stereo_fs << "EssentialMatrix"            << st_out.E;
        stereo_fs << "FundamentalMatrix"          << st_out.F;
        stereo_fs << "VectorOfRotationVectors"    << st_out.rvecs;
        stereo_fs << "VectorOfTranslationVectors" << st_out.tvecs;
        stereo_fs << "PerViewErrors"              << st_out.perViewErrors;
        stereo_fs << "RMS"                        << st_out.RMS;
        stereo_fs.release();
        std::cout << "File " << filename << " was created." << std::endl;
    } else {
        std::cerr << "Error while reading the " << filename << "." << std::endl;
    }
}


void calibrate_camera(std::vector<cv::String> images, std::string path, std::string dataset_name,
                      int checkerboard_c, int checkerboard_r, mono_output_par_t &mono_out){

    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<checkerboard_r; i++)
    {
        for(int j{0}; j<checkerboard_c; j++)
            objp.push_back(cv::Point3f(j,i,0));
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
        success = cv::findChessboardCorners(gray, cv::Size(checkerboard_c, checkerboard_r), corner_pts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

      /*
       * If desired number of corner are detected,
       * we refine the pixel coordinates and display
       * them on the images of checker board
      */

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

    //cv::destroyAllWindows();
    cv::TermCriteria criteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON);

    mono_out.RMS = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols),
                                mono_out.cameraMatrix, mono_out.distCoeffs, mono_out.rvecs, mono_out.tvecs,
                                mono_out.stdDevIntrinsics, mono_out.stdDevExtrinsics, mono_out.perViewErrors, 0, criteria);

    mono_out.cameraMatrix = cv::getOptimalNewCameraMatrix(mono_out.cameraMatrix, mono_out.distCoeffs, cv::Size(gray.rows, gray.cols),
                                  1, gray.size(),0);

    std::string filename = dataset_name + "_" +"camera_parameters.yml";
    cv::FileStorage fs;

    fs.open(filename, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "cameraMatrix" << mono_out.cameraMatrix;
        fs << "distCoeffs" << mono_out.distCoeffs;
        fs << "PerViewErrors" << mono_out.perViewErrors;
        fs << "STDIntrinsics" << mono_out.stdDevIntrinsics;
        fs << "STDExtrinsics" << mono_out.stdDevExtrinsics;
        fs << "RotationVector" << mono_out.rvecs;
        fs << "TranslationVector" << mono_out.tvecs;
        fs << "RMS" << mono_out.RMS;
        fs.release();
        std::cout << "Файл " << filename << " был создан и записан." << std::endl;
    } else {
        std::cerr << "Ошибка при открытии файла " << filename << " для записи." << std::endl;
    }
}

void calibrate_stereo(cv::Mat newCameraML, cv::Mat newCameraMR,
                      std::vector<cv::String> im1, std::vector<cv::String> im2,
                      std::string path1, std::string path2,
                      std::string dataset_name, int checkerboard_c, int checkerboard_r,
                      stereo_output_par_t &outp_params){

    std::vector<std::vector<cv::Point3f> > objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints_left, imgpoints_right;

    glob(path1, im1);
    glob(path2, im2);

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    std::vector<cv::Point3f> objp;
    for(int i{0}; i< checkerboard_r; i++){
        for(int j{0}; j<checkerboard_c; j++)
            objp.push_back(cv::Point3f(j,i,0));
    }

    std::vector<cv::Mat> c1_images, c2_images;

    for(int i{0}; i<im1.size(); i++){
        cv::Mat im_1 = cv::imread(im1[i], 1);
        c1_images.push_back(im_1);

        cv::Mat im_2  = cv::imread(im2[i], 1);
        c2_images.push_back(im_2);
    }


    for (size_t i = 0; i < im1.size(); i++) {
        cv::Mat gray1, gray2;
        cv::cvtColor(c1_images[i], gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(c2_images[i], gray2, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners1, corners2;
        bool c_ret1 = cv::findChessboardCorners(gray1, cv::Size(checkerboard_c, checkerboard_r), corners1);
        bool c_ret2 = cv::findChessboardCorners(gray2, cv::Size(checkerboard_c, checkerboard_r), corners2);

        if (c_ret1 && c_ret2) {
            cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::cornerSubPix(gray2, corners2, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            cv::drawChessboardCorners(c1_images[i], cv::Size(checkerboard_c, checkerboard_r), corners1, c_ret1);
            cv::imshow("img", c1_images[i]);

            cv::drawChessboardCorners(c2_images[i], cv::Size(checkerboard_c, checkerboard_r), corners2, c_ret2);
            cv::imshow("img2", c2_images[i]);
            //cv::waitKey(0);

            objpoints.push_back(objp);
            imgpoints_left.push_back(corners1);
            imgpoints_right.push_back(corners2);
        }
    }

    outp_params.RMS = cv::stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                          newCameraML,outp_params.distCoeffs1, newCameraMR, outp_params.distCoeffs2,
                                          c1_images[0].size(), outp_params.R, outp_params.T,
                                          outp_params.E, outp_params.F, outp_params.rvecs,
                                          outp_params.tvecs, outp_params.perViewErrors,
                                          0 | CALIB_FIX_INTRINSIC, criteria);

    outp_params.cameraM1 = newCameraML;
    outp_params.cameraM2 = newCameraMR;


    std::string filename = dataset_name +"_stereo_camera_parameters.yml";
    cv::FileStorage stereo_fs;

    stereo_fs.open(filename, cv::FileStorage::WRITE);
    if (stereo_fs.isOpened()) {
        stereo_fs << "cameraMatrixL"              << outp_params.cameraM1;
        stereo_fs << "cameraMatrixR"              << outp_params.cameraM2;
        stereo_fs << "DistorsionCoeffsL"          << outp_params.distCoeffs1;
        stereo_fs << "DistorsionCoeffsR"          << outp_params.distCoeffs2;
        stereo_fs << "RotationMatrix"             << outp_params.R;
        stereo_fs << "TranslationMatrix"          << outp_params.T;
        stereo_fs << "EssentialMatrix"            << outp_params.E;
        stereo_fs << "FundamentalMatrix"          << outp_params.F;
        stereo_fs << "VectorOfRotationVectors"    << outp_params.rvecs;
        stereo_fs << "VectorOfTranslationVectors" << outp_params.tvecs;
        stereo_fs << "PerViewErrors"              << outp_params.perViewErrors;
        stereo_fs << "RMS"                        << outp_params.RMS;
        stereo_fs.release();
        std::cout << "File " << filename << " was created." << std::endl;
    } else {
        std::cerr << "Error while reading the " << filename << "." << std::endl;
    }
}

// Отображение параметров камеры
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

// Отображение параметров стерео камеры
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

/*
// Предзагрузка параметров калибровки камер
cv::FileStorage fs;
if (fs.open(name + "_left_camera_parameters.yml", cv::FileStorage::READ) && (!calibrate)){
    if (fs.isOpened()){
        fs["cameraMatrix"] >> mono_parL.cameraMatrix;
        fs["distCoeffs"] >> mono_parL.distCoeffs;
        fs["PerViewErrors"] >> mono_parL.perViewErrors;
        fs["STDIntrinsics"] >> mono_parL.stdDevIntrinsics;
        fs["STDExtrinsics"] >> mono_parL.stdDevExtrinsics;
        fs["RotationVector"] >> mono_parL.rvecs;
        fs["TranslationVector"] >> mono_parL.tvecs;
        fs["RMS"] >> mono_parL.RMS;
        fs.release();
      }
  } else {
    cout << "Left calibration procedure is running..." << endl;
    calibrate_camera(imagesL, pathL, name+ "_left", checkerboard_c,checkerboard_r, mono_parL);
}

if (fs.open(name + "_right_camera_parameters.yml", cv::FileStorage::READ) && (!calibrate)){
    if (fs.isOpened()){
      fs["cameraMatrix"] >> mono_parR.cameraMatrix;
      fs["distCoeffs"] >> mono_parR.distCoeffs;
      fs["PerViewErrors"] >> mono_parR.perViewErrors;
      fs["STDIntrinsics"] >> mono_parR.stdDevIntrinsics;
      fs["STDExtrinsics"] >> mono_parR.stdDevExtrinsics;
      fs["RotationVector"] >> mono_parR.rvecs;
      fs["TranslationVector"] >> mono_parR.tvecs;
      fs["RMS"] >> mono_parR.RMS;
      fs.release();
    }
} else {
    cout << "Right calibration procedure is running..." << endl;
    calibrate_camera(imagesR, pathR, name + "_right", checkerboard_c,checkerboard_r, mono_parR);
}

// Show cameras parameters
print_mono_camera_parameters("Left_camera", mono_parL);
print_mono_camera_parameters("Right_camera", mono_parR);
*/
