#include "calibrate.h"

//int r = 4;
//int c = 7;
//int CHECKERBOARD[2]{c,r}; // 6 x 9 - число узлов вдоль столбцов и строк шахматной доски

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
      //msleep(1);
      //cv::imshow("Image",frame);
      //cv::waitKey(0);
    }

    //cv::destroyAllWindows();
    cv::TermCriteria criteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON);

    mono_out.RMS = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols),
                                mono_out.cameraMatrix, mono_out.distCoeffs, mono_out.rvecs, mono_out.tvecs,
                                mono_out.stdDevIntrinsics, mono_out.stdDevExtrinsics, mono_out.perViewErrors, 0, criteria);

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


void calibrate_stereo(std::vector<cv::String> im1, std::vector<cv::String> im2, std::string path1, std::string path2,
                      std::string dataset_name, int checkerboard_c, int checkerboard_r, stereo_output_par_t &outp_params){

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
            cv::waitKey(0);

            objpoints.push_back(objp);
            imgpoints_left.push_back(corners1);
            imgpoints_right.push_back(corners2);
        }
    }
/*
    outp_params.RMS = cv::stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                          outp_params.cameraM1, outp_params.distCoeffs1,
                                          outp_params.cameraM2, outp_params.distCoeffs2,
                                          c1_images[0].size(), outp_params.R, outp_params.T,
                                          outp_params.E, outp_params.F, outp_params.rvecs,
                                          outp_params.tvecs, outp_params.perViewErrors,
                                          CALIB_ZERO_DISPARITY, criteria);

    stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 Size imageSize, InputArray R, InputArray T,
                                 OutputArray R1, OutputArray R2,
                                 OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags = CALIB_ZERO_DISPARITY,
                                 double alpha = -1, Size newImageSize = Size(),
                                 CV_OUT Rect* validPixROI1 = 0, CV_OUT Rect* validPixROI2 = 0 );

*/
    outp_params.RMS = cv::stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                          outp_params.cameraM1, outp_params.distCoeffs1,
                                          outp_params.cameraM2, outp_params.distCoeffs2,
                                          c1_images[0].size(), outp_params.R, outp_params.T,
                                          outp_params.E, outp_params.F, outp_params.rvecs,
                                          outp_params.tvecs, outp_params.perViewErrors,
                                          CALIB_FIX_INTRINSIC, criteria);


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
