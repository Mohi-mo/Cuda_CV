#include "../../three_dimensional_proc.h"


int main(){

    std::string file_name = "../../test_calib.yml";
    stereo_output_par_t calib_par = read_stereo_params(file_name);

    cv::Mat imageL = cv::imread("../../../../Fotoset/FEDOR/L_27_4_37.jpg");
    cv::Mat imageR = cv::imread("../../../../Fotoset/FEDOR/R_27_4_37.jpg");

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

    // Поск 3д точек и сохранение их в формате x y 3d_x 3d_y 3d_z
    std::vector<std::vector<double>> coords3d = point3d_finder(rectifiedLeft, rectifiedRight, Q);

    // Запись найденных точек в файл (опционально)
    write_coords_file(coords3d, "3d_points.txt");

    return 0;
}
