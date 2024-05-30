#include "../../three_dimensional_proc.h"


cv::Mat rectifiedLeft, rectifiedRight;


// Функция обратной связи для обработки кликов мыши
void onClick(int event, int x, int y, int flags, void *userdata){
    cv::Mat* Q = static_cast<cv::Mat*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN){
        cv::Point clickPoint(x,y);
        std::cout << third_coords(rectifiedLeft, rectifiedRight, clickPoint, *Q) << std::endl;
    }
}


int main() {
    // Загрузка левого и правого изображений
    cv::Mat imageLeft = cv::imread("../../../../Fotoset/FEDOR/L_27_4_37.jpg");
    cv::Mat imageRight = cv::imread("../../../../Fotoset/FEDOR/R_27_4_37.jpg");

    // Чтение параметров калибровки
    std::string file_name = "../../test_calib.yml";
    stereo_output_par_t calib_par = read_stereo_params(file_name);

    // Перевод изображений из цветного формата в монохромный
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(imageLeft, grayImageLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRight, grayImageRight, cv::COLOR_BGR2GRAY);

    // Стереоректификация изображений
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Mat Q, R1, R2, P1, P2;

    cv::stereoRectify(calib_par.cameraM1, calib_par.distCoeffs1, calib_par.cameraM2, calib_par.distCoeffs2,
                      cv::Size(grayImageLeft.cols, grayImageLeft.rows), calib_par.R, calib_par.T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY);

    cv::initUndistortRectifyMap(calib_par.cameraM1, calib_par.distCoeffs1, R1, P1,
                                cv::Size(imageLeft.cols, imageLeft.rows), CV_32FC1, mapLx, mapLy);

    cv::initUndistortRectifyMap(calib_par.cameraM2, calib_par.distCoeffs2, R2, P2,
                                cv::Size(imageRight.cols, imageRight.rows), CV_32FC1, mapRx, mapRy);


    // Ректификация и устранение искажений
    cv::remap(grayImageLeft, rectifiedLeft, mapLx, mapLy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(grayImageRight, rectifiedRight, mapRx, mapRy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // Вывод левого изображения и регистрация кликов с помощью функции обратной связи
    imshow("Test", rectifiedLeft);
    cv::setMouseCallback("Test", onClick, &Q);

    // Закрытие всех активных окон по нажатию любой клавиши
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
