#include "../../three_dimensional_proc.h"


cv::Mat imageLeft, imageRight;


// Функция обратной связи для обработки кликов мыши
void onClick(int event, int x, int y, int flags, void *userdata){
    stereo_output_par_t* calib_par  = static_cast<stereo_output_par_t*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN){
        cv::Point clickPoint(x,y);
        std::cout << third_coords(imageLeft, imageRight, clickPoint, *calib_par) << std::endl;
    }
}


int main() {
    // Загрузка левого и правого изображений
    imageLeft = cv::imread("../../../../Fotoset/FEDOR/L_27_4_37.jpg");
    imageRight = cv::imread("../../../../Fotoset/FEDOR/R_27_4_37.jpg");

    // Чтение параметров калибровки
    std::string file_name = "../../test_calib.yml";
    stereo_output_par_t calib_par = read_stereo_params(file_name);

    // Вывод левого изображения и регистрация кликов с помощью функции обратной связи
    imshow("Test", imageLeft);
    cv::setMouseCallback("Test", onClick, &calib_par);

    // Закрытие всех активных окон по нажатию любой клавиши
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
