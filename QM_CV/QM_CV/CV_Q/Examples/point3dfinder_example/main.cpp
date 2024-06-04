#include "../../three_dimensional_proc.h"


int main(){

    std::string file_name = "../../test_calib.yml";
    stereo_output_par_t calib_par = read_stereo_params(file_name);

    cv::Mat imageL = cv::imread("../../../../Fotoset/FEDOR/L_27_4_37.jpg");
    cv::Mat imageR = cv::imread("../../../../Fotoset/FEDOR/R_27_4_37.jpg");

    // Поск 3д точек и сохранение их в формате x y 3d_x 3d_y 3d_z
    std::vector<std::vector<double>> coords3d = point3d_finder(imageL, imageR, calib_par);

    // Запись найденных точек в файл (опционально)
    write_coords_file(coords3d, "3d_points.txt");

    return 0;
}
