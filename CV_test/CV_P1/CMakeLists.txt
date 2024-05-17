cmake_minimum_required(VERSION 3.5)

project(CV_P1 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(CV_P1 disparity.cpp calibrate.cpp main.cpp) #triangulate.cpp

include_directories($CMAKE_SOURCE_DIR)

find_package(OpenCV REQUIRED)
target_link_libraries(CV_P1 ${OpenCV_LIBS})

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
target_link_libraries(CV_P1 Qt5::Widgets Qt5::Core Qt5::Gui)
