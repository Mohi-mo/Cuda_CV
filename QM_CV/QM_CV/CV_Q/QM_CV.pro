QT += core
QT -= gui

CONFIG += c++17
CONFIG =+ console
CONFIG -= app_bundle

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS = \
    calibrate.h \
    disparity.h \
    three_dimensional_proc.h

SOURCES = \
    calibrate.cpp \
    disparity.cpp \
    main.cpp \
    three_dimensional_proc.cpp


TEMPLATE = subdirs

SUBDIRS += Examples/calib_example \
    Examples/point3dfinder_example \
    Examples/points2dTo3d_example


win32 {

	#INCLUDEPATH += E:\OpenCV\install\include
	#DEPENDPATH += E:\OpenCV\install\include

	#LIBS += -LE:\OpenCV\opencv\build\bin\Debug\libopencv_core480d.dll
	#LIBS += E:\OpenCV\opencv\build\bin\Debug\opencv_highgui480d.dll
	#LIBS += E:\OpenCV\opencv\build\bin\Debug\opencv_imgcodecs480d.dll
	#LIBS += E:\OpenCV\opencv\build\bin\Debug\opencv_features2d480d.dll
	#LIBS += E:\OpenCV\opencv\build\bin\Debug\opencv_calib3d480d.dll
	#LIBS += E:\OpenCV\opencv\build\bin\Debug\opencv_videoio480d.dll
	#LIBS += E:\OpenCV\opencv\build\bin\Debug\opencv_imgproc480d.dll

	INCLUDEPATH += E:\OpenCV\OpenCV-Build\install\include

	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_core455.dll
	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_highgui455.dll
	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_imgcodecs455.dll
	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_features2d455.dll
	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_calib3d455.dll
	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_videoio455.dll
	LIBS += E:\OpenCV\OpenCV-Build\bin\libopencv_imgproc455.dll

}
unix {

	# Каталог включаемых файлов
	INCLUDEPATH += $$PWD/. \
	INCLUDEPATH += /usr/local/include/opencv4/

	# Подключение библиотек OpenCV
	LIBS += -L/usr/local/lib -lopencv_world


	#INCLUDEPATH += /usr/include/opencv4
	#LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d -lopencv_videoio -lopencv_imgproc
}


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
