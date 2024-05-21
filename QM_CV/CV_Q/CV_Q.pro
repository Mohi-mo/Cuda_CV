# Стандарт C++
CONFIG += c++11

# Включаем автоматическую генерацию MOC-файлов
CONFIG += qtmoc

HEADERS = \
    calibrate.h \
    disparity.h \
    three_dimensional_proc.h

SOURCES = \
    calibrate.cpp \
    disparity.cpp \
    main.cpp \
    three_dimensional_proc.cpp

# Каталог включаемых файлов
INCLUDEPATH += $$PWD/. \
INCLUDEPATH += /usr/local/include/opencv4/

# Подключение библиотек OpenCV
LIBS += -L/usr/local/lib -lopencv_world

# Подключение библиотек Qt
QT = widgets core gui



# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
