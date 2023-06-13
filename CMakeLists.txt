cmake_minimum_required(VERSION 3.15)
project(Faster_RCNN)

set(CMAKE_CXX_STANDARD 11)

add_executable(Faster_RCNN scope_guard.hpp tf_utils.hpp tf_utils.cpp faster_rcnn.cpp)

# Add OpenCV
include_directories("/home/sang/LIB/opencv-extra-3.4.2/include/")
set(TARGET_LIBS
        "/home/sang/LIB/opencv-extra-3.4.2/lib/libopencv_core.so"
        "/home/sang/LIB/opencv-extra-3.4.2/lib/libopencv_imgproc.so"
        "/home/sang/LIB/opencv-extra-3.4.2/lib/libopencv_highgui.so"
        "/home/sang/LIB/opencv-extra-3.4.2/lib/libopencv_imgcodecs.so"
        "/home/sang/LIB/opencv-extra-3.4.2/lib/libopencv_dnn.so"
        )
target_link_libraries(${PROJECT_NAME} ${TARGET_LIBS})

# Tensorflow C api
include_directories("/home/sang/LIB/libtensorflow-cpu-linux-x86_64-1.15.0/include/")
set(TARGET_LIBS
        "/home/sang/LIB/libtensorflow-cpu-linux-x86_64-1.15.0/lib/libtensorflow.so"
        "/home/sang/LIB/libtensorflow-cpu-linux-x86_64-1.15.0/lib/libtensorflow_framework.so"
        )
target_link_libraries(${PROJECT_NAME} ${TARGET_LIBS})
