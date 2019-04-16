cmake_minimum_required(VERSION 3.14.0)
project(hdr VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wpedantic -Wextra")

# install prefix
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX} /usr/local")

# add the executable
add_executable(hdr "main.cpp" "utils.cpp")

# include directories
target_include_directories(hdr PRIVATE "${PROJECT_SOURCE_DIR}/include")

# Boost
set(BOOST_ROOT "/usr/local")
# set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include/boost")
# set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
set(Boost_NO_SYSTEM_PATHS TRUE)
set(Boost_USE_MULTITHREADED OFF)
# set(Boost_DEBUG TRUE)
set(Boost_DETAILED_FAILURE_MSG TRUE)

find_package(Boost 1.69.0 EXACT REQUIRED COMPONENTS system filesystem)
if (Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})  
    message(STATUS "Boost_INCLUDE_DIR:  " ${Boost_INCLUDE_DIR})     
    message(STATUS "Boost_LIBRARY_DIRS: " ${Boost_LIBRARY_DIRS})     
    message(STATUS "Boost_LIBRARIES:    " ${Boost_LIBRARIES})    
    message(STATUS "Boost_VERSION:      " ${Boost_VERSION})    
endif()
target_include_directories(hdr PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(hdr ${Boost_LIBRARIES})

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(hdr PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hdr ${OpenCV_LIBS})
message(STATUS "OpenCV_INCLUDE_DIRS:    " ${OpenCV_INCLUDE_DIRS})
message(STATUS "Opencv_LIBS:            " ${OpenCV_LIBS})


# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULES)
target_link_libraries(hdr Eigen3::Eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

