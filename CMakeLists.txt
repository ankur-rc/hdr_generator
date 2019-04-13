cmake_minimum_required(VERSION 3.14.0)
project(hdr VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wpedantic -Wextra")

# include directories
include_directories("${PROJECT_SOURCE_DIR}/include")
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")

# Boost libraries
set(BOOST_ROOT "/usr/local")
set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include/boost")
set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
set(Boost_NO_SYSTEM_PATHS TRUE)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_DEBUG TRUE)
# set(boost_PATH_SUFFIXES "")
set(Boost_DETAILED_FAILURE_MSG TRUE)

find_package(Boost 1.69.0 EXACT REQUIRED COMPONENTS system filesystem)
if (Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")  
    message(STATUS "Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")     
    message(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")     
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")    
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")    
endif()

include_directories(BEFORE ${BOOST_INCLUDE_DIRS})
add_executable(hdr ${SOURCES})
target_link_libraries(hdr Boost::filesystem Boost::system)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

