/*
 * Created Date: Saturday April 13th 2019
 * Last Modified: Saturday April 13th 2019 12:56:03 am
 * Author: ankurrc
 */

#pragma once

#include <vector>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <map>

namespace hdr
{
namespace utils
{

// pixel intensity range
const uint8_t Z_min = 0, Z_max = 255;

// random number seeding
static std::random_device rd;      // obtain a random number from hardware
static std::mt19937 rnd_eng(rd()); // seed the generator

// name-channel map
const std::map<std::string, uint> CHANNELS = {
    {"blue", 0}, {"green", 1}, {"red", 2}};

// get all the image paths in the directory
std::vector<boost::filesystem::path> get_paths_in_directory(const boost::filesystem::path &dir) noexcept(false);

// get crf for camera calibration
std::vector<std::vector<float>> calibrate(const std::vector<boost::filesystem::path> &paths, const float& lambda);

// show loaded image
void show_image(const cv::Mat &img);

// calculate exposure time from the path name
float get_exposure_time(const boost::filesystem::path &imgPath);

// get random samples of indices(x, y) for extracting a random pixel
std::vector<std::pair<uint, uint>> get_random_indices(const int &num_rows, const int &num_cols, const int &samples);

// hat function
uint hat(const uint &pixel);

// save calibration into file
bool save_crf(const std::vector<std::vector<float>> &crfs, boost::filesystem::path &save_path);

// load calibration back into vector
bool load_crf(std::vector<std::vector<float>> &crfs, boost::filesystem::path &load_path);

// plot crf for different channels
void plot_crf(const std::vector<std::vector<float>> &crf, const std::vector<std::string> &names);

// generate the radiance map
void generate_hdr(const std::vector<std::vector<float>> &crf, boost::filesystem::path &image_dir);
} // namespace utils
} // namespace hdr