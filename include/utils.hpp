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

namespace hdr
{
namespace utils
{
using namespace boost::filesystem;
using namespace std;

// get all the image paths in the directory
vector<path> get_paths_in_directory(const path &dir) noexcept(false);

// solve for Ax=b
void solve_lls(const vector<path> &paths);

// show loaded image
void show_image(const cv::Mat &img);

// calculate exposure time from the path name
float get_exposure_time(const path &imgPath);

// get random samples of indices(x, y) for extracting a random pixel
vector<pair<uint, uint>> get_random_indices(const int &num_rows, const int &num_cols, const int &samples);

// hat function
uint hat(const uint &pixel);
} // namespace utils
} // namespace hdr