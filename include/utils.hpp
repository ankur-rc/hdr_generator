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
using namespace boost::filesystem;
using namespace std;

// pixel intensity range
const uint8_t Z_min = 0, Z_max = 255;

// random number seeding
static random_device rd;      // obtain a random number from hardware
static mt19937 rnd_eng(rd()); // seed the generator

// name-channel map
const map<string, uint> CHANNELS = {
    {"blue", 0}, {"green", 1}, {"red", 2}};

// channel-lambda map
const map<uint, float> LAMBDAS = {
    {0, 6.f}, {1, 4.f}, {2, 6.f}};

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