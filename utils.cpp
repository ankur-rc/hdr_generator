/*
 * Created Date: Friday April 12th 2019
 * Last Modified: Friday April 12th 2019 11:55:17 pm
 * Author: ankurrc
 */

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <math.h>
#include <random>
#include <utility>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include "utils.hpp"
#include "matplotlibcpp.h"

namespace hdr
{
namespace utils
{
using namespace boost::filesystem;
using namespace std;

vector<path> get_paths_in_directory(const path &dir) noexcept(false)
{
    path dirPath(dir);
    vector<path> imagePaths = {};
    cout << "Directory path is: " << dirPath.string() << "\n";
    if (exists(dirPath))
    {
        if (is_directory(dirPath))
        {
            for (const directory_entry &dirEntry : directory_iterator(dirPath))
            {

                if (is_regular_file(dirEntry))
                    imagePaths.emplace_back(dirEntry);
            }
        }
    }
    else
    {
        cout << dirPath << " does not exist.\n";
    }

    return imagePaths;
}

void solve_lls(const vector<path> &imagePaths)
{
    // no. images
    const uint num_images = imagePaths.size();
    // min. samples required
    const uint min_samples = static_cast<int>(ceilf((Z_max - Z_min + 1) / (static_cast<float>(num_images) - 1)));
    // samples per image
    // const uint N = 2 * static_cast<int>(ceilf(static_cast<float>(min_samples) / num_images));
    const uint N = 5 * min_samples;

    cout << "Minimum Number of samples required for " << num_images << " images: " << min_samples
         << "\nPer picture (min samples/P): " << N << "\n";

    const uint n = Z_max - Z_min + 1;

    // setup image dimensions
    // get the image
    cv::Mat init_image = cv::imread(imagePaths[0].string(), cv::IMREAD_COLOR);
    if (!init_image.data)
        throw "No image data for " + imagePaths[0].string();

    // sample the indices
    vector<pair<uint, uint>> rnd_indices = get_random_indices(init_image.rows, init_image.cols, N);

    vector<Eigen::MatrixXf> A(CHANNELS.size());
    vector<Eigen::VectorXf> b(CHANNELS.size());

    for (const auto &ch : CHANNELS)
    {
        // construct A, b for Ax=b;
        uint ch_num = ch.second;
        string ch_name = ch.first;
        A[ch_num] = Eigen::MatrixXf::Zero(N * num_images + n + 1, N + n);
        b[ch_num] = Eigen::VectorXf::Zero(A[ch_num].rows());

        cout << "A [" << ch_name << "]:" << A[ch_num].rows() << "x" << A[ch_num].cols() << ")\t"
             << "b [" << ch_name << "]:" << b[ch_num].rows() << "x" << b[ch_num].cols() << ")\n";
    }

    // add entries to A & b
    size_t k = 0;
    for (size_t j = 0; j < num_images; j++)
    {
        path imgPath = imagePaths[j];

        // get the exposure time
        float exposure_time = get_exposure_time(imgPath);
        // cout << "Exposure time: " << exposure_time << endl;

        // get the image
        cv::Mat image = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
        if (!image.data)
            throw "No image data for " + imgPath.string();
        //show_image(image);
        assert(init_image.size == image.size);

        // get the pixel values
        for (size_t i = 0; i < rnd_indices.size(); i++)
        {
            cv::Vec3b zij = image.at<cv::Vec3b>(rnd_indices[i].first, rnd_indices[i].second);

            for (const auto &ch : CHANNELS)
            {
                // construct A, b for Ax=b;
                uint ch_num = ch.second;
                string ch_name = ch.first;

                const uint pixel = zij[ch_num];
                const uint weight = hat(pixel);

                // for g(wij)
                A[ch_num](k, pixel) = static_cast<float>(weight);
                // for log(Ei)
                A[ch_num](k, n + i) = -static_cast<float>(weight);

                b[ch_num](k) = weight * log(exposure_time);
            }

            ++k;
        }
    }

    // fix the curve by setting middle value to 0
    for (const auto &ch : CHANNELS)
    {
        A[ch.second](k, n / 2) = 1.f;
    }

    ++k;

    // regularize; enforce smoothness
    for (size_t i = 0; i < n - 1; i++)
    {
        for (const auto &ch : CHANNELS)
        {
            uint ch_num = ch.second;
            float lambda = LAMBDAS.at(ch_num);
            size_t pos = i + 1;
            uint weight = hat(pos);
            A[ch_num](k, i) = lambda * weight;
            A[ch_num](k, i + 1) = -2.f * lambda * weight;
            A[ch_num](k, i + 2) = lambda * weight;
        }
        ++k;
    }

    // solve for x
    vector<Eigen::VectorXf> x(CHANNELS.size());
    vector<vector<float>> results(CHANNELS.size());

    for (const auto &ch : CHANNELS)
    {
        uint ch_num = ch.second;
        // Eigen::VectorXf x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        x[ch_num] = A[ch_num].householderQr().solve(b[ch_num]);
        x[ch_num] = x[ch_num].segment<n>(0);
        results[ch_num] = vector<float>(x[ch_num].data(), x[ch_num].data() + x[ch_num].size());
        // sort(results.begin(), results.end());
    }

    vector<size_t> linspace(n);
    iota(linspace.begin(), linspace.end(), 0);

    for (const auto &ch : CHANNELS)
    {
        uint ch_num = ch.second;
        string ch_name = ch.first;
        matplotlibcpp::plot(results[ch_num], linspace, "-");
    }
    // matplotlibcpp::plot(results[2], linspace);
    matplotlibcpp::legend();
    matplotlibcpp::show(true);
}

void show_image(const cv::Mat &img)
{
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", img);
    cv::waitKey(0);
}

float get_exposure_time(const path &imgPath)
{
    string filename = imgPath.filename().string();
    vector<string> parts = {};
    boost::split(parts, filename, [](char c) { return c == '.'; });
    vector<string> exposure_fraction = {};
    boost::split(exposure_fraction, parts[0], [](char c) { return c == '_'; });
    float numerator = stof(exposure_fraction[0]);
    float denominator = stof(exposure_fraction[1]);
    float exposure = numerator / denominator;

    return exposure;
}

vector<pair<uint, uint>> get_random_indices(const int &num_rows, const int &num_cols, const int &samples)
{
    std::uniform_int_distribution<> dist_r(0, num_rows);
    std::uniform_int_distribution<> dist_c(0, num_cols);
    vector<pair<uint, uint>> indices = {};

    for (int i = 0; i < samples; i++)
    {
        int r_idx = dist_r(rnd_eng);
        int c_idx = dist_c(rnd_eng);
        indices.emplace_back(make_pair<>(r_idx, c_idx));
    }

    return indices;
}

uint hat(const uint &pixel)
{
    return pixel <= (Z_max - Z_min + 1) / 2 ? pixel - Z_min : Z_max - pixel;
}

} // namespace utils
} // namespace hdr