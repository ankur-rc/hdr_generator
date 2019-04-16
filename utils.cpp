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

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include "utils.hpp"

namespace hdr
{
namespace utils
{
using namespace boost::filesystem;
using namespace std;

// pixel intensity range
static const uint8_t Z_min = 0, Z_max = 255;
// holders for reconstructing E after CRF has been computed
static vector<cv::Vec3b> Zij = {};
static vector<float> Tij = {};

// regularization parameter
static const float lambda = 50.f;

// random number seeding
static random_device rd;      // obtain a random number from hardware
static mt19937 rnd_eng(rd()); // seed the generator

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
    const uint min_samples = 10 * static_cast<int>(ceilf((Z_max - Z_min + 1) / (static_cast<float>(num_images) - 1)));
    // samples per image
    const uint N = static_cast<int>(ceilf(static_cast<float>(min_samples) / num_images));

    cout << "Minimum Number of samples required for " << num_images << " images: " << min_samples
         << "\nPer picture (min samples/P): " << N << "\n";

    const uint n = Z_max - Z_min + 1;
    // construct A, b for Ax=b;
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(N * num_images + n + 1, N + n);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(A.rows());

    cout << "A (" << A.rows() << ", " << A.cols() << ")\t"
         << "b (" << b.rows() << ", " << b.cols() << ")\n";

    // add entries to A & b
    size_t k = 0;
    for (size_t j = 0; j < num_images; j++)
    {
        path imgPath = imagePaths[j];

        // get the exposure time
        float exposure_time = get_exposure_time(imgPath);
        cout << "Exposure time: " << exposure_time << endl;

        // get the image
        cv::Mat image = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
        if (!image.data)
            throw "No image data for " + imgPath.string();
        //show_image(image);

        // sample the indices
        vector<pair<uint, uint>> indices = get_random_indices(image.rows, image.cols, N);

        // get the pixel values
        for (size_t i = 0; i < indices.size(); i++)
        {
            cv::Vec3b zij = image.at<cv::Vec3b>(indices[i].first, indices[i].second);
            const uint pixel = zij[0];
            const uint weight = hat(pixel);

            // for g(wij)
            A(k, pixel) = static_cast<float>(weight);
            // for log(Ei)
            A(k, n + i) = -static_cast<float>(weight);

            b(k) = weight * log(exposure_time);

            Zij.emplace_back(zij);
            Tij.emplace_back(exposure_time);

            ++k;
        }
    }

    // fix the curve by setting middle value to 0
    A(k++, n / 2) = 1.f;

    // regularize; enforce smoothness
    for (size_t i = 0; i < n - 1; i++)
    {
        size_t pos = i + 1;
        uint weight = hat(pos);
        A(k, i) = lambda * weight;
        A(k, i + 1) = -2.f * lambda * weight;
        A(k, i + 2) = lambda * weight;
        ++k;
    }

    // solve for x
    Eigen::VectorXf x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    cout << "A:" << A << endl;
    cout << "b:" << b << endl;
    cout << "x:" << x.segment<n>(0) << endl;
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
    return pixel < (Z_max - Z_min + 1) / 2 ? pixel : ((Z_max - Z_min + 1) / 2) - pixel;
}

} // namespace utils
} // namespace hdr