/*
 * Created Date: Friday April 12th 2019
 * Last Modified: Friday April 12th 2019 11:55:17 pm
 * Author: ankurrc
 */

#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <random>
#include <utility>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

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
    path dir_path(dir);
    vector<path> image_paths = {};
    cout << "\nDirectory path is: " << dir_path.string() << "\n";
    if (exists(dir_path))
    {
        if (is_directory(dir_path))
        {
            for (const directory_entry &dir_entry : directory_iterator(dir_path))
            {

                if (is_regular_file(dir_entry))
                    image_paths.emplace_back(dir_entry);
            }
        }
    }
    else
    {
        throw "does not exist.\n";
    }

    cout << "Has " << image_paths.size() << " files.\n";

    return image_paths;
}

vector<vector<float>> calibrate(const vector<path> &imagePaths, const float &lambda)
{

    cout << "\n Calibrating...\n";
    // no. images
    const uint num_images = imagePaths.size();
    // min. samples required
    const uint min_samples = static_cast<int>(ceilf((Z_max - Z_min + 1) / (static_cast<float>(num_images) - 1)));
    // samples per image
    // const uint N = 2 * static_cast<int>(ceilf(static_cast<float>(min_samples) / num_images));
    const uint N = 5 * min_samples;

    cout << "Minimum samples required (per image) for " << num_images << " images: " << min_samples
         << "\nUsing per image: " << N / min_samples << "x" << min_samples << "=" << N << " samples\n";

    const uint n = Z_max - Z_min + 1;

    // setup image dimensions
    // get the image
    cv::Mat init_image = cv::imread(imagePaths[0].string(), cv::IMREAD_COLOR);
    if (!init_image.data)
        throw "No image data for " + imagePaths[0].string();

    double min, max;
    cv::minMaxLoc(init_image, &min, &max);
    cout << "Image channels, max, min: " << init_image.channels() << ", " << max << "," << min << endl;

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
    }

    cout << "A: (" << A[0].rows() << "x" << A[0].cols() << "x" << A.size() << ")\t"
         << "b: (" << b[0].rows() << "x" << b[0].cols() << "x" << b.size() << ")\n";

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
                const float weight = static_cast<float>(hat(pixel));

                // for g(wij)
                A[ch_num](k, pixel) = weight;
                // for log(Ei)
                A[ch_num](k, n + i) = -weight;

                b[ch_num](k) = weight * logf(exposure_time);
            }

            ++k;
        }
    }

    // fix the curve by setting middle value to 0
    for (const auto &ch : CHANNELS)
    {
        A[ch.second](k, (n / 2)) = 1.f;
    }

    ++k;

    // regularize; enforce smoothness
    for (size_t i = 0; i <= n - 2; i++)
    {
        for (const auto &ch : CHANNELS)
        {
            const uint ch_num = ch.second;
            size_t pos = i + 1;
            const float weight = static_cast<float>(hat(pos));
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
        // // x[ch_num] = x[ch_num].segment<n>(0);
        x[ch_num].conservativeResize(n);
        results[ch_num] = vector<float>(x[ch_num].data(), x[ch_num].data() + x[ch_num].size());
        // sort(results.begin(), results.end());
    }

    plot_crf(results, {"blue", "green", "red"});

    return results;
}

void show_image(const cv::Mat &img, const string &name, const float &wait = 0.0f)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img);
    cv::waitKey(wait);
}

float get_exposure_time(const path &imgPath)
{
    string filename = imgPath.filename().string();
    vector<string> parts(2);
    boost::split(parts, filename, [](char c) { return c == '.'; });
    vector<string> exposure_fraction(2);
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

bool save_crf(const vector<vector<float>> &crfs, path &save_path)
{
    bool succeeded = false;
    if (!save_path.is_absolute())
        save_path = absolute(save_path);

    cout << "\nWriting calibration files to " << save_path << endl;

    if (!exists(save_path))
    {
        cout << "Directory " << save_path << " does not exist...";
        if (create_directories(save_path))
            cout << "created\n";
        else
        {
            cout << "failed\n";
            return false;
        }
    }
    else
    {
        cout << "Directory " << save_path << " exists!\n";
    }

    try
    {
        int i = 0;
        string extension = ".calib";
        for (const auto &crf : crfs)
        {
            path path = save_path / string("channel_" + to_string(i) + extension);
            cout << "Saving file: " << path << endl;
            std::ofstream ofs(path.string());
            boost::archive::text_oarchive oa(ofs);
            oa &crf;
            i++;
        }

        succeeded = true;
    }
    catch (...)
    {
        cout << "Exception occured while writing files...\n";
        succeeded = false;
    }

    return succeeded;
}

bool load_crf(vector<vector<float>> &crfs, path &load_path)
{
    bool succeeded = false;
    if (!load_path.is_absolute())
        load_path = absolute(load_path);

    cout << "\nLoading calibration files from " << load_path << endl;

    if (!exists(load_path) || !is_directory(load_path))
    {
        cout << load_path << " is not a valid path to a directory.\n";
        return false;
    }

    try
    {
        int i = 0;
        string ext = ".calib";
        vector<path> files;
        for (const directory_entry &file : directory_iterator(load_path))
        {
            if (is_regular_file(file) && extension(file) == ext)
            {
                files.emplace_back(file);
            }
        }
        std::sort(files.begin(), files.end());

        for (const auto &file : files)
        {
            cout << "Loading file: " << file << endl;
            vector<float> crf;
            std::ifstream ifs(file.string());
            boost::archive::text_iarchive ia(ifs);
            ia &crf;
            crfs.emplace_back(crf);
            i++;
        }

        succeeded = true;
    }
    catch (...)
    {
        cout << "Exception occured while loadingfiles...\n";
        succeeded = false;
    }

    return succeeded;
}

void plot_crf(const vector<vector<float>> &crfs, const vector<string> &names)
{

    assert(names.size() == crfs.size());

    uint i = 0;
    for (const auto &crf : crfs)
    {
        string ch_name = names[i];
        vector<size_t> linspace(crf.size());
        iota(linspace.begin(), linspace.end(), 0);
        matplotlibcpp::figure();
        matplotlibcpp::plot(crf, linspace, "bo");
        matplotlibcpp::title(ch_name);
        matplotlibcpp::xlabel("log exposure X");
        matplotlibcpp::ylabel("pixel value z");
        matplotlibcpp::show(false);
        i++;
    }
    matplotlibcpp::show(true);
}

void generate_hdr(const vector<vector<float>> &crfs, path &image_dir, const float &alpha, const bool &cmp_opencv)
{
    cout << "\nGenerating using global tonemapper...\n";
    vector<path> image_paths = get_paths_in_directory(image_dir);
    vector<cv::Mat> images;
    vector<float> exposure_times;
    for (const auto &img_path : image_paths)
    {
        cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        // image.convertTo(image, 5);
        float exposure_time = get_exposure_time(img_path);
        images.emplace_back(image);
        exposure_times.emplace_back(exposure_time);
    }

    vector<Eigen::MatrixXf> radiance_maps(images[0].channels());
    vector<Eigen::MatrixXf> weight_accumulators(images[0].channels());
    for (size_t i = 0; i < radiance_maps.size(); i++)
    {
        radiance_maps[i] = Eigen::MatrixXf::Zero(images[0].rows, images[0].cols);
        weight_accumulators[i] = Eigen::MatrixXf::Zero(images[0].rows, images[0].cols);
    }

    uint j = 0;
    for (const auto &image : images)
    {
        vector<cv::Mat> img_channels(images[0].channels());
        cv::split(image, img_channels);
        // loop over each channel
        uint i = 0;
        for (const auto &img : img_channels)
        {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> radiance_map;
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> weight_accumulator;
            cv::cv2eigen(img, radiance_map);
            cv::cv2eigen(img, weight_accumulator);
            // cout << "Image map (min, max): " << radiance_map.minCoeff() << ", " << radiance_map.maxCoeff() << endl;
            // g(Zij)
            radiance_map = radiance_map.unaryExpr([&crfs, &i](float c) { 
                float e = crfs[i][static_cast<int>(c)];
                // cout << "taking Zij:" << c << " getting e: " << e <<endl;
                return e; });
            // cout << "Radiance map g(Zij) (min, max): " << radiance_map.minCoeff() << ", " << radiance_map.maxCoeff() << endl;
            // g(Zij) - log(T)
            radiance_map = (radiance_map.array() - logf(exposure_times[j])).matrix();
            // cout << "Radiance map g(Zij)  - log(Tj) (min, max): " << radiance_map.minCoeff() << ", " << radiance_map.maxCoeff() << endl;
            // W(Zij)
            weight_accumulator = weight_accumulator.unaryExpr(ref(hat)).cast<float>();
            // W(Zij)(g(Zij) - log(T))
            radiance_map = radiance_map.cwiseProduct(weight_accumulator);
            // if (i == 0)
            //     cout << "Radiance map: T[" << j << "]: " << exposure_times[j] << " log(T): " << logf(exposure_times[j])
            //          << " \n\tW(Zij)[g(Zij)  - log(Tj)] (min, max): " << radiance_map.minCoeff() << ", " << radiance_map.maxCoeff() << endl;
            // // Accumulate W(Zij)
            weight_accumulators[i] = weight_accumulators[i] + weight_accumulator;
            // cout << "Accumulated weight for channel " << i << " (min, max): " << weight_accumulators[i].minCoeff() << ", " << weight_accumulators[i].maxCoeff() << endl;
            // Accumulate W(Zij)(g(Zij) - log(T))
            radiance_maps[i] = radiance_maps[i] + radiance_map;
            // if (i == 0)
            // cout << "Accumulated radiance W(Zij)(g(Zij) - log(T)) for channel " << i << " (min, max): " << radiance_maps[i].minCoeff() << ", " << radiance_maps[i].maxCoeff() << endl
            //      << endl;

            ++i;
        }
        ++j;
    }

    for (uint i = 0; i < radiance_maps.size(); i++)
    {
        radiance_maps[i] = radiance_maps[i].cwiseQuotient(weight_accumulators[i]);
        // if (i == 0)
        // {
        // cout << "Radiance map W(Zij)(g(Zij) + log(T))/W(Zij)(min, max): " << radiance_maps[i].minCoeff() << ", " << radiance_maps[i].maxCoeff() << endl;
        // cout << "Weights (min, max): " << weight_accumulators[i].minCoeff() << ", " << weight_accumulators[i].maxCoeff() << endl;
        // }
        radiance_maps[i] = radiance_maps[i].unaryExpr(ref(expf));
        // if (i == 0)
        // cout << "Radiance map(E)-->" << i << " (min, max): " << radiance_maps[i].minCoeff() << ", " << radiance_maps[i].maxCoeff() << endl;

        radiance_maps[i] = radiance_maps[i] * alpha;
        // if (i == 0)
        // cout << "After a: Radiance map(E)-->" << i << " (min, max): " << radiance_maps[i].minCoeff() << ", " << radiance_maps[i].maxCoeff() << endl;

        // tone map
        radiance_maps[i] = radiance_maps[i].cwiseQuotient((radiance_maps[i].array() + 1.f).matrix());
        // if (i == 0)
        //     cout << "Radiance map(E/1+E)(min, max): " << radiance_maps[i].minCoeff() << ", " << radiance_maps[i].maxCoeff() << endl;
    }

    vector<cv::Mat> img_channels(images[0].channels());
    for (size_t i = 0; i < radiance_maps.size(); i++)
    {
        cv::eigen2cv(radiance_maps[i], img_channels[i]);
        //show_image(img_channels[i]);
    }

    cv::Mat hdr_img;
    cv::merge(img_channels, hdr_img);
    show_image(hdr_img, image_dir.remove_trailing_separator().filename().string() + string("_global"));

    if (cmp_opencv)
        compare_opencv(images, exposure_times, image_dir.remove_trailing_separator().filename().string() + string("_local"));
}

void compare_opencv(const vector<cv::Mat> &images, const vector<float> &exposure_times, const string &name)
{
    cout << "\nRunning local tonemapper....\n";
    cv::Mat responseDebevec;
    cv::Ptr<cv::CalibrateDebevec> calibrateDebevec = cv::createCalibrateDebevec();
    calibrateDebevec->process(images, responseDebevec, exposure_times);

    cv::Mat hdrDebevec;
    cv::Ptr<cv::MergeDebevec> mergeDebevec = cv::createMergeDebevec();
    mergeDebevec->process(images, hdrDebevec, exposure_times, responseDebevec);

    cv::Mat ldrDurand;
    cv::Ptr<cv::TonemapDurand> tonemapDurand = cv::createTonemapDurand(1.5, 4, 1.0, 1, 1);
    tonemapDurand->process(hdrDebevec, ldrDurand);
    ldrDurand = 3 * ldrDurand;

    show_image(ldrDurand, name);
}

} // namespace utils
} // namespace hdr