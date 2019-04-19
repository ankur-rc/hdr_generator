/*
 * Created Date: Thursday April 11th 2019
 * Last Modified: Thursday April 11th 2019 9:36:31 pm
 * Author: ankurrc
*/

#include <iostream>
#include <string>
#include <vector>
#include <math.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "utils.hpp"

int main(int argc, char **argv)
{
    using namespace std;
    using namespace boost::filesystem;

    if (argc < 7)
    {
        cout << R"(
Generate HDR images from SDR images taken with different exposures
Filenames should follow x_y_z.jpg format, where 'x' is name, 'y/z' is the exposure time.
            
    Usage: ./hdr dirpath calibdir calibflag lambda alpha
            dirpath:    Path to images
            calibdir:   Path to load/save calibration files
            calibflag:  Flag to specify if we want to calibrate or not
            lambda:     Regularization constant while calibrating
            alpha:      Global tone-mapping constant
            opencv_cmp: Compare with openCV's Debevec-Durand Algorithm
    )";
        return 1;
    }

    vector<path> imagePaths = {};

    path dir(argv[1]);
    path calDir(argv[2]);
    bool calibrate = stoi(argv[3]);
    float lambda = stof(argv[4]);
    float alpha = stof(argv[5]);
    bool cmp_opencv = stoi(argv[6]);

    try
    {
        vector<vector<float>> crfs;
        if (calibrate)
        {
            imagePaths = hdr::utils::get_paths_in_directory(dir);
            crfs = hdr::utils::calibrate(imagePaths, lambda);
            if (!hdr::utils::save_crf(crfs, calDir))
            {
                cerr << "failed to save the calibration files!\n";
                return 1;
            }
        }
        else
        {
            if (!hdr::utils::load_crf(crfs, calDir))
            {
                cerr << "failed to load the calibration files!\n";
                return 1;
            }

            hdr::utils::plot_crf(crfs, {"blue", "green", "red"});
        }

        hdr::utils::generate_hdr(crfs, dir, alpha, cmp_opencv);
    }
    catch (const filesystem_error &e)
    {
        cout << e.what() << '\n';
        return 1;
    }
    catch (const char *e)
    {
        cout << e << "\nException occured! Exiting...\n";
        return 1;
    }
    catch (...)
    {
        exception_ptr e_ptr = current_exception();
        cout << e_ptr.__cxa_exception_type() << "\nException occured! Exiting...\n";
        return 1;
    }

    return 0;
}