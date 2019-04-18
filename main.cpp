/*
 * Created Date: Thursday April 11th 2019
 * Last Modified: Thursday April 11th 2019 9:36:31 pm
 * Author: ankurrc
*/

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

// static std::wstring imageDir = L"/media/ankurrc/new_volume/689_csce_comp_photo/hw4/StarterCode-04/Images/0_Calib_Chapel";
// static std::wstring calibDir = L"/media/ankurrc/new_volume/689_csce_comp_photo/hw4/calib/0_Calib_Chapel";

int main(int argc, char **argv)
{
    using namespace std;
    using namespace boost::filesystem;

    if (argc < 5)
    {
        cout << "Usage: ./main dirpath calibdir calibflag lambda alpha\n";
        return 1;
    }

    vector<path> imagePaths = {};

    bool calibrate = stoi(argv[3]);
    float lambda = stof(argv[4]);
    path dir(argv[1]);
    path calDir(argv[2]);

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

        hdr::utils::generate_hdr(crfs, dir);
    }
    catch (const filesystem_error &e)
    {
        cout << e.what() << '\n';
        return 1;
    }
    catch (const exception &e)
    {
        cout << e.what() << "\nException occured! Exiting...\n";
        return 1;
    }

    return 0;
}