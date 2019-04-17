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

static std::wstring imageDir = L"/media/ankurrc/new_volume/689_csce_comp_photo/hw4/StarterCode-04/Images/0_Calib_Chapel";
static std::wstring calibDir = L"/media/ankurrc/new_volume/689_csce_comp_photo/hw4/calib/0_Calib_Chapel";

int main(int argc, char **argv)
{
    using namespace std;
    using namespace boost::filesystem;

    // if (argc != 2)
    // {
    //     cout << "Usage: ./main dir_path\n";
    //     return 1;
    // }

    vector<path> imagePaths = {};
    path dir(imageDir);
    path calDir(calibDir);

    try
    {
        imagePaths = hdr::utils::get_paths_in_directory(dir);
        vector<vector<float>> crfs;
        bool success;
        // crfs = hdr::utils::calibrate(imagePaths);
        // success = hdr::utils::save_crf(crfs, calDir);
        // crfs.clear();
        success = hdr::utils::load_crf(crfs, calDir);
        hdr::utils::plot_crf(crfs, {"blue", "green", "red"});
    }
    catch (const filesystem_error &e)
    {
        cout << e.what() << '\n';
        return 1;
    }
    catch (...)
    {
        cout << "Exception occured! Exiting...";
        return 1;
    }

    return 0;
}