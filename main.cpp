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

static std::string imageDir = "/media/ankurrc/new_volume/689_csce_comp_photo/hw4/StarterCode-04/Images/0_Calib_Chapel";

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

    try
    {
        imagePaths = hdr::utils::get_paths_in_directory(dir);
        hdr::utils::solve_lls(imagePaths);
    }
    catch (const filesystem_error &e)
    {
        cout << e.what() << '\n';
        return 1;
    }

    return 0;
}