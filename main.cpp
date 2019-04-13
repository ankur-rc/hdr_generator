/*
 * Created Date: Thursday April 11th 2019
 * Last Modified: Thursday April 11th 2019 9:36:31 pm
 * Author: ankurrc
*/

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <vector>
#include "include/utils.hpp"

static std::string imagesPath = "/media/ankurrc/new_volume/689_csce_comp_photo/hw4/StarterCode-04/Images/0_Calib_Chapel";

int main(int argc, char **argv)
{
    using namespace std;
    using namespace boost::filesystem;

    if (argc != 2)
    {
        cout << "Usage: ./main dir_path\n";
        return 1;
    }

    vector<path> imagePaths = {};
    path dir(argv[1]);

    try
    {
        imagePaths = u_utils::get_paths_in_directory(dir);
    }
    catch (const filesystem_error &e)
    {
        cout << e.what() << '\n';
    }
    for (const auto &path : imagePaths)
        cout << path << "\n ";
    cout << "\n";
    return 0;
}