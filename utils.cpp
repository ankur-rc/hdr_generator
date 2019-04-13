/*
 * Created Date: Friday April 12th 2019
 * Last Modified: Friday April 12th 2019 11:55:17 pm
 * Author: ankurrc
 */

#include <vector>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include "utils.hpp"

namespace u_utils
{
using namespace boost::filesystem;
using namespace std;

vector<path> get_paths_in_directory(const path &dir) throw(filesystem_error)
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
} // namespace u_utils