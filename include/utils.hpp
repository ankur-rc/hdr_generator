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

namespace u_utils
{
using namespace boost::filesystem;
using namespace std;

vector<path> get_paths_in_directory(const path &dir) throw(filesystem_error);
} // namespace u_utils