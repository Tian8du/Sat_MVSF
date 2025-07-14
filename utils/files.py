
"""
This is a script for running the Sat-MVSF.
Copyright (C) <2023> <Jian Gao & GPCV>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os


def ensure_forward_slash(path):
    return path.replace("\\", "/")


def get_all_files(root, select_extensions=None, need_whole_path=True):
    filenames = os.listdir(root)

    select_file_paths = []
    for file_name in filenames:
        file_extension = os.path.splitext(file_name)[-1]

        if isinstance(select_extensions, list):
            if file_extension in select_extensions:
                if need_whole_path:
                    select_file_paths.append(ensure_forward_slash(os.path.join(root, file_name)))
                else:
                    select_file_paths.append(ensure_forward_slash(file_name))

        elif isinstance(select_extensions, str):
            if file_extension == select_extensions:
                if need_whole_path:
                    select_file_paths.append(ensure_forward_slash(os.path.join(root, file_name)))
                else:
                    select_file_paths.append(ensure_forward_slash(file_name))

    select_file_paths.sort()

    return select_file_paths


def mkdir_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
