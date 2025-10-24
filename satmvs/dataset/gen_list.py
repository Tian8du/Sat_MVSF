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
import glob


def gen_all_mvs_list_rpc(data_folder, view_num):
    """ generate data paths for zy3 dataset """
    sample_list = []

    for r in range(view_num):
        image_folder = os.path.join(data_folder, ('image/%s' % r)).replace("\\", "/")
        rpc_folder = os.path.join(data_folder, ('rpc/%s' % r)).replace("\\", "/")
        height_folder = os.path.join(data_folder, ('height/%s' % r)).replace("\\", "/")

        image_files = os.listdir(image_folder)

        for p in image_files:
            sample = []

            name = os.path.splitext(p)[0]
            ref_image = os.path.join(image_folder, '{}.png'.format(name)).replace("\\", "/")
            ref_rpc = os.path.join(rpc_folder, '{}.rpc'.format(name)).replace("\\", "/")
            ref_height = os.path.join(height_folder, '{}.pfm'.format(name)).replace("\\", "/")

            sample.append(ref_image)
            sample.append(ref_rpc)

            for s in range(view_num):
                sv = (r + s) % view_num

                if sv != r:
                    source_image = os.path.join(data_folder, 'image/{}/{}.png'.format(sv, name)).replace("\\", "/")
                    source_rpc = os.path.join(data_folder, 'rpc/{}/{}.rpc'.format(sv, name)).replace("\\", "/")

                    sample.append(source_image)
                    sample.append(source_rpc)
            sample.append(ref_height)

            sample_list.append(sample)

    return sample_list


def gen_ref_list_rpc(data_folder, view_num, ref_view=2):
    sample_list = []

    image_folder = os.path.join(data_folder, ('image/%s' % ref_view)).replace("\\", "/")
    rpc_folder = os.path.join(data_folder, ('rpc/%s' % ref_view)).replace("\\", "/")
    height_folder = os.path.join(data_folder, ('height/%s' % ref_view)).replace("\\", "/")

    image_files = os.listdir(image_folder)

    for p in image_files:
        sample = []

        name = os.path.splitext(p)[0]
        ref_image = os.path.join(image_folder, '{}.png'.format(name)).replace("\\", "/")
        ref_rpc = os.path.join(rpc_folder, '{}.rpc'.format(name)).replace("\\", "/")
        ref_height = os.path.join(height_folder, '{}.pfm'.format(name)).replace("\\", "/")

        sample.append(ref_image)
        sample.append(ref_rpc)

        for s in range(view_num):
            sv = (ref_view + s) % view_num

            if sv != ref_view:
                source_image = os.path.join(data_folder, 'image/{}/{}.png'.format(sv, name)).replace("\\", "/")
                source_rpc = os.path.join(data_folder, 'rpc/{}/{}.rpc'.format(sv, name)).replace("\\", "/")

                sample.append(source_image)
                sample.append(source_rpc)
        sample.append(ref_height)

        sample_list.append(sample)

    return sample_list