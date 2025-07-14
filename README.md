# Sat_MVSF (Production Version)

## Introduction

**This repository is an adaptation of the official Sat-MVSF framework ([GPCV/Sat-MVSF](https://github.com/GPCV/Sat-MVSF))**.  
It is modified and optimized for practical multi-view satellite 3D reconstruction and production scenarios, with improvements in data organization, batch processing, and usability.

Sat-MVSF is a general deep learning MVS framework for three-dimensional (3D) reconstruction from multi-view optical satellite images.

## Differences from Official Sat-MVSF

- Data pipeline is optimized for large-scale satellite datasets and actual production environments.
- Scripts and configuration support flexible multi-view group organization and real project workflows.
- Compatible with the original Sat-MVSF code and evaluation, but easier to use in automated or industrial processes.

## Official Introduction

This is the official implementation for our paper *A general deep learning based framework for 3D reconstruction from multi-view stereo satellite images*...

## Brief introduction
This is the official implementation for our paper *A general deep learning based framework for 3D reconstruction from multi-view stereo satellite images*. Sat-MVSF is a general deep learning MVS based framework to perform three-dimensional (3D) reconstruction of the Earth’s surface from multi-view optical satellite images. 

## Environment
The environment used is list here.
| Package               | Version     |
| --------------------- | ----------- |
| imageio               | 2.9.0       | 
| gdal                  | 3.3.1       |
| pytorch               | 1.4.0       |
| numpy                 | 1.19.2      |
| numpy-groupies        | 0.9.13      |
| pillow                | 8.1.0       |
| opencv-python-headless| 4.5.5.64    |
| pylas                 | 0.4.3       |

## How to run
#### 1. Create info files for your data
The info files includes: 
| File                  | Contents                               |
| --------------------- | -----------                            |
| projection.prj        | the projection infomation              |
| border.txt            | the extent and cell size of DSM        |
| cameras_info.txt      | the pathes of rpc files                |
| images_info.txt       | the pathes of image files              |
| pair.txt              | the pair infomation                    |
| range.txt             | the searh range                        |
**(1) projection.prj**
The *.prj* files can be easily exported from GIS software such as Arcgis.
```
An example
PROJCS["WGS_1984_UTM_Zone_8N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-135.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0],AUTHORITY["EPSG",32608]]
```
**(2) border.txt**
```
x coordinate of the top-left grid cell  # e.g. 493795.02546076314
y coordinate of the top-left grid cell  # e.g. 3323843.8488957686
number of grid in x-direction           # e.g. 2485
number of grid in y-direction           # e.g. 2022
cell size in x-direction                # e.g. 5.0
cell size in y-direction                # e.g. 5.0
```
**(3) cameras_info.txt**
```
number_of_views
id_of_view0 the_path_to_the_rpc_file_of_view0
id_of_view1 the_path_to_the_rpc_file_of_view1
id_of_view2 the_path_to_the_rpc_file_of_view2
...
```
**(4) images_info.txt**
```
number_of_views
id_of_view0 the_path_to_the_img_file_of_view0
id_of_view1 the_path_to_the_img_file_of_view1
id_of_view2 the_path_to_the_img_file_of_view2
...
```
\* Note: For the same satellite image, the id needs to be the same in file (3) and file (4)
**(5) pair.txt**
```
number_of_pairs
the_reference_view_id0
number_of_source_view_for_the_reference0 the_source_view_id01 the_source_view_score01 the_source_view_id02 the_source_view_score01 ...
the_reference_view_id1
number_of_source_view_for_the_reference1 the_source_view_id11 the_source_view_score11 the_source_view_id12 the_source_view_score12 ...
...
```
\* Note: the_source_view_score1 is a const value here and it's the interface left for future work.
**(6) range.txt**
```
height_min
height_max
height_interval
```
When the height_min= height_max = 0, the script will automatically determine the range from the *.rpc* file.
#### 2. Modify the config file
The config options are store in a *.json* file:
```
{
  "run_crop_img":true,              # run image cropping or not
  "run_mvs": true,                  # run mvs or not
  "run_generate_points":true,       # run points generation or not
  "run_generate_dsm":true,          # run dsm generation or not
  "block_size_x": 768,              # the block size in x-direction
  "block_size_y": 384,              # the block size in y-direction
  "overlap_x": 0.0,                 # the overlap in x-direction
  "overlap_y": 0.0,                 # the overlap in y-direction
  "para": 64,                       # base size of the block
  "invalid_value": -999,            # invalid value in dsm
  "position_threshold": 1,          # the geometric consistency check threshold
  "depth_threshold": 500,           # the geometric consistency check threshold
  "relative_depth_threshold": 100,  # the geometric consistency check threshold
  "geometric_num": 2                # the geometric consistency check threshold
}
```
#### 3. Run the script for WHU-TLC dataset
The info files are already created and an example for running the script on WHU-TLC test dataset:
```
python run_whu_tlc.py
```
If you want to run the pipeline for your own data, please refer the *run_whu_tlc.py* and write a new script for your own data. The core code is:
```
pipeline = Pipeline(image_paths, camera_paths, config, prj_str,
                    border_info, depth_range, output, logger, args)
pipeline.run()
```

## Evaluate performance in WHU-TLC test dataset
Run the script *evaluate_tlc.py* by cmd:
```
python evaluate_tlc.py
```
\* Note: unzip the *test_mask.zip* in the same directory of *test* in the WHU-TLC dataset. 
## License
GPLv3. 

## Citation
If you find this code helpful, please cite our work:
```
@article{GAO2023446,
title = {A general deep learning based framework for 3D reconstruction from multi-view stereo satellite images},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {195},
pages = {446-461},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2022.12.012},
url = {https://www.sciencedirect.com/science/article/pii/S0924271622003276},
author = {Jian Gao and Jin Liu and Shunping Ji},
}
```

## Acknowledgement
Thanks to the authors for opening up their outstanding work:
VisSat Satellite Stereo @https://github.com/Kai-46/VisSatToolSet
Cascade MVS-Net: https://github.com/alibaba/cascade-stereo
UCSNet: https://github.com/touristCheng/UCSNet