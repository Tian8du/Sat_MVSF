
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

from osgeo import osr
import numpy as np

from osgeo import osr, gdal
osr.UseExceptions()

class Projection:
    def __init__(self, wkt_str):
        self.spatial_reference = osr.SpatialReference()
        self.spatial_reference.ImportFromWkt(wkt_str)
        self.spatial_reference.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER
        )
    def proj(self, pts, reverse=False):
        shape = pts.shape
        reshaped_pts = pts.reshape(-1, 2)
        if reverse:
            output = self.proj2geo(reshaped_pts)
        else:
            output = self.geo2proj(reshaped_pts)
        return output.reshape(shape)

    def geo2proj(self, geopts):
        geo_sr = self.spatial_reference.CloneGeogCS()

        ct = osr.CoordinateTransformation(geo_sr, self.spatial_reference)

        coords = ct.TransformPoints(geopts)

        projpts = np.array(coords)

        return projpts[:, :2]

    def proj2geo(self, projpts):
        geo_sr = self.spatial_reference.CloneGeogCS()
        ct = osr.CoordinateTransformation(self.spatial_reference, geo_sr)

        coords = ct.TransformPoints(projpts)

        geopts = np.array(coords)

        return geopts[:, :2]


if __name__ == "__main__":
    wtk_str = 'PROJCS["WGS_1984_UTM_Zone_32N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",9.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0],AUTHORITY["EPSG",32632]]'
    proj = Projection(wtk_str)

    coordinates = """406038 332668.498 4816628.783 50.620 15217.000 1764.667 15665.000 1768.000 23696.333 1773.667
483006 332460.841 4816123.659 107.130 15195.000 1911.667 15644.000 1929.333 23662.000 2034.667
483018 300665.265 4794192.126 59.870 7390.000 10179.667 7817.000 10204.333 11552.333 16056.000
483025 311211.168 4792607.508 52.110 10663.333 9852.000 11099.333 9886.000 16631.000 15507.333
483043 280703.704 4799367.120 232.210 1020.667 10196.333 1429.667 10225.333 1673.000 16085.000
483046 285974.773 4785157.789 154.860 3699.000 13691.667 4114.667 13739.333 5825.333 22026.667
483052 277446.301 4783382.588 127.430 1287.000 14798.000 1696.333 14828.667 2082.667 23887.667
9154008 282968.619 4820287.098 283.500 78.000 4327.333 483.000 4331.667 209.333 6116.333
9154009 292199.291 4815077.513 256.450 3240.000 5078.333 3655.000 5103.667 5115.000 7407.333
9154010 291536.272 4802850.345 129.640 3989.333 8474.000 4406.000 8488.667 6278.333 13154.333
9154011 301514.389 4804823.826 257.090 6819.333 7199.333 7244.000 7254.667 10668.333 11027.667
9154012 299622.476 4822102.637 302.320 4915.333 2621.667 5334.667 2656.667 7715.667 3251.000"""

    data = np.array([d.split(" ") for d in coordinates.splitlines()], dtype=float)
    proj_pts = data[:, 1:3]

    pts = proj.proj(proj_pts, True)
    print(pts.shape)
    print("PointID", "LONG", "LAT")

    for i in range(len(data[:, 0])):
        print(int(data[i][0]), pts[i][0], pts[i][1], data[i][3], data[i][4],
              data[i][5], data[i][6], data[i][7], data[i][8], data[i][9])

