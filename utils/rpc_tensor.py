

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


import cupy as cp
import numpy as np
import os
import torch


class RPCModelParameter:
    def __init__(self, data=np.zeros(170, dtype=np.float64)):
        data = cp.asarray(data)

        self.LINE_OFF, self.SAMP_OFF, self.LAT_OFF, self.LONG_OFF, self.HEIGHT_OFF = data[0:5]
        self.LINE_SCALE, self.SAMP_SCALE, self.LAT_SCALE, self.LONG_SCALE, self.HEIGHT_SCALE = data[5:10]

        self.LNUM = self.to_T(data[10:30])
        self.LDEM = self.to_T(data[30:50])
        self.SNUM = self.to_T(data[50:70])
        self.SDEM = self.to_T(data[70:90])

        self.LATNUM = self.to_T(data[90:110])
        self.LATDEM = self.to_T(data[110:130])
        self.LONNUM = self.to_T(data[130:150])
        self.LONDEM = self.to_T(data[150:170])

    @staticmethod
    def to_T(data):
        assert data.shape[0] == 20 and len(data.shape) == 1
        coeff_tensor = cp.array(
            [[[data[0], data[1] / 3.0, data[2] / 3.0, data[3] / 3.0],
              [data[1] / 3.0, data[7] / 3.0, data[4] / 6.0, data[5] / 6.0],
              [data[2] / 3.0, data[4] / 6.0, data[8] / 3.0, data[6] / 6.0],
              [data[3] / 3.0, data[5] / 6.0, data[6] / 6.0, data[9] / 3.0]],

             [[data[1] / 3.0, data[7] / 3.0, data[4] / 6.0, data[5] / 6.0],
              [data[7] / 3.0, data[11], data[14] / 3.0, data[17] / 3.0],
              [data[4] / 6.0, data[14] / 3.0, data[12] / 3.0, data[10] / 6.0],
              [data[5] / 6.0, data[17] / 3.0, data[10] / 6.0, data[13] / 3.0]],

             [[data[2] / 3.0, data[4] / 6.0, data[8] / 3.0, data[6] / 6.0],
              [data[4] / 6.0, data[14] / 3.0, data[12] / 3.0, data[10] / 6.0],
              [data[8] / 3.0, data[12] / 3.0, data[15], data[18] / 3.0],
              [data[6] / 6.0, data[10] / 6.0, data[18] / 3.0, data[16] / 3.0]],

             [[data[3] / 3.0, data[5] / 6.0, data[6] / 6.0, data[9] / 3.0],
              [data[5] / 6.0, data[17] / 3.0, data[10] / 6.0, data[13] / 3.0],
              [data[6] / 6.0, data[10] / 6.0, data[18] / 3.0, data[16] / 3.0],
              [data[9] / 3.0, data[13] / 3.0, data[16] / 3.0, data[19]]]
             ]
        )

        return coeff_tensor

    @staticmethod
    def QC_cal(x, T):
        assert x.shape[0] == 4 and T.shape == (4, 4, 4)

        # x (i, n) (j, n) (k, n)
        # T (i, j, k)
        # print(x.shape, T.shape)
        Tx = cp.tensordot(x, T, axes=[0, 1]) # (100, 4, 4)
        xx = cp.tensordot(x, x, axes=[1, 1])
        y = cp.tensordot(Tx, xx, axes=[(1, 2), (0, 1)])
        # Txx = cp.tensordot(Tx, x, axes=[(1, 0), (0, 1)]) # (4)
        # y = cp.tensordot(Txx, x, axes=[0, 0])

        return y

    @staticmethod
    def QC_cal_en(x, T):
        assert x.shape[0] == 4 and T.shape == (4, 4, 4)
        y = cp.einsum('ijk, in, jn, kn->n', T, x, x, x)

        return y

    def load_dirpc_from_file(self, filepath):
        """
        Read the direct and inverse rpc from a file
        :param filepath:
        :return:
        """
        if os.path.exists(filepath) is False:
            print("Error#001: cann't find " + filepath + " in the file system!")
            return

        with open(filepath, 'r') as f:
            all_the_text = f.read().splitlines()

        data = [text.split(' ')[1] for text in all_the_text]
        # print(data)
        data = cp.array(data, dtype=cp.float64)

        self.LINE_OFF, self.SAMP_OFF, self.LAT_OFF, self.LONG_OFF, self.HEIGHT_OFF = data[0:5]
        self.LINE_SCALE, self.SAMP_SCALE, self.LAT_SCALE, self.LONG_SCALE, self.HEIGHT_SCALE = data[5:10]

        self.LNUM = self.to_T(data[10:30])
        self.LDEM = self.to_T(data[30:50])
        self.SNUM = self.to_T(data[50:70])
        self.SDEM = self.to_T(data[70:90])

        self.LATNUM = self.to_T(data[90:110])
        self.LATDEM = self.to_T(data[110:130])
        self.LONNUM = self.to_T(data[130:150])
        self.LONDEM = self.to_T(data[150:170])

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei):
        assert inlat.shape == inlon.shape and inlon.shape == inhei.shape
        lat = cp.asarray(inlat)
        lon = cp.asarray(inlon)
        hei = cp.asarray(inhei)

        tmp = cp.ones_like(lat)
        x = cp.stack((tmp, lon, lat, hei), axis=0)

        x[1] -= self.LONG_OFF
        x[1] /= self.LONG_SCALE

        x[2] -= self.LAT_OFF
        x[2] /= self.LAT_SCALE

        x[3] -= self.HEIGHT_OFF
        x[3] /= self.HEIGHT_SCALE

        samp = self.QC_cal_en(x, self.SNUM) / self.QC_cal_en(x, self.SDEM)
        line = self.QC_cal_en(x, self.LNUM) / self.QC_cal_en(x, self.LDEM)

        samp *= self.SAMP_SCALE
        samp += self.SAMP_OFF

        line *= self.LINE_SCALE
        line += self.LINE_OFF

        return cp.asnumpy(samp), cp.asnumpy(line)

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei):
        assert insamp.shape == inline.shape and inline.shape == inhei.shape
        samp = cp.asarray(insamp)
        line = cp.asarray(inline)
        hei = cp.asarray(inhei)

        tmp = cp.ones_like(samp)
        x = cp.stack((tmp, line, samp, hei), axis=0)

        x[1] -= self.LINE_OFF
        x[1] /= self.LINE_SCALE

        x[2] -= self.SAMP_OFF
        x[2] /= self.SAMP_SCALE

        x[3] -= self.HEIGHT_OFF
        x[3] /= self.HEIGHT_SCALE

        lat = self.QC_cal_en(x, self.LATNUM) / self.QC_cal_en(x, self.LATDEM)
        lon = self.QC_cal_en(x, self.LONNUM) / self.QC_cal_en(x, self.LONDEM)

        lat *= self.LAT_SCALE
        lat += self.LAT_OFF

        lon *= self.LONG_SCALE
        lon += self.LONG_OFF

        return cp.asnumpy(lat), cp.asnumpy(lon)


class RPCModel():
    def __init__(self, data=np.zeros(170, dtype=np.float64)):
        data = cp.asarray(data)

        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        self.LATNUM = data[90:110]
        self.LATDEM = data[110:130]
        self.LONNUM = data[130:150]
        self.LONDEM = data[150:170]

    def RPC_PLH_COEF(self, P, L, H):
        n_num = P.shape[0]
        coef = cp.zeros((n_num, 20))
        coef[:, 0] = 1.0   # a000
        coef[:, 1] = L     # a100
        coef[:, 2] = P      # a010
        coef[:, 3] = H      # a001
        coef[:, 4] = L * P  # a110
        coef[:, 5] = L * H  # a101
        coef[:, 6] = P * H  # a011
        coef[:, 7] = L * L  # a200
        coef[:, 8] = P * P  # a020
        coef[:, 9] = H * H  # a002
        coef[:, 10] = P * L * H # a111
        coef[:, 11] = L * L * L # a300
        coef[:, 12] = L * P * P # a120
        coef[:, 13] = L * H * H # a102
        coef[:, 14] = L * L * P # a210
        coef[:, 15] = P * P * P # a030
        coef[:, 16] = P * H * H # a012
        coef[:, 17] = L * L * H # a201
        coef[:, 18] = P * P * H # a021
        coef[:, 19] = H * H * H # a003

        return coef

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei):
        """
        From (lat, lon, hei) to (samp, line) using the direct rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        lat = cp.asarray(inlat)
        lon = cp.asarray(inlon)
        hei = cp.asarray(inhei)

        lat -= self.LAT_OFF
        lat /= self.LAT_SCALE

        lon -= self.LONG_OFF
        lon /= self.LONG_SCALE

        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(lat, lon, hei)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = cp.sum(coef * self.SNUM, axis=-1) / cp.sum(coef * self.SDEM, axis=-1)
        line = cp.sum(coef * self.LNUM, axis=-1) / cp.sum(coef * self.LDEM, axis=-1)

        samp *= self.SAMP_SCALE
        samp += self.SAMP_OFF

        line *= self.LINE_SCALE
        line += self.LINE_OFF

        return cp.asnumpy(samp), cp.asnumpy(line)

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei):
        """
        From (samp, line, hei) to (lat, lon) using the inverse rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        # import time

        samp = cp.asarray(insamp)
        line = cp.asarray(inline)
        hei = cp.asarray(inhei)

        samp -= self.SAMP_OFF
        samp /= self.SAMP_SCALE

        line -= self.LINE_OFF
        line /= self.LINE_SCALE

        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        # t1 = time.time()

        coef = self.RPC_PLH_COEF(samp, line, hei)

        # t2 = time.time()

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        lat = cp.sum(coef * self.LATNUM, axis=-1) / cp.sum(coef * self.LATDEM, axis=-1)
        lon = cp.sum(coef * self.LONNUM, axis=-1) / cp.sum(coef * self.LONDEM, axis=-1)

        # t3 = time.time()

        # print(self.LATDEM)
        lat *= self.LAT_SCALE
        lat += self.LAT_OFF

        lon *= self.LONG_SCALE
        lon += self.LONG_OFF

        return cp.asnumpy(lat), cp.asnumpy(lon)


def test_tensordot():
    rpcs = []
    heights = []
    for i in range(3):
        from dataset.data_io import load_rpc_as_array, load_pfm

        rpc_path = "D:/pipeline_result/index7_idx1/rpc/{}/block0000.rpc".format(i)

        rpc, _, _ = load_rpc_as_array(rpc_path)
        rpcs.append(rpc)

        height_map_path = "D:/pipeline_result/index7_idx1/mvs_results/{}/init/block0000.pfm".format(i)
        height_map = load_pfm(height_map_path)
        heights.append(height_map)

    ref_rpc = RPCModelParameter(rpcs[2])
    src_rpc = RPCModelParameter(rpcs[0])

    import time

    height, width = heights[0].shape
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height), indexing='ij')
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    hei_ref = heights[0].reshape([-1])

    start = time.time()
    lat, lon = ref_rpc.RPC_PHOTO2OBJ(x_ref.astype(np.float), y_ref.astype(np.float), hei_ref)
    print(lat, lon)

    samp, line = src_rpc.RPC_OBJ2PHOTO(lat, lon, hei_ref)
    print(samp, line)
    end = time.time()
    print(end - start)


def test():
    rpcs = []
    heights = []
    for i in range(3):
        from dataset.data_io import load_rpc_as_array, load_pfm

        rpc_path = "D:/pipeline_result/index7_idx1/rpc/{}/block0000.rpc".format(i)

        rpc, _, _ = load_rpc_as_array(rpc_path)
        rpcs.append(rpc)

        height_map_path = "D:/pipeline_result/index7_idx1/mvs_results/{}/init/block0000.pfm".format(i)
        height_map = load_pfm(height_map_path)
        heights.append(height_map)

    ref_rpc = RPCModelParameter(rpcs[2])
    src_rpc = RPCModelParameter(rpcs[0])

    import time

    h, w, d = 768, 384, 8
    x_ref, y_ref, h_ref = np.meshgrid(np.arange(0, h), np.arange(0, w), np.arange(0, d),indexing='ij')

    start1 = time.time()
    x, y, h = x_ref.reshape([-1]), y_ref.reshape([-1]), h_ref.reshape([-1])
    x = cp.asarray(x)
    y = cp.asarray(y)
    h = cp.asarray(h)

    tmp = cp.ones_like(x)
    x = cp.stack((tmp, y, x, h), axis=0)

    y = cp.einsum('ijk, in, jn, kn->n', ref_rpc.SNUM, x, x, x)
    end1 = time.time()
    print(end1 - start1)

    start2 = time.time()
    x = cp.asarray(x_ref)
    y = cp.asarray(y_ref)
    h = cp.asarray(h_ref)

    tmp = cp.ones_like(x)
    x = cp.stack((tmp, y, x, h), axis=0)

    y = cp.einsum("ijk, ibcd, jbcd, kbcd->bcd", ref_rpc.SNUM, x, x, x)
    end2 = time.time()
    print(end2 - start2)

    torch_T = torch.from_numpy(cp.asnumpy(ref_rpc.SNUM))
    x = torch.from_numpy(x_ref).double()
    y = torch.from_numpy(y_ref).double()
    h = torch.from_numpy(h_ref).double()

    tmp = torch.ones_like(x, dtype=torch.float64)
    x = torch.stack((tmp, y, x, h), axis=0)

    start3 = time.time()
    y = torch.einsum("ijk, ibcd, jbcd, kbcd->bcd", torch_T, x, x, x)
    end3 = time.time()
    print(end3 - start3)


if __name__ == "__main__":
    test()
