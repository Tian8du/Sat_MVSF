import numpy as np
import os
import json


def load_rpc_as_array(filepath):
    if os.path.exists(filepath) is False:
        raise Exception("RPC not found! Can not find " + filepath + " in the file system!")

    with open(filepath, 'r') as f:
        all_the_text = f.read().splitlines()

    data = [text.split(' ')[1] for text in all_the_text]
    # print(data)
    data = np.array(data, dtype=np.float64)

    h_min = data[4] - data[9]
    h_max = data[4] + data[9]

    return data, h_max, h_min


class RPCModelParameter:
    def __init__(self, data=np.zeros(170, dtype=np.float64)):
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

    """Read orginal RPC File"""
    def load_from_file(self, filepath):
        """
        Here, we define:
            direct rpc: from object (lat, lon, hei) to photo (sample, line)
            inverse rpc: from photo (sample, line, hei) to object (lat, lon)
        Function: Read direct rpc from file and then calculate the inverse rpc
        """
        if os.path.exists(filepath) is False:
            print("Error#001: cann't find " + filepath + " in the file system!")
            return

        # with open(filepath, 'r') as f:
        #     all_the_text = f.read().splitlines()

        with open(filepath, 'r') as f:
            data = []
            for line in f:
                if ':' in line:
                    value_str = line.split(':')[1].strip().split(' ')[0]
                    try:
                        value = float(value_str)
                        data.append(value)
                    except ValueError:
                        continue
        data = np.array(data, dtype=np.float64)

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

        self.Calculate_Inverse_RPC()
        print("Read OK")

    def load_from_file2(self, filepath):
        """
        Here, we define:
            direct rpc: from object (lat, lon, hei) to photo (sample, line)
            inverse rpc: from photo (sample, line, hei) to object (lat, lon)
        Function: Read direct rpc from file and then calculate the inverse rpc
        """
        if os.path.exists(filepath) is False:
            print(f"Error#001: can't find {filepath} in the file system!")
            return

        with open(filepath, 'r') as f:
            # 读取 JSON 格式数据
            data = json.load(f)

        # 提取相关的值并转换为浮点数
        self.LINE_OFF = float(data['LINE_OFF'])
        self.SAMP_OFF = float(data['SAMP_OFF'])
        self.LAT_OFF = float(data['LAT_OFF'])
        self.LONG_OFF = float(data['LONG_OFF'])
        self.HEIGHT_OFF = float(data['HEIGHT_OFF'])
        self.LINE_SCALE = float(data['LINE_SCALE'])
        self.SAMP_SCALE = float(data['SAMP_SCALE'])
        self.LAT_SCALE = float(data['LAT_SCALE'])
        self.LONG_SCALE = float(data['LONG_SCALE'])
        self.HEIGHT_SCALE = float(data['HEIGHT_SCALE'])

        # 解析系数部分，注意这些是字符串类型，需进一步转换为列表
        self.LNUM= np.array([float(x) for x in data['LINE_NUM_COEFF'].split()])
        self.SNUM = np.array([float(x) for x in data['SAMP_NUM_COEFF'].split()])
        self.LDEM = np.array([float(x) for x in data['LINE_DEN_COEFF'].split()])
        self.SDEM = np.array([float(x) for x in data['SAMP_DEN_COEFF'].split()])

        self.Calculate_Inverse_RPC()

    def GetH_MAX_MIN(self):
        """
        Get the max and min value of height based on rpc
        :return: hmax, hmin
        """
        hmax = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hmin = self.HEIGHT_OFF - self.HEIGHT_SCALE

        return hmax, hmin

    def Create_Virtual_3D_Grid(self, xy_sample=30, z_sample=20):
        """
        Create_Virtual 3D control grid in the object space
        :return: grid (N, 5)
        """
        lat_max = self.LAT_OFF + self.LAT_SCALE
        lat_min = self.LAT_OFF - self.LAT_SCALE
        lon_max = self.LONG_OFF + self.LONG_SCALE
        lon_min = self.LONG_OFF - self.LONG_SCALE
        hei_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hei_min = self.HEIGHT_OFF - self.HEIGHT_SCALE
        samp_max = self.SAMP_OFF + self.SAMP_SCALE
        samp_min = self.SAMP_OFF - self.SAMP_SCALE
        line_max = self.LINE_OFF + self.LINE_SCALE
        line_min = self.LINE_OFF - self.LINE_SCALE

        lat = np.linspace(lat_min, lat_max, xy_sample)
        lon = np.linspace(lon_min, lon_max, xy_sample)
        hei = np.linspace(hei_min, hei_max, z_sample)

        lat, lon, hei = np.meshgrid(lat, lon, hei)

        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp, line = self.RPC_OBJ2PHOTO(lat, lon, hei)
        grid = np.stack((samp, line, lat, lon, hei), axis=-1)

        selected_grid = []
        for g in grid:
            flag = [g[0] < samp_min, g[0] > samp_max, g[1] < line_min, g[1] > line_max]
            if True in flag:
                continue
            else:
                selected_grid.append(g)

        grid = np.array(selected_grid)
        return grid

    def RPC_PLH_COEF(self, P, L, H):
        n_num = P.shape[0]
        coef = np.zeros((n_num, 20))
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

    def Recalculate_RPC(self, grid):
        """
        Calculate the direct rpc based on Virtual 3D control grid using lst method
        :param grid:
        :return: nothing
        """
        samp, line, lat, lon, hei = np.hsplit(grid.copy(), 5)
        samp -= self.SAMP_OFF
        samp /= self.SAMP_SCALE
        line -= self.LINE_OFF
        line /= self.LINE_SCALE

        lat -= self.LAT_OFF
        lat /= self.LAT_SCALE
        lon -= self.LONG_OFF
        lon /= self.LONG_SCALE
        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        coef = self.RPC_PLH_COEF(lat, lon, hei)

        n_num = coef.shape[0]
        A = np.zeros((n_num * 2, 78))
        A[0: n_num, 0:20] = - coef
        A[0: n_num, 20:39] = samp.reshape(-1, 1) * coef[:, 1:]
        A[n_num:, 39:59] = - coef
        A[n_num:, 59:78] = line.reshape(-1, 1) * coef[:, 1:]

        l = np.concatenate((samp, line), -1)
        l = -l

        x, res, rank, sv = np.linalg.lstsq(A, l, rcond=None)

        self.SNUM = x[0:20]
        self.SDEM[0] = 1.0
        self.SDEM[1:20] = x[20:39]
        self.LNUM = x[39:59]
        self.LDEM[0] = 1.0
        self.LDEM[1:20] = x[59:]

    def Calculate_Inverse_RPC(self):
        grid = self.Create_Virtual_3D_Grid()
        times = self.Solve_Inverse_RPC_ICCV(grid)
        return times

    def Solve_Inverse_RPC_ICCV(self, grid):
        samp, line, lat, lon, hei = np.hsplit(grid.copy(), 5)

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        # 归一化
        samp -= self.SAMP_OFF
        samp /= self.SAMP_SCALE
        line -= self.LINE_OFF
        line /= self.LINE_SCALE

        lat -= self.LAT_OFF
        lat /= self.LAT_SCALE
        lon -= self.LONG_OFF
        lon /= self.LONG_SCALE
        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp, line, hei)

        n_num = coef.shape[0]
        A = np.zeros((n_num * 2, 78))
        A[0: n_num, 0:20] = - coef
        A[0: n_num, 20:39] = lat.reshape(-1, 1) * coef[:, 1:]
        A[n_num:, 39:59] = - coef
        A[n_num:, 59:78] = lon.reshape(-1, 1) * coef[:, 1:]

        l = np.concatenate((lat, lon), -1)
        l = -l

        ATA = np.matmul(A.T, A)

        ATl = np.matmul(A.T, l)

        from utils.iccv_solver import solve_iccv
        x, times = solve_iccv(ATA, ATl)

        self.LATNUM = x[0:20]
        self.LATDEM[0] = 1.0
        self.LATDEM[1:20] = x[20:39]
        self.LONNUM = x[39:59]
        self.LONDEM[0] = 1.0
        self.LONDEM[1:20] = x[59:]

        return times

    def save_orgrpc_to_file(self, filepath):
        """
        Save the direct rpc to a file
        :param filepath: where to store the file
        :return:
        """
        addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                     'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
                     'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:', 'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:',
                     'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:', 'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:',
                     'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:', 'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:',
                     'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:', 'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:',
                     'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:', 'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:',
                     'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:', 'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:',
                     'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:', 'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:',
                     'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:', 'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:',
                     'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:', 'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:',
                     'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:', 'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:',
                     'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:', 'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:',
                     'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:', 'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:',
                     'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:', 'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:',
                     'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:', 'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:',
                     'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:', 'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:',
                     'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:', 'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:',
                     'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:', 'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:',
                     'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:', 'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:',
                     'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:', 'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:',
                     'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:']
        addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                     'meters']

        text = ""

        text += addition0[0] + " " + str(self.LINE_OFF) + " " + addition1[0] + "\n"
        text += addition0[1] + " " + str(self.SAMP_OFF) + " " + addition1[1] + "\n"
        text += addition0[2] + " " + str(self.LAT_OFF) + " " + addition1[2] + "\n"
        text += addition0[3] + " " + str(self.LONG_OFF) + " " + addition1[3] + "\n"
        text += addition0[4] + " " + str(self.HEIGHT_OFF) + " " + addition1[4] + "\n"
        text += addition0[5] + " " + str(self.LINE_SCALE) + " " + addition1[5] + "\n"
        text += addition0[6] + " " + str(self.SAMP_SCALE) + " " + addition1[6] + "\n"
        text += addition0[7] + " " + str(self.LAT_SCALE) + " " + addition1[7] + "\n"
        text += addition0[8] + " " + str(self.LONG_SCALE) + " " + addition1[8] + "\n"
        text += addition0[9] + " " + str(self.HEIGHT_SCALE) + " " + addition1[9] + "\n"

        for i in range(10, 30):
            text += addition0[i] + " " + str(self.LNUM[i - 10]) + "\n"
        for i in range(30, 50):
            text += addition0[i] + " " + str(self.LDEM[i - 50]) + "\n"
        for i in range(50, 70):
            text += addition0[i] + " " + str(self.SNUM[i - 50]) + "\n"
        for i in range(70, 90):
            text += addition0[i] + " " + str(self.SDEM[i - 70]) + "\n"

        f = open(filepath, "w")
        f.write(text)
        f.close()

    """Read direct and inverse RPC"""
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
        data = np.array(data, dtype=np.float64)

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

    def save_dirpc_to_file(self, filepath):
        """
        Save the direct and inverse rpc to a file
        :param filepath: where to store the file
        :return:
        """
        addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                     'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
                     'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:', 'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:',
                     'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:', 'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:',
                     'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:', 'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:',
                     'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:', 'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:',
                     'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:', 'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:',
                     'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:', 'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:',
                     'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:', 'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:',
                     'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:', 'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:',
                     'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:', 'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:',
                     'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:', 'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:',
                     'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:', 'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:',
                     'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:', 'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:',
                     'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:', 'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:',
                     'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:', 'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:',
                     'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:', 'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:',
                     'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:', 'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:',
                     'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:', 'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:',
                     'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:', 'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:',
                     'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:', 'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:',
                     'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:', 'LAT_NUM_COEFF_1:', 'LAT_NUM_COEFF_2:',
                     'LAT_NUM_COEFF_3:', 'LAT_NUM_COEFF_4:', 'LAT_NUM_COEFF_5:', 'LAT_NUM_COEFF_6:',
                     'LAT_NUM_COEFF_7:', 'LAT_NUM_COEFF_8:', 'LAT_NUM_COEFF_9:', 'LAT_NUM_COEFF_10:',
                     'LAT_NUM_COEFF_11:', 'LAT_NUM_COEFF_12:', 'LAT_NUM_COEFF_13:', 'LAT_NUM_COEFF_14:',
                     'LAT_NUM_COEFF_15:', 'LAT_NUM_COEFF_16:', 'LAT_NUM_COEFF_17:', 'LAT_NUM_COEFF_18:',
                     'LAT_NUM_COEFF_19:', 'LAT_NUM_COEFF_20:', 'LAT_DEN_COEFF_1:', 'LAT_DEN_COEFF_2:',
                     'LAT_DEN_COEFF_3:', 'LAT_DEN_COEFF_4:', 'LAT_DEN_COEFF_5:', 'LAT_DEN_COEFF_6:',
                     'LAT_DEN_COEFF_7:', 'LAT_DEN_COEFF_8:', 'LAT_DEN_COEFF_9:', 'LAT_DEN_COEFF_10:',
                     'LAT_DEN_COEFF_11:', 'LAT_DEN_COEFF_12:', 'LAT_DEN_COEFF_13:', 'LAT_DEN_COEFF_14:',
                     'LAT_DEN_COEFF_15:', 'LAT_DEN_COEFF_16:', 'LAT_DEN_COEFF_17:', 'LAT_DEN_COEFF_18:',
                     'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:', 'LONG_NUM_COEFF_1:', 'LONG_NUM_COEFF_2:',
                     'LONG_NUM_COEFF_3:', 'LONG_NUM_COEFF_4:', 'LONG_NUM_COEFF_5:', 'LONG_NUM_COEFF_6:',
                     'LONG_NUM_COEFF_7:', 'LONG_NUM_COEFF_8:', 'LONG_NUM_COEFF_9:', 'LONG_NUM_COEFF_10:',
                     'LONG_NUM_COEFF_11:', 'LONG_NUM_COEFF_12:', 'LONG_NUM_COEFF_13:', 'LONG_NUM_COEFF_14:',
                     'LONG_NUM_COEFF_15:', 'LONG_NUM_COEFF_16:', 'LONG_NUM_COEFF_17:', 'LONG_NUM_COEFF_18:',
                     'LONG_NUM_COEFF_19:', 'LONG_NUM_COEFF_20:', 'LONG_DEN_COEFF_1:', 'LONG_DEN_COEFF_2:',
                     'LONG_DEN_COEFF_3:', 'LONG_DEN_COEFF_4:', 'LONG_DEN_COEFF_5:', 'LONG_DEN_COEFF_6:',
                     'LONG_DEN_COEFF_7:', 'LONG_DEN_COEFF_8:', 'LONG_DEN_COEFF_9:', 'LONG_DEN_COEFF_10:',
                     'LONG_DEN_COEFF_11:', 'LONG_DEN_COEFF_12:', 'LONG_DEN_COEFF_13:', 'LONG_DEN_COEFF_14:',
                     'LONG_DEN_COEFF_15:', 'LONG_DEN_COEFF_16:', 'LONG_DEN_COEFF_17:', 'LONG_DEN_COEFF_18:',
                     'LONG_DEN_COEFF_19:', 'LONG_DEN_COEFF_20:']
        addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                     'meters']

        text = ""

        text += addition0[0] + " " + str(self.LINE_OFF) + " " + addition1[0] + "\n"
        text += addition0[1] + " " + str(self.SAMP_OFF) + " " + addition1[1] + "\n"
        text += addition0[2] + " " + str(self.LAT_OFF) + " " + addition1[2] + "\n"
        text += addition0[3] + " " + str(self.LONG_OFF) + " " + addition1[3] + "\n"
        text += addition0[4] + " " + str(self.HEIGHT_OFF) + " " + addition1[4] + "\n"
        text += addition0[5] + " " + str(self.LINE_SCALE) + " " + addition1[5] + "\n"
        text += addition0[6] + " " + str(self.SAMP_SCALE) + " " + addition1[6] + "\n"
        text += addition0[7] + " " + str(self.LAT_SCALE) + " " + addition1[7] + "\n"
        text += addition0[8] + " " + str(self.LONG_SCALE) + " " + addition1[8] + "\n"
        text += addition0[9] + " " + str(self.HEIGHT_SCALE) + " " + addition1[9] + "\n"

        for i in range(10, 30):
            text += addition0[i] + " " + str(self.LNUM[i - 10]) + "\n"
        for i in range(30, 50):
            text += addition0[i] + " " + str(self.LDEM[i - 30]) + "\n"
        for i in range(50, 70):
            text += addition0[i] + " " + str(self.SNUM[i - 50]) + "\n"
        for i in range(70, 90):
            text += addition0[i] + " " + str(self.SDEM[i - 70]) + "\n"
        for i in range(90, 110):
            text += addition0[i] + " " + str(self.LATNUM[i - 90]) + "\n"
        for i in range(110, 130):
            text += addition0[i] + " " + str(self.LATDEM[i - 110]) + "\n"
        for i in range(130, 150):
            text += addition0[i] + " " + str(self.LONNUM[i - 130]) + "\n"
        for i in range(150, 170):
            text += addition0[i] + " " + str(self.LONDEM[i - 150]) + "\n"

        f = open(filepath, "w")
        f.write(text)
        f.close()

    """CALCULATE"""
    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei):
        """
        From (lat, lon, hei) to (samp, line) using the direct rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        lat = np.copy(inlat)
        lon = np.copy(inlon)
        hei = np.copy(inhei)

        lat -= self.LAT_OFF
        lat /= self.LAT_SCALE

        lon -= self.LONG_OFF
        lon /= self.LONG_SCALE

        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(lat, lon, hei)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = np.sum(coef * self.SNUM, axis=-1) / np.sum(coef * self.SDEM, axis=-1)
        line = np.sum(coef * self.LNUM, axis=-1) / np.sum(coef * self.LDEM, axis=-1)

        samp *= self.SAMP_SCALE
        samp += self.SAMP_OFF

        line *= self.LINE_SCALE
        line += self.LINE_OFF

        return samp, line

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei):
        """
        From (samp, line, hei) to (lat, lon) using the inverse rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        import time

        samp = np.copy(insamp)
        line = np.copy(inline)
        hei = np.copy(inhei)

        samp -= self.SAMP_OFF
        samp /= self.SAMP_SCALE

        line -= self.LINE_OFF
        line /= self.LINE_SCALE

        hei -= self.HEIGHT_OFF
        hei /= self.HEIGHT_SCALE

        t1 = time.time()

        coef = self.RPC_PLH_COEF(samp, line, hei)

        # t2 = time.time()

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        lat = np.sum(coef * self.LATNUM, axis=-1) / np.sum(coef * self.LATDEM, axis=-1)
        lon = np.sum(coef * self.LONNUM, axis=-1) / np.sum(coef * self.LONDEM, axis=-1)

        # t3 = time.time()

        # print(self.LATDEM)
        lat *= self.LAT_SCALE
        lat += self.LAT_OFF

        lon *= self.LONG_SCALE
        lon += self.LONG_OFF

        return lat, lon

    def get_data(self):
        data = [self.LINE_OFF, self.SAMP_OFF, self.LAT_OFF, self.LONG_OFF, self.HEIGHT_OFF,
                self.LINE_SCALE, self.SAMP_SCALE, self.LAT_SCALE, self.LONG_SCALE, self.HEIGHT_SCALE]

        for i in range(10, 30):
            data.append(self.LNUM[i - 10])
        for i in range(30, 50):
            data.append(self.LDEM[i - 30])
        for i in range(50, 70):
            data.append(self.SNUM[i - 50])
        for i in range(70, 90):
            data.append(self.SDEM[i - 70])
        for i in range(90, 110):
            data.append(self.LATNUM[i - 90])
        for i in range(110, 130):
            data.append(self.LATDEM[i - 110])
        for i in range(130, 150):
            data.append(self.LONNUM[i - 130])
        for i in range(150, 170):
            data.append(self.LONDEM[i - 150])

        return data

    def Check_RPC(self, width, height, xy_sample_num, h_sample_num):
        h_max, h_min = self.GetH_MAX_MIN()

        x = np.linspace(0, width, xy_sample_num)
        y = np.linspace(0, height, xy_sample_num)
        h = np.linspace(h_min, h_max, h_sample_num)

        x, y, h = np.meshgrid(x, y, h)
        x = x.reshape(-1)
        y = y.reshape(-1)
        h = h.reshape(-1)

        lat, lon = self.RPC_PHOTO2OBJ(x, y, h)
        # print(lat)
        new_x, new_y = self.RPC_OBJ2PHOTO(lat, lon, h)

        proj_error_x = (new_x - x) * (new_x - x)
        proj_error_y = (new_y - y) * (new_y - y)
        proj_error = np.sqrt(proj_error_x + proj_error_y)

        return proj_error


if __name__ == "__main__":
    rpcs = []
    for i in range(3):
        from dataset.data_io import load_rpc_as_array

        rpc_path = "D:/pipeline_result/index7_idx1/rpc/{}/block0000.rpc".format(i)

        rpc, _, _ = load_rpc_as_array(rpc_path)
        rpcs.append(rpc)

    ref_rpc = RPCModelParameter(rpcs[2])
    src_rpc = RPCModelParameter(rpcs[0])

    import time

    samp = np.zeros(2048 * 1472)
    line = np.zeros(2048 * 1472)
    hei = np.zeros(2048 * 1472) + 665.9574
    start = time.time()
    lat, lon = ref_rpc.RPC_PHOTO2OBJ(samp, line, hei)
    print(lat, lon)

    samp, line = src_rpc.RPC_OBJ2PHOTO(lat, lon, hei)
    end = time.time()
    print(end - start)