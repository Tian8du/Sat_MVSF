"""
Modernized RPC tensor computation module for Sat-MVSF.
Fully compatible with NumPy>=1.26, CuPy>=13.x, and PyTorch>=2.2.
Author: Chen Liu, 2025
License: GPLv3
"""

import numpy as np
import os
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# === Compatibility patch for removed NumPy aliases ===
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "float"):
    np.float = np.float64
# =====================================================

# =====================================================
# Try GPU (CuPy); if unavailable or driver too old, fallback to NumPy
# =====================================================
try:
    import cupy as cp
    try:
        # Try initializing CUDA runtime to ensure driver compatibility
        _ = cp.zeros((1,))
        _ = cp.cuda.runtime.getDeviceCount()
        _use_gpu = True
        print("✅ CuPy GPU backend enabled")
    except Exception as e:
        print(f"⚠️ CuPy detected but unusable ({e}); falling back to NumPy CPU mode.")
        import numpy as cp
        _use_gpu = False
except ImportError:
    import numpy as cp
    _use_gpu = False
    print("⚠️ CuPy not installed; using NumPy CPU mode")


# ===================================================================
#  RPCModelParameter: handles polynomial tensor form of RPC equations
# ===================================================================
class RPCModelParameter:
    """RPC model using tensor polynomial form (CuPy-accelerated)."""

    def __init__(self, data=np.zeros(170, dtype=np.float64)):
        data = cp.asarray(data, dtype=cp.float64)
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
        """Convert 20 coefficients into symmetric 4×4×4 tensor form."""
        data = cp.asarray(data, dtype=cp.float64)
        assert data.shape[0] == 20
        return cp.array([
            [
                [data[0], data[1]/3, data[2]/3, data[3]/3],
                [data[1]/3, data[7]/3, data[4]/6, data[5]/6],
                [data[2]/3, data[4]/6, data[8]/3, data[6]/6],
                [data[3]/3, data[5]/6, data[6]/6, data[9]/3],
            ],
            [
                [data[1]/3, data[7]/3, data[4]/6, data[5]/6],
                [data[7]/3, data[11], data[14]/3, data[17]/3],
                [data[4]/6, data[14]/3, data[12]/3, data[10]/6],
                [data[5]/6, data[17]/3, data[10]/6, data[13]/3],
            ],
            [
                [data[2]/3, data[4]/6, data[8]/3, data[6]/6],
                [data[4]/6, data[14]/3, data[12]/3, data[10]/6],
                [data[8]/3, data[12]/3, data[15], data[18]/3],
                [data[6]/6, data[10]/6, data[18]/3, data[16]/3],
            ],
            [
                [data[3]/3, data[5]/6, data[6]/6, data[9]/3],
                [data[5]/6, data[17]/3, data[10]/6, data[13]/3],
                [data[6]/6, data[10]/6, data[18]/3, data[16]/3],
                [data[9]/3, data[13]/3, data[16]/3, data[19]],
            ],
        ], dtype=cp.float64)

    @staticmethod
    def QC_cal_en(x, T):
        """Efficient cubic tensor evaluation with einsum."""
        assert x.shape[0] == 4 and T.shape == (4, 4, 4)
        return cp.einsum("ijk,in,jn,kn->n", T, x, x, x, optimize=True)

    def load_dirpc_from_file(self, filepath):
        """Load RPC coefficients from .rpc text file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find RPC file: {filepath}")
        with open(filepath, "r") as f:
            data = cp.array([float(line.split()[1]) for line in f], dtype=cp.float64)
        self.__init__(data)

    def RPC_OBJ2PHOTO(self, lat, lon, hei):
        """(lat, lon, h) → (sample, line)."""
        lat, lon, hei = map(cp.asarray, (lat, lon, hei))
        ones = cp.ones_like(lat)
        x = cp.stack((ones, lon, lat, hei), axis=0)
        x[1] = (x[1] - self.LONG_OFF) / self.LONG_SCALE
        x[2] = (x[2] - self.LAT_OFF) / self.LAT_SCALE
        x[3] = (x[3] - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        samp = self.QC_cal_en(x, self.SNUM) / self.QC_cal_en(x, self.SDEM)
        line = self.QC_cal_en(x, self.LNUM) / self.QC_cal_en(x, self.LDEM)
        samp = samp * self.SAMP_SCALE + self.SAMP_OFF
        line = line * self.LINE_SCALE + self.LINE_OFF
        return tuple(map(cp.asnumpy, (samp, line)))

    def RPC_PHOTO2OBJ(self, samp, line, hei):
        """(sample, line, h) → (lat, lon)."""
        samp, line, hei = map(cp.asarray, (samp, line, hei))
        ones = cp.ones_like(samp)
        x = cp.stack((ones, line, samp, hei), axis=0)
        x[1] = (x[1] - self.LINE_OFF) / self.LINE_SCALE
        x[2] = (x[2] - self.SAMP_OFF) / self.SAMP_SCALE
        x[3] = (x[3] - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        lat = self.QC_cal_en(x, self.LATNUM) / self.QC_cal_en(x, self.LATDEM)
        lon = self.QC_cal_en(x, self.LONNUM) / self.QC_cal_en(x, self.LONDEM)
        lat = lat * self.LAT_SCALE + self.LAT_OFF
        lon = lon * self.LONG_SCALE + self.LONG_OFF
        return tuple(map(cp.asnumpy, (lat, lon)))


# ===================================================================
#  RPCModel: simplified polynomial representation (vectorized form)
# ===================================================================
class RPCModel:
    """Vectorized RPC model for forward/backward projection."""

    def __init__(self, data=np.zeros(170, dtype=np.float64)):
        data = cp.asarray(data, dtype=cp.float64)
        (
            self.LINE_OFF, self.SAMP_OFF, self.LAT_OFF, self.LONG_OFF, self.HEIGHT_OFF,
            self.LINE_SCALE, self.SAMP_SCALE, self.LAT_SCALE, self.LONG_SCALE, self.HEIGHT_SCALE
        ) = data[0:10]
        self.LNUM, self.LDEM, self.SNUM, self.SDEM = data[10:30], data[30:50], data[50:70], data[70:90]
        self.LATNUM, self.LATDEM, self.LONNUM, self.LONDEM = data[90:110], data[110:130], data[130:150], data[150:170]

    @staticmethod
    def RPC_PLH_COEF(P, L, H):
        """Construct 20 polynomial coefficients for cubic RPC evaluation."""
        P, L, H = map(cp.asarray, (P, L, H))
        coef = cp.stack([
            cp.ones_like(P),
            L, P, H,
            L * P, L * H, P * H,
            L * L, P * P, H * H,
            P * L * H, L**3, L * P**2, L * H**2,
            L**2 * P, P**3, P * H**2, L**2 * H, P**2 * H, H**3
        ], axis=-1)
        return coef

    def RPC_OBJ2PHOTO(self, lat, lon, hei):
        """From (lat, lon, hei) → (sample, line)."""
        lat, lon, hei = map(cp.asarray, (lat, lon, hei))
        lat = (lat - self.LAT_OFF) / self.LAT_SCALE
        lon = (lon - self.LONG_OFF) / self.LONG_SCALE
        hei = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        coef = self.RPC_PLH_COEF(lat, lon, hei)
        samp = cp.sum(coef * self.SNUM, axis=-1) / cp.sum(coef * self.SDEM, axis=-1)
        line = cp.sum(coef * self.LNUM, axis=-1) / cp.sum(coef * self.LDEM, axis=-1)
        samp = samp * self.SAMP_SCALE + self.SAMP_OFF
        line = line * self.LINE_SCALE + self.LINE_OFF
        return tuple(map(cp.asnumpy, (samp, line)))

    def RPC_PHOTO2OBJ(self, samp, line, hei):
        """From (sample, line, hei) → (lat, lon)."""
        samp, line, hei = map(cp.asarray, (samp, line, hei))
        samp = (samp - self.SAMP_OFF) / self.SAMP_SCALE
        line = (line - self.LINE_OFF) / self.LINE_SCALE
        hei = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        coef = self.RPC_PLH_COEF(samp, line, hei)
        lat = cp.sum(coef * self.LATNUM, axis=-1) / cp.sum(coef * self.LATDEM, axis=-1)
        lon = cp.sum(coef * self.LONNUM, axis=-1) / cp.sum(coef * self.LONDEM, axis=-1)
        lat = lat * self.LAT_SCALE + self.LAT_OFF
        lon = lon * self.LONG_SCALE + self.LONG_OFF
        return tuple(map(cp.asnumpy, (lat, lon)))


# ===================================================================
#  Test utilities
# ===================================================================
def test_tensordot():
    """Benchmark and validate tensor RPC operations."""
    print("==> Testing RPCModelParameter tensor operations")
    rpc_data = np.random.randn(170).astype(np.float64)
    rpc = RPCModelParameter(rpc_data)
    n = 1000
    lat, lon, hei = cp.random.rand(n), cp.random.rand(n), cp.random.rand(n) * 100
    samp, line = rpc.RPC_OBJ2PHOTO(lat, lon, hei)
    print(f"Forward RPC_OBJ2PHOTO OK, sample mean = {np.mean(samp):.4f}")
    lat2, lon2 = rpc.RPC_PHOTO2OBJ(samp, line, hei)
    print(f"Inverse RPC_PHOTO2OBJ OK, mean diff = {(np.mean(lat - lat2)):.6f}")


def test():
    """Quick performance test for RPCModel (CPU/GPU)."""
    print(f"==> Using {'GPU (CuPy)' if _use_gpu else 'CPU (NumPy)'} backend")
    rpc_data = np.random.randn(170).astype(np.float64)
    rpc = RPCModel(rpc_data)
    n = 200000
    lat, lon, hei = cp.random.rand(n), cp.random.rand(n), cp.random.rand(n) * 100
    samp, line = rpc.RPC_OBJ2PHOTO(lat, lon, hei)
    lat2, lon2 = rpc.RPC_PHOTO2OBJ(samp, line, hei)
    print(f"Projection roundtrip OK, mean diff = {(cp.mean(lat - cp.asarray(lat2))):.6f}")


if __name__ == "__main__":
    test_tensordot()
    test()
