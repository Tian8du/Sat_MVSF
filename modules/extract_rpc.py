from osgeo import gdal

# 打开影像
dataset = gdal.Open("E:/Codes2/sat-mvsf/image/JAX_017_001_RGB.tif")

# 提取 RPC 信息
rpc_metadata = dataset.GetMetadata("RPC")
if rpc_metadata:
    with open("E:/Codes2/sat-mvsf/image/JAX_017_001_RGB.rpc", "w") as rpc_file:
        for key, value in rpc_metadata.items():
            rpc_file.write(f"{key}: {value}\n")
    print("RPC 信息已成功导出为 .rpc 文件")
else:
    print("未找到 RPC 信息")

