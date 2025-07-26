import os
import glob
import numpy as np
import rasterio
from joblib import load


# 作物列表
crops = ['Maize', 'Rice', 'Wheat', 'Soybean']

# SSP 情景列表
ssps = ['ssp245', 'ssp370', 'ssp585']

# GCM 模型列表
gcms = ['BCC-CSM2-MR', 'CanESM5', 'IPSL-CM6A-LR', 'GFDL-ESM4', 'MPI-ESM1-2-LR']

# 多波段 TIFF 根目录（每个作物一个子文件夹）
base_tif_dir = r"F:\Yearly_Mutiband\Future_Crop_Mutiband"

# 输出目录：RF 预测结果，按作物/ssp/GCM 分层存放
out_base = r"F:\Yearly_Mutiband\Future_Crop_Mutiband\RF_predictions\each_year"


# —— 函数定义 —— #

def load_rf_models(crop):
    """
    加载某作物对应的 20 个随机森林模型
    模型路径形如 F:\{crop}_seed\RF_{crop}_seed{i}.joblib
    """
    model_dir = rf"F:\{crop.lower()}_seed"
    models = []
    for i in range(1, 21):
        path = os.path.join(model_dir, f"RF_{crop.lower()}_seed{i}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        models.append(load(path))
    return models


def parse_tif_info(fname):
    """
    从 tif 文件名解析 crop, ssp, model, year 信息
    文件名格式：<Crop>_<ssp>_<GCM>_variable_<year>.tif
    """
    parts = os.path.basename(fname)[:-4].split('_')
    crop = parts[0]
    ssp  = parts[1]
    model= parts[2]
    year = parts[-1]
    return crop, ssp, model, year



for crop in crops:
    print(f"=== Processing crop: {crop} ===")
    # 1) 加载 20 个 RF 模型
    rf_models = load_rf_models(crop)

    # 2) 找到该作物所有的多波段 TIFF
    tif_folder = os.path.join(base_tif_dir, crop.lower() + "_mutiband")
    all_tifs = glob.glob(os.path.join(tif_folder, f"{crop}_*.tif"))

    for tif_path in all_tifs:
        crop0, ssp, model, year = parse_tif_info(tif_path)
        # 只处理特定的 SSP 和 GCM
        if ssp not in ssps or model not in gcms:
            continue

        # 3) 读取多波段自变量栅格，并保留 profile、nodata 和 mask
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()    # 保存投影、仿射变换、dtype、nodata 等属性
            nodata = src.nodata             # 原始 NoData 值
            # 以 masked array 方式读取，自动屏蔽 nodata
            X_masked = src.read(masked=True)  # shape = (bands, H, W)
            bands, H, W = X_masked.shape

            # 构造有效像元掩膜：所有波段均非 nodata
            valid_mask = ~X_masked.mask.any(axis=0)  # shape = (H, W)

            # 将栅格数据展平为 (n_pixels, bands) 方便 predict
            X_flat = X_masked.data.reshape(bands, -1).T  # (H*W, bands)

        # 4) 用 20 个模型依次预测，收集所有预测结果
        preds = np.zeros((20, H, W), dtype=np.float32)
        for idx, rf in enumerate(rf_models):
            # 批量预测：得到长度为 H*W 的一维
            y_flat = rf.predict(X_flat)
            preds[idx] = y_flat.reshape(H, W)

        # 5) 计算 20 个模型的平均结果
        mean_map = preds.mean(axis=0)  # shape = (H, W)

        # 6) 将无效像元置为 nodata
        if nodata is None:
            # 如果原 TIFF 没定义 nodata，可以自定义一个（如 -9999）
            nodata = -9999
        mean_map[~valid_mask] = nodata

        # 7) 准备输出 profile：单波段、float32、包含 nodata
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata)

        # 8) 写出 GeoTIFF
        out_dir = os.path.join(out_base, crop, ssp, model)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"mean_{year}.tif")

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mean_map, 1)

        print(f"  → Written: {out_path}")

print("Annual RF prediction completed!")
