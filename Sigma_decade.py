import os
import numpy as np
import rasterio
from joblib import load

# 作物列表（文件夹和 joblib 均以小写作物名命名）
crops = ['Maize', 'Rice', 'Wheat', 'Soybean']
# SSP 情景列表
ssps = ['ssp245', 'ssp370', 'ssp585']
# GCM 模型列表
gcms = ['BCC-CSM2-MR', 'CanESM5', 'IPSL-CM6A-LR', 'GFDL-ESM4', 'MPI-ESM1-2-LR']
# 十年期定义
decades = {
    '2030s': list(range(2030, 2040)),
    '2050s': list(range(2050, 2060)),
    '2080s': list(range(2080, 2090))
}

# 年度平均产量结果根目录
annual_dir = r"F:\Yearly_Mutiband\Future_Crop_Mutiband\RF_predictions\each_year"
# 十年期总不确定度输出根目录
out_dir    = r"F:\Yearly_Mutiband\Future_Crop_Mutiband\RF_predictions\decade_sigma_total"

# 加载 RF 模型函数
def load_rf_models(crop):
    """
    加载位于 F:\{crop_lower}_seed\ 的 20 个 RF 模型，
    文件名格式：RF_{crop_lower}_seed1.joblib ... RF_{crop_lower}_seed20.joblib
    crop 参数：'Maize','Rice','Wheat','Soybean'
    """
    crop_l = crop.lower()
    model_dir = f"F:\\{crop_l}_seed"
    models = []
    for i in range(1, 21):
        fname = f"RF_{crop_l}_seed{i}.joblib"
        path  = os.path.join(model_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
        models.append(load(path))
    return models

# 确保输出根目录存在
os.makedirs(out_dir, exist_ok=True)

# ------------------ 计算十年期总不确定度（处理 NoData=-9999） ------------------
for crop in crops:
    print(f"Processing crop: {crop}")
    # 加载该作物的 20 个 RF 模型
    rf_models = load_rf_models(crop)

    for ssp in ssps:
        for decade_tag, years in decades.items():
            print(f"  SSP: {ssp}, Decade: {decade_tag}")
            # 存储所有单模型十年平均栅格（含 NaN 表示 NoData）
            stack = []
            nodata = None
            profile = None

            # 遍历 5 个 GCM 与 20 个模型
            for gcm in gcms:
                for rf in rf_models:
                    # 收集 10 年的年度平均栅格，并转换 NaN
                    dec_maps = []
                    for year in years:
                        path = os.path.join(
                            annual_dir, crop, ssp, gcm,
                            f"mean_{year}.tif"
                        )
                        with rasterio.open(path) as src:
                            if profile is None:
                                profile = src.profile.copy()
                                nodata = src.nodata
                            arr = src.read(1).astype(np.float32)
                        # 将 NoData 转为 NaN
                        if nodata is not None:
                            arr[arr == nodata] = np.nan
                        dec_maps.append(arr)
                    # 该模型在十年期的时间平均，忽略 NaN
                    dec_avg = np.nanmean(np.stack(dec_maps, axis=0), axis=0)
                    # 保持全 NaN 区域为 nodata
                    if nodata is not None:
                        dec_avg[np.isnan(dec_avg)] = nodata
                    stack.append(dec_avg)

            # 合并 100 张栅格，按像素计算标准差，忽略 NaN
            arr_stack = np.stack(stack, axis=0)
            sigma_total = np.nanstd(arr_stack, axis=0, ddof=1)
            # 保持无数据区域
            if nodata is not None:
                sigma_total[np.isnan(sigma_total)] = nodata

            # 写出总不确定度影像（保留 nodata）
            out_path = os.path.join(
                out_dir, crop, ssp,
                f"decade_sigma_total_{crop}_{ssp}_{decade_tag}.tif"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            profile.update(count=1, dtype=rasterio.float32, nodata=nodata)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(sigma_total.astype(np.float32), 1)

            print(f"    → Written: {out_path}")

print("All decade sigma_total images generated.")
