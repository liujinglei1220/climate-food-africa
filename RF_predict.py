import os
import joblib
import rasterio
import numpy as np
import warnings

warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    message="X does not have valid feature names"
)

# 要处理的作物
crops = ['Maize', 'Rice', 'Wheat', 'Soybean']


parent_root = r"F:\Yearly_Mutiband\historical"
pred_root   = os.path.join(parent_root, "RF_predictions")

for crop in crops:
    crop_lower = crop.lower()
    print(f"\n=== 处理作物：{crop} ===")

    # 模型所在文件夹：F:\maize_seed, F:\rice_seed, ...
    model_folder = fr"F:\{crop_lower}_seed"
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"找不到模型文件夹: {model_folder}")

    # 加载 20 个模型：RF_{crop}_seed1.joblib … RF_{crop}_seed20.joblib
    models = []
    for i in range(1, 21):
        mpath = os.path.join(model_folder, f"RF_{crop_lower}_seed{i}.joblib")
        if not os.path.isfile(mpath):
            raise FileNotFoundError(f"模型文件不存在: {mpath}")
        models.append(joblib.load(mpath))

    # 输入 TIFF 文件夹：F:\…\Maize_mutiband 等
    tif_folder = os.path.join(parent_root, f"{crop}_mutiband")
    if not os.path.isdir(tif_folder):
        raise FileNotFoundError(f"找不到 TIFF 文件夹: {tif_folder}")

    # 输出目录：F:\…\RF_predictions\Maize
    out_folder = os.path.join(pred_root, crop)
    os.makedirs(out_folder, exist_ok=True)

    # 遍历所有 variable_{year}.tif
    for fname in os.listdir(tif_folder):
        if not fname.lower().startswith("variable_") or not fname.lower().endswith(".tif"):
            continue

        year_str = fname.split("_")[-1].replace(".tif", "")
        try:
            year = int(year_str)
        except ValueError:
            print(f"跳过无效年份文件: {fname}")
            continue
        if year < 1981 or year > 2015:
            print(f"跳过不在 1981-2015 范围: {fname}")
            continue

        src_path = os.path.join(tif_folder, fname)
        with rasterio.open(src_path) as src:
            bands   = src.read()
            profile = src.profile

        # 构建掩码并重塑到 (pixels, bands)
        nodata_val = profile.get('nodata', -9999)
        nodata_mask = (bands == nodata_val)
        flat = bands.reshape(bands.shape[0], -1).T
        valid_mask = ~np.any(nodata_mask.reshape(bands.shape[0], -1), axis=0)

        # 累加预测
        sum_pred = np.zeros(flat.shape[0], dtype=np.float64)
        cnt_pred = np.zeros(flat.shape[0], dtype=np.int32)

        for model in models:
            data = np.full_like(flat, np.nan, dtype=np.float64)
            data[valid_mask] = flat[valid_mask]
            vpix = ~np.isnan(data).any(axis=1)
            preds = model.predict(data[vpix])
            sum_pred[vpix] += preds
            cnt_pred[vpix] += 1

        cnt_pred[cnt_pred == 0] = 1
        pred_flat = (sum_pred / cnt_pred).reshape(src.height, src.width)

        # 写出结果
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata_val)
        out_path = os.path.join(out_folder, f"{crop}_{year}.tif")
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred_flat.astype(rasterio.float32), 1)

        print(f"已保存: {out_path}")
