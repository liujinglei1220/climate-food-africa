import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
'''
config = {
    "font.family": 'Times New Roman',
    "font.size": 24,
    "mathtext.fontset": 'stix'
}
rcParams.update(config)

# 作物列表
crops = ['maize', 'rice', 'wheat', 'soybean']

# 存储所有作物评估指标
metrics_list = []

for crop in crops:
    # 路径配置
    data_path  = rf"F:\{crop.capitalize()}_Data.csv"
    models_dir = rf"F:\{crop}_seed"
    out_png    = rf"F:\{crop}_density_plot.png"

    # 1. 读取数据并划分留出集
    data = pd.read_csv(data_path)
    X    = data.iloc[:, 1:-1]
    y    = data.iloc[:, -1]
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 加载 20 个模型
    models = []
    for seed in range(1, 21):
        fn = os.path.join(models_dir, f"RF_{crop}_seed{seed}.joblib")
        models.append(joblib.load(fn))

    # 3. 在测试集上做预测并取平均
    preds  = np.column_stack([m.predict(X_holdout) for m in models])
    y_pred = preds.mean(axis=1)

    # 4. 计算并存储评估指标
    MSE = mean_squared_error(y_holdout, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_holdout, y_pred)
    R2 = r2_score(y_holdout, y_pred)
    n = len(y_holdout)
    p = X_holdout.shape[1]
    Adjusted_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    MSPE = np.mean(((y_holdout - y_pred) / y_holdout) ** 2)

    metrics_list.append({
        'Crop': crop,
        'R2': R2,
        'RMSE': RMSE,
        'MSE': MSE,
        'MAE': MAE
    })

    # 5. 计算点密度（KDE）绘图数据
    xy = np.vstack([y_holdout.values, y_pred])
    z  = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_true_sorted = y_holdout.iloc[idx]
    y_pred_sorted = y_pred[idx]
    z_sorted      = z[idx]

    # 6. 绘制密度散点图
    fig, ax = plt.subplots(figsize=(12, 9), dpi=1200)
    sc = ax.scatter(
        y_true_sorted, y_pred_sorted,
        c=z_sorted, cmap='Spectral_r',
        s=15, marker='o', edgecolors='none'
    )
    cbar = plt.colorbar(
        sc, shrink=1, orientation='vertical',
        extend='both', pad=0.015, aspect=30
    )
    cbar.set_label('Density')

    # 7. 统一坐标范围与刻度
    min_val = 0
    max_val = max(y_true_sorted.max(), y_pred_sorted.max()) * 1.05
    step    = 2
    ticks   = np.arange(min_val, max_val + step, step)
    max_tick = ticks[-1]

    ax.set_xlim(min_val, max_tick)
    ax.set_ylim(min_val, max_tick)
    ax.set_aspect('equal', 'box')
    ax.plot([min_val, max_tick], [min_val, max_tick], 'black', lw=1.5)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))

    plt.xlabel('Actual Values (t/ha)',   family='Times New Roman')
    plt.ylabel('Predicted Values (t/ha)', family='Times New Roman')

    # 8. 保存并关闭
    plt.tight_layout()
    plt.savefig(out_png, dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Saved {crop} density plot to {out_png}")

# 4. 汇总所有作物的评估指标并保存为 CSV
df_metrics = pd.DataFrame(metrics_list)
out_csv = r'F:\cv_metrics_summary.csv'
df_metrics.to_csv(out_csv, index=False)
print(f"\nAll metrics saved to {out_csv}\n")
'''



#小麦为何在15t/ha处截断
import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# 作物文件夹和标签
folders = {
    'Maize': r'F:\ljl\yield\Africa_maize',
    'Rice': r'F:\ljl\yield\Africa_rice',
    'Soybean': r'F:\ljl\yield\Africa_soybean',
    'Wheat': r'F:\ljl\yield\Africa_wheat'
}

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (crop, folder) in zip(axes, folders.items()):
    pattern = os.path.join(folder, '*.tif')
    values = []

    for tif_path in glob.glob(pattern):
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            mask = (data != src.nodata) & ~np.isnan(data)
            values.append(data[mask])

    if not values:
        ax.text(0.5, 0.5, 'No data found', ha='center', va='center')
        ax.set_title(crop)
        continue

    # 拼接所有值
    all_vals = np.concatenate(values)

    # 绘制柔和灰蓝色直方图
    bins = 100
    ax.hist(
        all_vals, bins=bins, density=True,
        color='#7fb8ce', edgecolor='#375a70', alpha=0.85
    )

    # 计算并叠加 KDE 曲线（深红色）
    kde = gaussian_kde(all_vals)
    xs = np.linspace(all_vals.min(), all_vals.max(), 200)
    ax.plot(xs, kde(xs), color='#c62828', linewidth=2)

    ax.set_title(f'{crop} Yield Distribution')
    ax.set_xlabel('Yield (t/ha)')
    ax.set_ylabel('Density')

plt.tight_layout()

# 保存图像
out_path = r'F:\ljl\yield\yield_distributions.png'
plt.savefig(out_path, dpi=1200)
plt.show()
print(f"图像已保存到：{out_path}")
