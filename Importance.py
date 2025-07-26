import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestRegressor

# 你的作物配置
crops = {
    'maize': {
        'data_path': r'F:\Maize_Data.csv',
        'best_params': r'F:\maize_seed\best_params.csv',
    },
    'rice': {
        'data_path': r'F:\Rice_Data.csv',
        'best_params': r'F:\rice_seed\best_params.csv',
    },
    'wheat': {
        'data_path': r'F:\Wheat_Data.csv',
        'best_params': r'F:\wheat_seed\best_params.csv',
    },
    'soybean': {
        'data_path': r'F:\Soybean_Data.csv',
        'best_params': r'F:\soybean_seed\best_params.csv',
    },
}


def load_rf_models(crop):

    #加载某作物对应的 20 个随机森林模型,模型路径形如 F:\{crop}_seed\RF_{crop}_seed{i}.joblib

    model_dir = rf"F:\{crop}_seed"
    models = []
    for i in range(1, 21):
        path = os.path.join(model_dir, f"RF_{crop}_seed{i}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        models.append(load(path))
    return models


for crop, info in crops.items():
    # 1. 读取数据
    df = pd.read_csv(info['data_path'])

    # 2. 特征：第2列到倒数第2列
    X = df.iloc[:, 1:-1]
    feat_names = X.columns.tolist()

    # 3. 加载并汇总 20 个模型的特征重要性
    models = load_rf_models(crop)
    all_importances = np.vstack([m.feature_importances_ for m in models])
    mean_importance = all_importances.mean(axis=0)

    # 4. 构建 DataFrame，按重要性排序
    fi_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': mean_importance
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    # 5. 打印并保存
    print(f"\n=== {crop.upper()} 平均特征重要性 ===")
    print(fi_df.to_string(index=False))

    out_csv = os.path.splitext(info['data_path'])[0] + '_feature_importances.csv'
    fi_df.to_csv(out_csv, index=False)
    print(f"已保存：{out_csv}")
