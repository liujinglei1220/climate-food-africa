import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error


crops = {
    'maize'  : {
        'data_path'   : r'F:\Maize_planting_Data.csv',
        'best_params' : r'F:\maize_seed\best_params.csv',
    },
    'rice'   : {
        'data_path'   : r'F:\Rice_planting_Data.csv',
        'best_params' : r'F:\rice_seed\best_params.csv',
    },
    'wheat'  : {
        'data_path'   : r'F:\Wheat_planting_Data.csv',
        'best_params' : r'F:\wheat_seed\best_params.csv',
    },
    'soybean': {
        'data_path'   : r'F:\Soybean_planting_Data.csv',
        'best_params' : r'F:\soybean_seed\best_params.csv',
    },
}

out_base = r'F:\time_cv_results'
os.makedirs(out_base, exist_ok=True)


for crop, cfg in crops.items():
    print(f'\n>> Processing time-based CV for {crop.upper()}')

    # 1. 读取数据
    df = pd.read_csv(cfg['data_path'])
    years = df['Year'].values
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # 2. 加载最佳超参数，并把 NaN 转回 None
    best_series = pd.read_csv(cfg['best_params'], index_col=0)['Value']
    best = best_series.to_dict()
    for k, v in best.items():
        if pd.isna(v):
            best[k] = None
        # 如果原来就是浮点数但需要整数，比如 n_estimators：
        elif k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']:
            best[k] = int(v)

    # 3. 分块交叉验证（GroupKFold 按年份分块）
    n_splits = min(5, len(np.unique(years)))
    gkf = GroupKFold(n_splits=n_splits)
    rows_block = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=years), start=1):
        rf = RandomForestRegressor(**best, n_jobs=-1, random_state=0)
        rf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        y_pred = rf.predict(X.iloc[va_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[va_idx], y_pred))
        rows_block.append({
            'Fold': fold,
            'BlockCV_R2':   r2_score(y.iloc[va_idx], y_pred),
            'BlockCV_RMSE': rmse
        })
    df_block = pd.DataFrame(rows_block)
    df_block.to_csv(
        os.path.join(out_base, f'{crop}_GroupKFold_per_fold.csv'),
        index=False
    )

    # 4. 逐年留一交叉（Leave-One-Year-Out，跳过最早年份）
    unique_years = sorted(np.unique(years))
    rows_lyo = []
    for yr in unique_years[1:]:
        mask_tr = years < yr
        mask_va = years == yr
        if mask_va.sum() == 0:
            continue
        rf = RandomForestRegressor(**best, n_jobs=-1, random_state=0)
        rf.fit(X[mask_tr], y[mask_tr])
        y_pred = rf.predict(X[mask_va])
        rmse = np.sqrt(mean_squared_error(y[mask_va], y_pred))
        rows_lyo.append({
            'Year': yr,
            'LYO_R2':   r2_score(y[mask_va], y_pred),
            'LYO_RMSE': rmse
        })
    df_lyo = pd.DataFrame(rows_lyo)
    df_lyo.to_csv(
        os.path.join(out_base, f'{crop}_LYO_per_year.csv'),
        index=False
    )

    # 5. 汇总两种时序CV的均值与标准差
    summary = {
        'BlockCV_R2_mean':   df_block['BlockCV_R2'].mean(),
        'BlockCV_R2_std':    df_block['BlockCV_R2'].std(),
        'BlockCV_RMSE_mean': df_block['BlockCV_RMSE'].mean(),
        'BlockCV_RMSE_std':  df_block['BlockCV_RMSE'].std(),
        'LYO_R2_mean':       df_lyo['LYO_R2'].mean()  if not df_lyo.empty else np.nan,
        'LYO_R2_std':        df_lyo['LYO_R2'].std()   if not df_lyo.empty else np.nan,
        'LYO_RMSE_mean':     df_lyo['LYO_RMSE'].mean() if not df_lyo.empty else np.nan,
        'LYO_RMSE_std':      df_lyo['LYO_RMSE'].std()  if not df_lyo.empty else np.nan,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_base, f'{crop}_timeblock_cv_summary.csv'),
        index=False
    )

    print(f'✔ Finished {crop.upper()} time-CV.')
