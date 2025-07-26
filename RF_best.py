import os
import time
import joblib
import warnings

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, GroupKFold, cross_val_score
)
from sklearn.metrics import r2_score, root_mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------- 定义作物及路径 -----------------------
crops = {
    'maize':   {'data_path': r"F:\Maize_planting_Data.csv",   'out_dir': r"F:\maize_seed"},
    'rice':    {'data_path': r"F:\Rice_planting_Data.csv",    'out_dir': r"F:\rice_seed"},
    'wheat':   {'data_path': r"F:\Wheat_planting_Data.csv",   'out_dir': r"F:\wheat_seed"},
    'soybean': {'data_path': r"F:\Soybean_planting_Data.csv", 'out_dir': r"F:\soybean_seed"},
}

# ----------------------- 超参网格 -----------------------
param_grid = {
    'n_estimators'     : [300, 400, 500, 600],
    'max_features'     : ['sqrt', 'log2', None],
    'max_depth'        : [None, 10, 20],
    'min_samples_leaf' : [1, 2, 3],
    'min_samples_split': [2, 5, 10],
}


for crop, cfg in crops.items():
    data_path = cfg['data_path']
    out_dir   = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== Processing {crop.upper()} =====")

    # 1. 读取数据
    data = pd.read_csv(data_path)
    years = data['Year'].values
    X     = data.iloc[:, 1:-1]
    y     = data.iloc[:, -1]

    # 2. 留出最终测试集 20%
    X_pool, X_holdout, y_pool, y_holdout, years_pool, years_holdout = train_test_split(
        X, y, years, test_size=0.2, random_state=42
    )

    # 3. 超参网格搜索（在 X_pool 上）
    base_rf = RandomForestRegressor(oob_score=True, n_jobs=-1, random_state=0)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=5,
        scoring={'mse': 'neg_mean_squared_error', 'r2': 'r2'},
        refit='r2',
        return_train_score=False,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_pool, y_pool)

    # 保存超参搜索结果
    cv = grid_search.cv_results_
    rows = []
    for i in range(len(cv['params'])):
        params   = cv['params'][i].copy()
        mean_mse = -cv['mean_test_mse'][i]
        mean_r2  =  cv['mean_test_r2'][i]
        rows.append({
            **params,
            'Mean_CV_MSE' : mean_mse,
            'Mean_CV_RMSE': np.sqrt(mean_mse),
            'Mean_CV_R2'  : mean_r2
        })
    df_search = pd.DataFrame(rows)
    best      = grid_search.best_params_
    best_row  = {**best, 'Mean_CV_MSE': np.nan, 'Mean_CV_RMSE': np.nan, 'Mean_CV_R2': np.nan}
    df_search = df_search.append(best_row, ignore_index=True)
    df_search.to_csv(os.path.join(out_dir, "hyperparam_search_results.csv"), index=False)
    pd.Series(best, name='Value').to_frame().to_csv(
        os.path.join(out_dir, "best_params.csv"), header=['Value']
    )

    # 4. 在 X_pool（80% 数据）上进行 20 次 80/20 随机抽样训练 & 评估
    n_seeds = 20
    seed_results    = []
    holdout_results = []
    models          = []

    for seed in range(1, n_seeds + 1):
        t0 = time.time()
        # 在 X_pool 上做 80/20 拆分
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_pool, y_pool, test_size=0.2, random_state=seed
        )

        rf = RandomForestRegressor(**best, oob_score=True, n_jobs=-1, random_state=seed)

        # 10 折 CV on training split
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        mse_cv = -cross_val_score(
            rf, X_tr, y_tr,
            cv=kf,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        r2_cv = cross_val_score(
            rf, X_tr, y_tr,
            cv=kf,
            scoring='r2',
            n_jobs=-1
        )

        # 训练并评估验证集
        rf.fit(X_tr, y_tr)
        y_val_pred = rf.predict(X_val)
        rmse_val   = root_mean_squared_error(y_val, y_val_pred)
        r2_val     = r2_score(y_val, y_val_pred)
        elapsed    = time.time() - t0

        seed_results.append({
            'Seed': seed,
            'Train_CV_R2':   r2_cv.mean(),
            'Train_CV_RMSE': np.sqrt(mse_cv).mean(),
            'OOB_Score':     rf.oob_score_,
            'Val_R2':        r2_val,
            'Val_RMSE':      rmse_val,
            'Time_s':        elapsed
        })

        # 评估该模型对固定测试集（X_holdout）的表现
        y_hold_pred = rf.predict(X_holdout)
        holdout_results.append({
            'Seed': seed,
            'Holdout_R2':   r2_score(y_holdout, y_hold_pred),
            'Holdout_RMSE': root_mean_squared_error(y_holdout, y_hold_pred)
        })

        models.append(rf)
        joblib.dump(rf, os.path.join(out_dir, f"RF_{crop}_seed{seed}.joblib"))

    pd.DataFrame(seed_results).to_csv(
        os.path.join(out_dir, f"RF_{crop}_seed_results.csv"), index=False
    )

    # 保存每个模型在测试集表现及其平均
    df_hold_per_model = pd.DataFrame(holdout_results)
    avg_row = {
        'Seed': 'Average',
        'Holdout_R2':   df_hold_per_model['Holdout_R2'].mean(),
        'Holdout_RMSE': df_hold_per_model['Holdout_RMSE'].mean()
    }
    df_hold_per_model = df_hold_per_model.append(avg_row, ignore_index=True)
    df_hold_per_model.to_csv(
        os.path.join(out_dir, f"RF_{crop}_holdout_per_model.csv"), index=False
    )

    # 5. 固定测试集上的 ensemble 评估
    preds = np.column_stack([m.predict(X_holdout) for m in models])
    y_ens = preds.mean(axis=1)
    rmse_holdout = root_mean_squared_error(y_holdout, y_ens)
    r2_holdout   = r2_score(y_holdout, y_ens)
    pd.DataFrame([{
        'Ensemble_R2':   r2_holdout,
        'Ensemble_RMSE': rmse_holdout
    }]).to_csv(
        os.path.join(out_dir, f"RF_{crop}_holdout_results.csv"), index=False
    )

    #  # 6. 时间交叉验证改为分块交叉和逐年留一交叉
    # # 6.1 分块交叉验证（GroupKFold 按年份分块）
    # gkf = GroupKFold(n_splits=min(5, len(np.unique(years_pool))))
    # rows_gkf = []
    # for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_pool, y_pool, groups=years_pool), 1):
    #     rf_g = RandomForestRegressor(**best, n_jobs=-1, random_state=0)
    #     rf_g.fit(X_pool.iloc[tr_idx], y_pool.iloc[tr_idx])
    #     y_pred = rf_g.predict(X_pool.iloc[va_idx])
    #     rows_gkf.append({
    #         'Fold': fold_idx,
    #         'BlockCV_R2':   r2_score(y_pool.iloc[va_idx], y_pred),
    #         'BlockCV_RMSE': root_mean_squared_error(y_pool.iloc[va_idx], y_pred)
    #     })
    # pd.DataFrame(rows_gkf).to_csv(
    #     os.path.join(out_dir, "GroupKFold_per_fold.csv"), index=False
    # )
    #
    # # 6.2 逐年留一交叉（跳过最早年份）
    # unique_years = sorted(np.unique(years))
    # rows_lyo = []
    # for yr in unique_years[1:]:
    #     mask_tr = years_pool < yr
    #     mask_va = years_pool == yr
    #     if mask_va.sum() == 0:
    #         continue
    #     rf_l = RandomForestRegressor(**best, n_jobs=-1, random_state=0)
    #     rf_l.fit(X_pool[mask_tr], y_pool[mask_tr])
    #     y_pred = rf_l.predict(X_pool[mask_va])
    #     rows_lyo.append({
    #         'Year': yr,
    #         'LYO_R2':   r2_score(y_pool[mask_va], y_pred),
    #         'LYO_RMSE': root_mean_squared_error(y_pool[mask_va], y_pred)
    #     })
    # pd.DataFrame(rows_lyo).to_csv(
    #     os.path.join(out_dir, "LYO_per_year.csv"), index=False
    # )
    #
    # # 汇总 timeblock CV
    # df_gkf = pd.read_csv(os.path.join(out_dir, "GroupKFold_per_fold.csv"))
    # df_lyo = pd.read_csv(os.path.join(out_dir, "LYO_per_year.csv"))
    # summary = {
    #     'BlockCV_R2_mean':  [df_gkf['BlockCV_R2'].mean()],
    #     'BlockCV_R2_std':   [df_gkf['BlockCV_R2'].std()],
    #     'BlockCV_RMSE_mean':[df_gkf['BlockCV_RMSE'].mean()],
    #     'BlockCV_RMSE_std': [df_gkf['BlockCV_RMSE'].std()],
    #     'LYO_R2_mean':      [df_lyo['LYO_R2'].mean()],
    #     'LYO_R2_std':       [df_lyo['LYO_R2'].std()],
    #     'LYO_RMSE_mean':    [df_lyo['LYO_RMSE'].mean()],
    #     'LYO_RMSE_std':     [df_lyo['LYO_RMSE'].std()]
    # }
    # pd.DataFrame(summary).to_csv(
    #     os.path.join(out_dir, "timeblock_cv_summary.csv"), index=False
    # )
    # print(f"  • Finished {crop.upper()}")