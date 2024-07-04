import gc
import cudf
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
import logging
import optuna

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.max_columns = None
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

logging.info("Libraries loaded successfully")

# All usable features
features = [f'state_t_{i}' for i in range(60)] + [f'state_q0001_{i}' for i in range(60)] + [f'state_q0002_{i}' for i in range(60)] + \
           [f'state_q0003_{i}' for i in range(60)] + [f'state_u_{i}' for i in range(60)] + [f'state_v_{i}' for i in range(60)] + \
           ['state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS', 'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND',
            'pbuf_ozone_0', 'pbuf_ozone_1', 'pbuf_ozone_2', 'pbuf_ozone_3', 'pbuf_ozone_4', 'pbuf_ozone_5', 'pbuf_ozone_6', 'pbuf_ozone_7', 'pbuf_ozone_8', 'pbuf_ozone_9', 'pbuf_ozone_10', 'pbuf_ozone_11', 'pbuf_ozone_12', 'pbuf_ozone_13', 'pbuf_ozone_14', 'pbuf_ozone_15', 'pbuf_ozone_16', 'pbuf_ozone_17', 'pbuf_ozone_18', 'pbuf_ozone_19', 'pbuf_ozone_20', 'pbuf_ozone_21', 'pbuf_ozone_22', 'pbuf_ozone_23', 'pbuf_ozone_24', 'pbuf_ozone_25', 'pbuf_ozone_26', 'pbuf_ozone_27', 'pbuf_ozone_28', 'pbuf_ozone_29', 'pbuf_ozone_30', 'pbuf_ozone_31', 'pbuf_ozone_32', 'pbuf_ozone_33', 'pbuf_ozone_34', 'pbuf_ozone_35', 'pbuf_ozone_36', 'pbuf_ozone_37', 'pbuf_ozone_38', 'pbuf_ozone_39', 'pbuf_ozone_40', 'pbuf_ozone_41', 'pbuf_ozone_42', 'pbuf_ozone_43', 'pbuf_ozone_44', 'pbuf_ozone_45', 'pbuf_ozone_46', 'pbuf_ozone_47', 'pbuf_ozone_48', 'pbuf_ozone_49', 'pbuf_ozone_50', 'pbuf_ozone_51', 'pbuf_ozone_52', 'pbuf_ozone_53', 'pbuf_ozone_54', 'pbuf_ozone_55', 'pbuf_ozone_56', 'pbuf_ozone_57', 'pbuf_ozone_58', 'pbuf_ozone_59',
            'pbuf_CH4_0', 'pbuf_CH4_1', 'pbuf_CH4_2', 'pbuf_CH4_3', 'pbuf_CH4_4', 'pbuf_CH4_5', 'pbuf_CH4_6', 'pbuf_CH4_7', 'pbuf_CH4_8', 'pbuf_CH4_9', 'pbuf_CH4_10', 'pbuf_CH4_11', 'pbuf_CH4_12', 'pbuf_CH4_13', 'pbuf_CH4_14', 'pbuf_CH4_15', 'pbuf_CH4_16', 'pbuf_CH4_17', 'pbuf_CH4_18', 'pbuf_CH4_19', 'pbuf_CH4_20', 'pbuf_CH4_21', 'pbuf_CH4_22', 'pbuf_CH4_23', 'pbuf_CH4_24', 'pbuf_CH4_25', 'pbuf_CH4_26',
            'pbuf_N2O_0', 'pbuf_N2O_1', 'pbuf_N2O_2', 'pbuf_N2O_3', 'pbuf_N2O_4', 'pbuf_N2O_5', 'pbuf_N2O_6', 'pbuf_N2O_7', 'pbuf_N2O_8', 'pbuf_N2O_9', 'pbuf_N2O_10', 'pbuf_N2O_11', 'pbuf_N2O_12', 'pbuf_N2O_13', 'pbuf_N2O_14', 'pbuf_N2O_15', 'pbuf_N2O_16', 'pbuf_N2O_17', 'pbuf_N2O_18', 'pbuf_N2O_19', 'pbuf_N2O_20', 'pbuf_N2O_21', 'pbuf_N2O_22', 'pbuf_N2O_23', 'pbuf_N2O_24', 'pbuf_N2O_25', 'pbuf_N2O_26']

# All the target variables
all_targets = [f'ptend_t_{i}' for i in range(60)] + [f'ptend_q0001_{i}' for i in range(60)] + [f'ptend_q0002_{i}' for i in range(60)] + \
              [f'ptend_q0003_{i}' for i in range(60)] + [f'ptend_u_{i}' for i in range(60)] + [f'ptend_v_{i}' for i in range(60)] + \
              ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

target_unpredictable = ['ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3', 'ptend_q0001_4', 'ptend_q0001_5', 'ptend_q0001_6', 'ptend_q0001_7', 'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10', 'ptend_q0001_11'] + \
                       ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26'] + \
                       ['ptend_q0003_0', 'ptend_q0003_1', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4', 'ptend_q0003_5', 'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8', 'ptend_q0003_9', 'ptend_q0003_10', 'ptend_q0003_11'] + \
                       ['ptend_u_0', 'ptend_u_1', 'ptend_u_2', 'ptend_u_3', 'ptend_u_4', 'ptend_u_5', 'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_u_10', 'ptend_u_11'] + \
                       ['ptend_v_0', 'ptend_v_1', 'ptend_v_2', 'ptend_v_3', 'ptend_v_4', 'ptend_v_5', 'ptend_v_6', 'ptend_v_7', 'ptend_v_8', 'ptend_v_9', 'ptend_v_10', 'ptend_v_11']

targets = [f for f in all_targets if f not in target_unpredictable]

# Compute target weights.
weights = pd.read_csv("/workspace/sample_submission.csv", nrows=1)
del weights['sample_id']
weights = weights.T
weights = weights.to_dict()[0]

logging.info("# Features: %d", len(features))
logging.info("# Targets Labels: %d", len(all_targets))

# Dataset: https://www.kaggle.com/datasets/titericz/leap-dataset-giba
train_files = sorted(glob("/workspace/train_batch/*.parquet"))
test_files = glob("/workspace/test_batch/*.parquet")

logging.info("%d train files, %d test files", len(train_files), len(test_files))

def load_and_combine_parquet(files, batch_size=2):
    combined_df = pd.DataFrame()
    for i in range(0, len(files), batch_size):
        dfs = [pd.read_parquet(file) for file in files[i:i+batch_size]]
        combined_df = pd.concat([combined_df] + dfs, ignore_index=True)
    return combined_df

# Load in smaller batches
train = load_and_combine_parquet(train_files[:6], batch_size=3)
train = cudf.from_pandas(train)  # Send to GPU for speedup
gc.collect()

valid = load_and_combine_parquet(train_files[6:8], batch_size=3)
valid = cudf.from_pandas(valid)  # Send to GPU for speedup
gc.collect()

test = load_and_combine_parquet(test_files, batch_size=3)
test = cudf.from_pandas(test)  # Send to GPU for speedup

logging.info("Train shape: %s, Valid shape: %s, Test shape: %s", train.shape, valid.shape, test.shape)

for i in range(60):
    M = train[f'state_q0002_{i}'].mean()
    S = train[f'state_q0002_{i}'].std()
    train[f'state_q0002_{i}'] = ((train[f'state_q0002_{i}'] - M) / S).astype('float32')
    valid[f'state_q0002_{i}'] = ((valid[f'state_q0002_{i}'] - M) / S).astype('float32')
    test[f'state_q0002_{i}'] = ((test[f'state_q0002_{i}'] - M) / S).astype('float32')
gc.collect()

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor'
    }

    partial_target = []
    for tnum, target in enumerate(all_targets):
        feat = target + '_pred'

        if target not in target_unpredictable:
            M = train[target].mean()
            S = train[target].std() + 1e-6

            es = xgb.callback.EarlyStopping(rounds=30, save_best=False)

            model = xgb.XGBRegressor(**param, objective='reg:squarederror', callbacks=[es])
            model.fit(
                train[features].to_pandas(),
                train[target].to_pandas(),
                eval_set=[(valid[features].to_pandas(), valid[target].to_pandas())],
                verbose=False,
            )

            valid[feat] = model.predict(valid[features].to_pandas())
            score_model = r2_score(valid[target].to_pandas(), valid[feat].to_pandas())

            if score_model <= 0:
                valid[feat] = 0.
                test[target] = 0.
            else:
                test[target] = model.predict(test[features].to_pandas())

            # Ensure best_iteration exists
            if hasattr(model, 'best_iteration'):
                bi = model.best_iteration
            else:
                bi = model.n_estimators

            del model
            gc.collect()
        else:
            valid[feat] = 0.
            valid[target] = 0.
            test[target] = 0.

        partial_target.append(target)
        score0 = r2_score(valid[target].to_pandas(), valid[feat].to_pandas())
        score1 = r2_score(valid[partial_target].to_pandas(), valid[[f + '_pred' for f in partial_target]].to_pandas())
        logging.info(f"{tnum} r2(accum): {score1:.4f} / r2({target}): {score0:.4f},  best iter: {bi}")

    return score1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
logging.info("Best hyperparameters: %s", best_params)

with open('best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

partial_target = []
for tnum, target in enumerate(all_targets):
    feat = target + '_pred'

    if target not in target_unpredictable:
        M = train[target].mean()
        S = train[target].std() + 1e-6

        es = xgb.callback.EarlyStopping(rounds=30, save_best=False)

        model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', callbacks=[es])
        model.fit(
            train[features].to_pandas(),
            train[target].to_pandas(),
            eval_set=[(valid[features].to_pandas(), valid[target].to_pandas())],
            verbose=False,
        )

        valid[feat] = model.predict(valid[features].to_pandas())
        score_model = r2_score(valid[target].to_pandas(), valid[feat].to_pandas())

        if score_model <= 0:
            valid[feat] = 0.
            test[target] = 0.
        else:
            test[target] = model.predict(test[features].to_pandas())

        bi = model.best_iteration
        del model
        gc.collect()
    else:
        valid[feat] = 0.
        valid[target] = 0.
        test[target] = 0.

    partial_target.append(target)
    score0 = r2_score(valid[target].to_pandas(), valid[feat].to_pandas())
    score1 = r2_score(valid[partial_target].to_pandas(), valid[[f + '_pred' for f in partial_target]].to_pandas())
    logging.info(f"{tnum} r2(accum): {score1:.4f} / r2({target}): {score0:.4f},  best iter: {bi}")

score = r2_score(valid[partial_target].to_pandas(), valid[[f + '_pred' for f in partial_target]].to_pandas())
logging.info(score)

score = r2_score(valid[targets].to_pandas(), valid[[f + '_pred' for f in targets]].to_pandas())
logging.info(score)

del train, valid; gc.collect()

submission = pd.read_csv("/workspace/sample_submission.csv")

for col in tqdm(all_targets):
    if (weights[col] > 0) and (col not in target_unpredictable):
        submission[col] = test[col].to_pandas().values
    else:
        submission[col] = 0.

for col in tqdm(target_unpredictable):
    submission[col] = 0.

submission.to_csv('submission.csv', index=False)
submission.head()
