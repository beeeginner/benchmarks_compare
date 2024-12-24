import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import random
import pickle as pkl

seed = 0
np.random.seed(seed)
random.seed(seed)

def train(data_path,alpha=0.1,tol=1e-6,n_iter=100):
    df = pd.read_csv(data_path, header=0)
    s = df['real_survival_time'].values
    d = df['vital_status'].values
    patient_id = df['patient_id'].values
    df = df.drop(columns=['real_survival_time', 'vital_status', 'patient_id'])
    features = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    indices = list(range(X_scaled.shape[0]))
    train_mask,test_mask = np.zeros(len(indices),dtype=bool),np.zeros(len(indices),dtype=bool)
    X_train, X_test, s_train, s_test, d_train, d_test, train_indices, test_indices = train_test_split(
        X_scaled, s, d, indices,
        test_size=0.2,
        stratify=d,
        random_state=seed
    )
    train_mask[train_indices]=True
    test_mask[test_indices]=True
    y_train = np.core.records.fromarrays([d_train.astype(bool), s_train], names='event,time')
    # # 将事件指示器和生存时间合并成结构化数组
    y_test = np.core.records.fromarrays([d_test.astype(bool), s_test], names='event,time')
    # # 拟合 CoxPH 模型
    cox_ph = CoxPHSurvivalAnalysis(
        ties="efron",
        alpha=alpha,  # regularization
        tol=tol,
        n_iter=n_iter # tolerance
        # verbose=True
    )
    cox_ph.fit(X_train, y_train)
    # 输出模型的性能
    train_score = cox_ph.score(X_train, y_train)
    test_score = cox_ph.score(X_test, y_test)
    prediction = cox_ph.predict(X_scaled)
    # print("Training C-index: {:.3f}".format(train_score))
    # print("Testing C-index: {:.3f}".format(test_score))
    res_df = pd.DataFrame({
        "patient_id":patient_id.tolist(),
        "prediction_risk": prediction.squeeze().tolist(),
        "real_survival_time":s.tolist(),
        "vital_status":d.tolist()
    })
    return test_score,res_df,train_mask,test_mask

def grid_search(data_path, param_grid):
    """
    网格搜索函数：遍历参数组合，寻找最佳参数
    """
    best_params = None
    best_score = 0
    best_df = None

    # 遍历每组参数组合
    for params in ParameterGrid(param_grid):
        alpha = params['alpha']
        tol = params['tol']
        n_iter = params['n_iter']
        # 调用训练函数
        test_score, cur_df, train_mask, test_mask = train(data_path, alpha=alpha, tol=tol,n_iter=n_iter)

        # 更新最佳参数
        if test_score > best_score:
            best_score = test_score
            best_params = params
            best_df = cur_df

    print(f"Best parameters: {best_params}")
    print(f"Best Testing C-index: {best_score:.3f}")
    return best_params, best_score, best_df, train_mask, test_mask

DISEASES = os.listdir("AE_features")
param_grid = {
        "alpha":  np.linspace(0.001, 1.0, 30).tolist(),  # 正则化强度
        "tol": [1e-5, 1e-6, 1e-7],  # 收敛容忍度
        "n_iter": [100,200,300,400,500,600,700,800,900,1000],  # 最大迭代次数
    }

prediction_root = "prediction_save"
if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)
for disease in DISEASES:
    print(f"Training disease: {disease}")
    data_path = os.path.join("AE_features",f'{disease}/feature.csv')
    best_params,best_score,best_df,train_mask,test_mask = grid_search(data_path,param_grid)
    save_path = os.path.join(prediction_root,f'{disease}_prediction.csv')
    best_df.to_csv(save_path,index=False)
    with open(os.path.join(prediction_root, f'{disease}_train_mask.pkl'),'wb') as f:
        pkl.dump(train_mask,f)
    with open(os.path.join(prediction_root, f'{disease}_test_mask.pkl'),'wb') as f:
        pkl.dump(test_mask,f)
