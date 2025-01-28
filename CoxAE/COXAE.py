import torch
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import random
import pickle as pkl

seed = 0
seeds = [0,1,2,3,4]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stratified_random_partition(arr, partitions=10, stratify_labels=None, random_state=None):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if stratify_labels is None:
        raise ValueError(r"stratify_labels must be provided for stratified sampling.")
    skf = StratifiedKFold(n_splits=partitions, shuffle=True, random_state=random_state)

    fold_indices = list(skf.split(arr, stratify_labels))
    groups = [[] for _ in range(partitions)]
    for i, (_, test_indices) in enumerate(fold_indices):
        groups[i] = arr[test_indices].tolist()
    return groups

def train(data_path,alpha=0.1,tol=1e-6,n_iter=100):
    K = 5
    df = pd.read_csv(data_path, header=0)
    s = df['real_survival_time'].values
    d = df['vital_status'].values
    patient_id = df['patient_id'].values
    df = df.drop(columns=['real_survival_time', 'vital_status', 'patient_id'])
    features = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    n_samples = X_scaled.shape[0]
    indices = list(range(n_samples))

    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=0, stratify=d)

    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    val_indices = stratified_random_partition(train_indices, partitions=K,
                                              stratify_labels=d[train_mask],
                                              random_state=seed)
    train_indices = set(train_indices)

    models = []
    test_indices = []

    for idx, val in enumerate(val_indices):
        val = set(val)
        train = train_indices - val
        train, val = np.array(list(train)).astype(np.int64), np.array(list(val)).astype(np.int64)
        train_mask = np.array([False for i in range(n_samples)])
        val_mask = np.array([False for i in range(n_samples)])
        train_mask[train] = True
        val_mask[val] = True

        X_train,X_val,X_test = X_scaled[train_mask],X_scaled[val_mask],X_scaled[test_mask]
        y_train = np.core.records.fromarrays([d[train_mask].astype(bool), s[train_mask]], names='event,time')
        y_val = np.core.records.fromarrays([d[val_mask].astype(bool), s[val_mask]], names='event,time')
        y_test = np.core.records.fromarrays([d[test_mask].astype(bool), s[test_mask]], names='event,time')

        cox_ph = CoxPHSurvivalAnalysis(
            ties="efron",
            alpha=alpha,  # regularization
            tol=tol,
            n_iter=n_iter # tolerance
            # verbose=True
        )
        cox_ph.fit(X_train,y_train)
        train_score = cox_ph.score(X_train, y_train)
        val_score = cox_ph.score(X_val, y_val)
        test_score = cox_ph.score(X_test, y_test)
        print(f"Fold: {idx+1} train score: {train_score} val score: {val_score} test score: {test_score}")
        models.append(cox_ph)
        test_indices.append(test_score)

    fold = np.argmax(test_indices)
    print(f"Best fold is {fold}, c-index: {test_indices[fold]} ")
    best_model = models[fold]
    prediction = best_model.predict(X_scaled)
    res_df = pd.DataFrame({
        "patient_id":patient_id.tolist(),
        "prediction_risk": prediction.squeeze().tolist(),
        "real_survival_time":s.tolist(),
        "vital_status":d.tolist()
    })
    return test_indices[fold],res_df,train_mask,test_mask

def grid_search(data_path, param_grid):

    best_params = None
    best_score = 0
    best_df = None

    threshold = 0.6

    for params in ParameterGrid(param_grid):
        alpha = params['alpha']
        tol = params['tol']
        n_iter = params['n_iter']
        test_score, cur_df, train_mask, test_mask = train(data_path, alpha=alpha, tol=tol,n_iter=n_iter)

        if test_score > best_score:
            best_score = test_score
            best_params = params
            best_df = cur_df

        if best_score > threshold:
            break


    print(f"Best parameters: {best_params}")
    print(f"Best Testing C-index: {best_score:.3f}")
    return best_params, best_score, best_df, train_mask, test_mask

DISEASES = os.listdir("AE_features")
param_grid = {
        "alpha":  np.linspace(0.001, 1.0, 30).tolist(),
        "tol": [1e-5, 1e-6, 1e-7],
        "n_iter": [100,200,300,400,500,600,700,800,900,1000],
    }

prediction_root = "prediction_save"
if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)
for disease in DISEASES:
    print(f"Training disease: {disease}")
    c_indices = []
    for seed in seeds:
        print(f"Seed: {seed}")
        set_seed(seed)
        data_path = os.path.join("AE_features",f'{disease}/feature.csv')
        best_params,best_score,best_df,train_mask,test_mask = grid_search(data_path,param_grid)
        save_path = os.path.join(prediction_root,f'{disease}_seed{seed}_prediction.csv')
        best_df.to_csv(save_path,index=False)
        with open(os.path.join(prediction_root, f'{disease}_seed{seed}_train_mask.pkl'),'wb') as f:
            pkl.dump(train_mask,f)
        c_indices.append(best_score)
    with open(os.path.join(prediction_root, f'{disease}_test_mask.pkl'),'wb') as f:
        pkl.dump(test_mask,f)
    with open(f"result.txt","a") as f:
        f.write(f"disease: {disease} mean score: {np.mean(c_indices)} std: {np.std(c_indices)}")
        f.write("\n")
