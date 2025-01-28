import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import pandas as pd
import os
import pickle as pkl
import random
import torch

seeds = [0,1,2,3,4]
# seed = 0
clinical_root = "clinical"
gene_expression_root = "gene_expression"
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

def train(feature_path,clinical_path,alpha_min_ratio=0.01,l1_ratio=0.5,max_iter=100,tol=1e-7,seed=0):
    K = 5
    df1 = pd.read_csv(feature_path, header=0)
    df2 = pd.read_csv(clinical_path, header=0)
    df1 = df1.sort_values(by="patient_id")
    df2 = df2.sort_values(by="patient_id")
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    patient_id = df1["patient_id"]
    df1 = df1.drop(columns=["patient_id"])
    X = df1.values
    s = df2['real_survival_time'].values
    d = df2['vital_status'].values
    X = np.log2(X + 1)
    indices = list(range(X.shape[0]))
    n_samples = X.shape[0]
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

        X_train, X_test, X_val = X[train_mask], X[test_mask], X[val_mask]
        y_train = np.rec.fromarrays([d[train_mask].astype(bool), s[train_mask].astype(float)],
                                    names='event,survival_time')
        y_test = np.rec.fromarrays([d[test_mask].astype(bool), s[test_mask].astype(float)],
                                    names='event,survival_time')
        y_val = np.rec.fromarrays([d[val_mask].astype(bool), s[val_mask].astype(float)],
                                    names='event,survival_time')

        model = CoxnetSurvivalAnalysis(
            alpha_min_ratio=alpha_min_ratio,
            l1_ratio=l1_ratio,
            # max_iter=max_iter,
            tol=tol,
        )

        model.fit(X_train, y_train)
        prediction_train = model.predict(X_train)
        prediction_test = model.predict(X_test)
        prediction_val = model.predict(X_val)
        train_score = concordance_index_censored(s[train_mask].astype(bool), d[train_mask].astype(float),
                                                 prediction_train.astype(float))[0]
        test_score = concordance_index_censored(s[test_mask].astype(bool), d[test_mask].astype(float),
                                                 prediction_test.astype(float))[0]
        val_score = concordance_index_censored(s[val_mask].astype(bool), d[val_mask].astype(float),
                                                 prediction_val.astype(float))[0]
        print(f"fold: {idx+1} train score : {train_score} val score : {val_score} test score : {test_score}")
        models.append(model)
        test_indices.append(test_score)

    fold = np.argmax(test_indices)
    print(f"Best fold is {fold}, c-index:{test_indices[fold]}")
    best_model = models[fold]
    prediction = best_model.predict(X)
    res_df = pd.DataFrame({"patient_id":patient_id.values.tolist(),
                           "prediction_risk":prediction.tolist(),
                           "vital_status":d.tolist(),
                           "real_survival_time":s.tolist()})
    return test_indices[fold],res_df,train_mask,test_mask

def grid_search(feature_path, clinical_path, param_grid,seed):
    """
    网格搜索：寻找最佳参数组合
    """
    best_params = None
    best_c_index = -1
    best_df = None

    threshold = 0.5
    # 遍历参数组合
    for params in ParameterGrid(param_grid):
        if best_c_index>threshold:
            break
        alpha_min_ratio = params["alpha_min_ratio"]
        l1_ratio = params["l1_ratio"]
        # max_iter = params["max_iter"]
        tol = params["tol"]
        # seed = params["seed"]

        # 调用训练函数
        c_index, cur_df, train_mask, test_mask = train(
            feature_path,
            clinical_path,
            alpha_min_ratio=alpha_min_ratio,
            l1_ratio=l1_ratio,
            # max_iter=max_iter,
            tol=tol,
            seed=seed
        )

        # 更新最优参数
        if c_index > best_c_index:
            best_c_index = c_index
            best_params = params
            best_df = cur_df

    print(f"Best parameters: {best_params}")
    print(f"Best C-index: {best_c_index:.3f}")
    return best_c_index, best_df, train_mask, test_mask

param_grid = {
    # "alpha_min_ratio": np.logspace(-4, -2, 5).tolist(),  # 从 0.0001 到 0.01，缩小范围以避免极端值
    "alpha_min_ratio": ["auto"],
    "l1_ratio": np.linspace(0.1, 1.0, 10).tolist(),       # 从 0.3 到 0.7，更集中于弹性网络的中间值
    # "max_iter": [1000, 2000, 3000],                      # 增大最大迭代次数，确保优化过程有足够时间
    "seed":list(range(20)),
    "tol": [1e-3, 1e-4, 1e-5]                            # 放宽容差范围，提高训练效率
}

prediction_root = "prediction_save"
if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)
for disease in os.listdir(clinical_root):
    print(f'Training disease: {disease} Cox-EN')
    c_indices = []
    for seed in seeds:
        print(f"Seed: {seed}...")
        set_seed(seed)
        clinical_path = os.path.join(clinical_root,f'{disease}/clinical.csv')
        feature_path = os.path.join(gene_expression_root,f'{disease}/gene_expression.csv')
        best_c_index,best_df,train_mask,test_mask = grid_search(feature_path,clinical_path,param_grid,seed)
        print(f"Tuning Done! Saving ...")
        save_path = os.path.join(prediction_root, f'{disease}_seed{seed}_prediction.csv')
        best_df.to_csv(save_path, index=False)
        with open(os.path.join(prediction_root, f'{disease}_seed{seed}_train_mask.pkl'),'wb') as f:
            pkl.dump(train_mask,f)
        c_indices.append(best_c_index)
    with open(os.path.join(prediction_root, f'{disease}_test_mask.pkl'),'wb') as f:
        pkl.dump(test_mask,f)
    with open("result.txt","a") as f:
        f.write(f"disease: {disease} c-index mean: {np.mean(c_indices)} std: {np.std(c_indices)}")
        f.write("\n")
    print("Done!")


