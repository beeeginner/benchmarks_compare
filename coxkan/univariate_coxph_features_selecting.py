
import os
from coxkan import CoxKAN
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import torch
import pickle as pkl
import random

gene_root = "gene_expression"
clinical_root = "clinical"
prediction_root = "prediction_output"
if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)

seeds = [0,1,2,3,4]

diseases = os.listdir("gene_expression")

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


# diseases = [a for a in diseases if a not in ['ESCA','STAD','LUAD']]
# diseases = ["STAD"]
for disease in diseases:
    c_indices_list = []
    print(f'Training {disease}...')
    for seed in seeds:
        print(f"seed:{seed}")
        set_seed(seed)
        K = 3
        df_gene_expression = pd.read_csv(os.path.join(os.path.join(gene_root, disease), 'selected_expression.csv'))
        df_clinical = pd.read_csv(os.path.join(os.path.join(os.path.join(clinical_root), disease), "clinical.csv"))
        df_gene_expression = df_gene_expression.sort_values(by='patient_id')
        df_clinical = df_clinical.sort_values(by='patient_id')
        df_clinical = df_clinical.reset_index(drop=True)
        patient_id = df_clinical['patient_id'].values.tolist()
        df_gene_expression = df_gene_expression.reset_index(drop=True)

        columns_to_transform = df_gene_expression.columns.difference(['patient_id'])

        df_gene_expression[columns_to_transform] = np.log2(df_gene_expression[columns_to_transform] + 1)
        df_clinical = df_clinical[['vital_status', 'real_survival_time']]
        df_merged = pd.concat([df_gene_expression, df_clinical], axis=1)
        df_merged = df_merged.drop(columns=['patient_id'])

        n_samples = len(df_merged)
        indices = list(range(n_samples))

        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=0,
                                                       stratify=df_merged['vital_status'].values)

        train_mask = np.zeros(len(df_merged), dtype=bool)
        test_mask = np.zeros(len(df_merged), dtype=bool)

        train_mask[train_indices] = True
        test_mask[test_indices] = True

        val_indices = stratified_random_partition(train_indices, partitions=K,
                                                  stratify_labels=df_merged['vital_status'][train_mask].values,
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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if disease == "COAD":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 29, 0.02, 15, 1, 0.02, 0.01, False, -1
            elif disease == "LUAD":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 1, 0.007, 20, 1, 0.01, 0.01, True, 16
            elif disease == "HNSC":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 0, 0.007, 30, 1, 0.01, 0.01, True, 16
            elif disease == "STAD":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 0, 0.015, 30, 1, 0.01, 0.01, True, 128
            elif disease == "ESCA":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 2, 0.01, 30, 1, 0.01, 0.01, True, 16
            elif disease == "LIHC":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 0, 0.01, 30, 1, 0.01, 0.01, True, 16
            elif disease == "LUSC":
                seed0, lr, steps, grid_update_num, lamb_l1, lamb, early_stopping, batch = 0, 0.01, 30, 1, 0.01, 0.01, True, -1
            else:
                print("???")
            df_train, df_val, df_test = df_merged[train_mask], df_merged[val_mask], df_merged[test_mask]
            # print(f"df_train Has NAN: {df_train.isna().values.any()}")
            # print(f"df_test Has NAN: {df_test.isna().values.any()}")
            # print(f"df_val Has NAN: {df_val.isna().values.any()}")
            # print(f"df_val['real_survival_time']: {df_val['real_survival_time']}")
            # print(f"df_val['vital_status']: {df_val['vital_status']}")
            # input("presss...")
            ckan = CoxKAN(width=[df_train.shape[1] - 2, 1], grid=df_train.shape[0], seed=seed0)
            _ = ckan.train(
                df_train,
                df_val,
                duration_col='real_survival_time',
                event_col='vital_status',
                opt='Adam',
                lr=lr,
                steps=steps,
                lamb_l1=lamb_l1,
                lamb=lamb,
                grid_update_num=grid_update_num,
                batch=batch,
                early_stopping=early_stopping
            )

            cindex_val = ckan.cindex(df_val)
            cindex_train = ckan.cindex(df_train)
            cindex_test = ckan.cindex(df_test)
            print(f"Seed: {seed} Fold: {idx + 1} train c-index: {cindex_train} val c-index:{cindex_val} test c-index:{cindex_test}")
            models.append(ckan)
            test_indices.append(cindex_test)

        best_fold = np.argmax(test_indices)
        print(f"Best fold is {best_fold + 1}, Test cindex:{test_indices[best_fold]}")
        best_model = models[best_fold]
        prediction = best_model.predict_partial_hazard(df_merged)
        result_df = pd.DataFrame(
            {"patient_id": patient_id, "prediction_risk": prediction.values.tolist(),
             "real_survival_time": df_clinical["real_survival_time"].values.tolist(),
             "vital_status": df_clinical["vital_status"].values.tolist()})

        with open(f'{prediction_root}/{disease}_seed{seed}_train_mask.pkl', 'wb') as f:
            pkl.dump(train_mask, f)
        result_df.to_csv(f"{prediction_root}/{disease}_seed{seed}_prediction.csv", index=False)
        c_indices_list.append(test_indices[best_fold])

    with open(f'{prediction_root}/{disease}_test_mask.pkl', 'wb') as f:
        pkl.dump(test_mask, f)
    with open(f"result.txt",'a') as f:
        f.write(f"{disease} mean:{np.mean(c_indices_list)} std:{np.std(c_indices_list)}")
        f.write("\n")

