import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import pandas as pd
import os
import pickle as pkl

seed = 0
clinical_root = "clinical"
gene_expression_root = "gene_expression"

def train(feature_path,clinical_path,alpha_min_ratio=0.01,l1_ratio=0.5,max_iter=100,tol=1e-7,seed=0):
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
    X_train, X_test, s_train, s_test, d_train, d_test, train_indices, test_indices = train_test_split(X, s, d, indices, test_size=0.2, random_state=seed,
                                                                         stratify=d)
    train_mask,test_mask = np.zeros(X.shape[0],dtype=bool),np.zeros(X.shape[0],dtype=bool)
    train_mask[train_indices]=True
    test_mask[test_indices]=True

    y_train = np.rec.fromarrays([d_train.astype(bool), s_train.astype(float)], names='event,survival_time')
    model = CoxnetSurvivalAnalysis(
        alpha_min_ratio=alpha_min_ratio,
        l1_ratio=l1_ratio,
        # max_iter=max_iter,
        tol=tol,
    )
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    prediction = model.predict(X)
    c_index = concordance_index_censored(s_test.astype(bool), d_test.astype(float), scores.astype(float))[0]
    res_df = pd.DataFrame({"patient_id":patient_id.values.tolist(),
                           "prediction_risk":prediction.tolist(),
                           "vital_status":d.tolist(),
                           "real_survival_time":s.tolist()})

    return c_index,res_df,train_mask,test_mask

def grid_search(feature_path, clinical_path, param_grid):

    best_params = None
    best_c_index = -1
    best_df = None

    for params in ParameterGrid(param_grid):
        alpha_min_ratio = params["alpha_min_ratio"]
        l1_ratio = params["l1_ratio"]
        tol = params["tol"]
        seed = params["seed"]

        c_index, cur_df, train_mask, test_mask = train(
            feature_path,
            clinical_path,
            alpha_min_ratio=alpha_min_ratio,
            l1_ratio=l1_ratio,
            # max_iter=max_iter,
            tol=tol,
            seed=seed
        )

        if c_index > best_c_index:
            best_c_index = c_index
            best_params = params
            best_df = cur_df

    print(f"Best parameters: {best_params}")
    print(f"Best C-index: {best_c_index:.3f}")
    return best_c_index, best_df, train_mask, test_mask

param_grid = {
    "alpha_min_ratio": ["auto"],
    "l1_ratio": np.linspace(0.1, 1.0, 10).tolist(),
    "seed":list(range(20)),
    "tol": [1e-3, 1e-4, 1e-5]
}

prediction_root = "prediction_save"
if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)
for disease in os.listdir(clinical_root):
    print(f'Training disease: {disease} Cox-EN')

    clinical_path = os.path.join(clinical_root,f'{disease}/clinical.csv')
    feature_path = os.path.join(gene_expression_root,f'{disease}/gene_expression.csv')
    best_c_index,best_df,train_mask,test_mask = grid_search(feature_path,clinical_path,param_grid)
    print(f"Tuning Done! Saving ...")
    save_path = os.path.join(prediction_root, f'{disease}_prediction.csv')
    best_df.to_csv(save_path, index=False)
    with open(os.path.join(prediction_root, f'{disease}_train_mask.pkl'),'wb') as f:
        pkl.dump(train_mask,f)
    with open(os.path.join(prediction_root, f'{disease}_test_mask.pkl'),'wb') as f:
        pkl.dump(test_mask,f)

    print("Done!")


