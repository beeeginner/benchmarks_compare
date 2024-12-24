import os
from coxkan import CoxKAN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import pickle as pkl

gene_root = "gene_expression"
clinical_root = "clinical"
prediction_root="prediction_output"
if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)

diseases = os.listdir("gene_expression")i
# diseases = ['LUAD']
for disease in diseases:

    print(f'Training {disease}...')
    df_gene_expression = pd.read_csv(os.path.join(os.path.join(gene_root, disease),'selected_expression.csv'))
    df_clinical = pd.read_csv(os.path.join(os.path.join(os.path.join(clinical_root), disease),"clinical.csv"))
    df_gene_expression = df_gene_expression.sort_values(by='patient_id')
    df_clinical = df_clinical.sort_values(by='patient_id')
    df_clinical = df_clinical.reset_index(drop=True)
    df_gene_expression = df_gene_expression.reset_index(drop=True)

    columns_to_transform = df_gene_expression.columns.difference(['patient_id'])

    df_gene_expression[columns_to_transform] = np.log2(df_gene_expression[columns_to_transform] + 1)
    df_clinical = df_clinical[['vital_status','real_survival_time']]
    df_merged = pd.concat([df_gene_expression, df_clinical], axis=1)
    df_merged = df_merged.drop(columns=['patient_id'])
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=0,stratify=df_merged['vital_status'].values)
    train_indices, test_indices = train_test_split(
        np.arange(len(df_merged)),  
        test_size=0.2,             
        random_state=0,            
        stratify=df_merged['vital_status'].values
    )

    train_mask = np.zeros(len(df_merged), dtype=bool)
    test_mask = np.zeros(len(df_merged), dtype=bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if disease=="COAD":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=29,0.02,15,1,0.02,0.01,False,-1
    elif disease=="LUAD":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=1,0.007,20,1,0.01,0.01,True,16
    elif disease=="HNSC":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=0,0.007,30,1,0.01,0.01,True,16
    elif disease=="STAD":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=0,0.015,30,1,0.01,0.01,True,128
    elif disease=="ESCA":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=2,0.01,30,1,0.01,0.01,True,16
    elif disease=="LIHC":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=0,0.01,30,1,0.01,0.01,True,16
    elif disease=="LUSC":
        seed,lr,steps,grid_update_num,lamb_l1,lamb,early_stopping,batch=0,0.01,30,1,0.01,0.01,True,-1
    else:
        print("???")

    ckan = CoxKAN(width=[train_df.shape[1]-2,1], grid=train_df.shape[0],seed=seed)
    _ = ckan.train(
        train_df,
        test_df,
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
    cindex = ckan.cindex(test_df)
    prediction = ckan.predict_partial_hazard(df_merged)
    result_df = pd.DataFrame(
        {"patient_id": df_clinical["patient_id"].values.tolist(), "prediction_risk": prediction.values.tolist(),
         "real_survival_time": df_clinical["real_survival_time"].values.tolist(),
         "vital_status": df_clinical["vital_status"].values.tolist()})
    # result_df = pd.DataFrame({
    # 'patient_id': df_gene_expression['patient_id'].values.tolist(),
    # 'prediction': prediction.values.tolist()
    # })
    print("\nCoxKAN C-Index: ", cindex)
    with open(f'{prediction_root}/{disease}_train_mask.pkl','wb') as f:
        pkl.dump(train_mask,f)
    with open(f'{prediction_root}/{disease}_test_mask.pkl','wb') as f:
        pkl.dump(test_mask,f)
    result_df.to_csv(f"{prediction_root}/{disease}_prediction.csv",index=False)


