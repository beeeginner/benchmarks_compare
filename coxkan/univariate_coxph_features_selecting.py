import os
import pandas as pd
from lifelines import CoxPHFitter
import numpy as np

gene_root = "gene_expression"
clinical_root = "clinical"

diseases = os.listdir(gene_root)
for disease in diseases:
    print(f'Processing {disease}...')

    df_gene_expression = pd.read_csv(os.path.join(gene_root, disease, 'gene_expression.csv'), header=0)
    df_clinical = pd.read_csv(os.path.join(clinical_root, disease, "clinical.csv"), header=0)

    df_gene_expression = df_gene_expression.sort_values(by='patient_id')
    patient_ids = df_gene_expression['patient_id']
    df_gene_expression = df_gene_expression.drop(columns=['patient_id'])

    df_gene_expression = df_gene_expression.astype(np.float32)
    df_gene_expression = np.log2(df_gene_expression + 1)

    df_clinical = df_clinical[['vital_status', 'real_survival_time']]
    df_merged = pd.concat([df_gene_expression, df_clinical], axis=1)

    low_variance_columns = df_merged.var() < 1e-5
    df_merged = df_merged.loc[:, ~low_variance_columns]

    # Univariate Cox-PH Analysis
    p_values = {}
    for column in df_merged.columns:
        data = df_merged[['real_survival_time', 'vital_status', column]].dropna()
        if len(data) > 0:  
            cph = CoxPHFitter()
            try:
                cph.fit(data, duration_col='real_survival_time', event_col='vital_status')
                p_values[column] = cph.summary['p'].values[0]  # Extracting P-values
            except Exception as e:
                print(f"Error fitting model for {column}: {e}")

    # Constructing a DataFrame including feature and p-values 
    p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-Value'])

    # selecting features by P-value<0.05
    selected_features = p_values_df[p_values_df['P-Value'] < 0.05]['Feature']

    selected_expression = pd.concat([patient_ids, df_gene_expression[selected_features].reset_index(drop=True)], axis=1)
    selected_expression.to_csv(os.path.join(gene_root, disease, 'selected_expression.csv'), index=False)

    print(f'Selected features saved for {disease}:')
    print(f"Total features: {len(selected_features.tolist())}")
