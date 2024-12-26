# benchmarks_compare
Based on the code and experimental procedures provided in the papers, we ran four different cancer prognosis models on seven TCGA datasets.

### Seven TCGA Datasets Used

The seven TCGA datasets used in this study can be downloaded from [https://portal.gdc.cancer.gov](https://portal.gdc.cancer.gov). The multi-omics data used by different methods vary but all come from the same patients.

### prediction results 

The prediction results have also been uploaded and can be found in the corresponding folder.
```
AutoSurv/prediction_save_path
CoxEN/prediction_save
CoxAE/prediction_save
coxkan/prediction_output
Cox-sage/prediction_save.zip
```

#### AutoSurv

Using KL-PMVAE to extract features from miRNA and high-dimensional gene expression data. The corresponding code is provided, and to facilitate reproducibility, we have uploaded the extracted features. You can directly perform hyperparameter tuning, training, and prediction with a single command:

```bash
python Tuning_and_prediciton.py
```

Additionally, the prediction results have been uploaded and can be found in the corresponding folder.

#### CoxKAN

Feature selection was performed based on univariate CoxPH analysis. We then used the Python package provided in the corresponding paper to perform detailed hyperparameter tuning. Due to the long runtime, we have recorded the optimal hyperparameters in a file for easier reproducibility. You can use the following command:

```bash
python reproduce_all.py
```

Similarly, the prediction results have been uploaded and can be found in the corresponding folder.

#### CoxEN

Based on the elastic net-regularized Cox proportional hazards model, we implemented this method using the Python library `scikit-survival`. The data used for this method are the expression data of protein-coding genes. Run the following command:

```bash
python COXEN.py
```

The prediction results have also been uploaded and can be found in the corresponding folder.

#### CoxAE

We used an autoencoder to extract multi-omics features and then performed CoxPH analysis. To simplify reproducibility, we have uploaded the extracted features, allowing you to directly perform hyperparameter tuning, training, and prediction with a single command:

```bash
python COXAE.py
```


