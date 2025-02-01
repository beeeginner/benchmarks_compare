import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import random

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "gene_expression"
DISEASES = os.listdir("gene_expression")
save_root = "AE_features"
clinical_root = "clinical"
if not os.path.exists(save_root):
    os.mkdir(save_root)
res = []
for disease in DISEASES:
    print(f'Training AutoEncoder for {disease}...')
    feature_path = root + '/' + disease
    clinical_path = os.path.join(f"{clinical_root}/{disease}", 'clinical.csv')
    feature_path = os.path.join(feature_path, 'gene_expression.csv')
    feature = pd.read_csv(feature_path, header=0)
    feature = feature.sort_values(by="patient_id")
    feature = feature.reset_index(drop=True)
    clinical = pd.read_csv(clinical_path,header=0)
    clinical = clinical.sort_values(by="patient_id")
    clinical = clinical.reset_index(drop=True)
    del feature['patient_id']
    feature = feature.values
    feature = np.log2(feature + 1)
    input_dim = feature.shape[1]
    encoding_dim = 100
    model = Autoencoder(input_dim, encoding_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
    X = torch.tensor(feature, dtype=torch.float32).to(device)
    batch_size = 32
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epochs = 25
    early_stopping_patience = 5
    best_loss = torch.inf
    counter = 0
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item() > best_loss:
            counter += 1
        else:
            counter = 0
            best_loss = loss.item()
        if counter >= early_stopping_patience:
            print(f'early stopping at epoch {epoch + 1}')
            break
    print('Done!')

    def extract_features(model, data_loader):
        features = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(device)
                encoded = model.encoder(inputs)
                features.append(encoded)
        return torch.cat(features, dim=0)

    features = extract_features(model, dataloader).detach().cpu().numpy()
    num_features = features.shape[1]
    column_names = [f'feature_{i + 1}' for i in range(num_features)]
    res_df = pd.DataFrame(features, columns=column_names)
    patient_ids = clinical["patient_id"].values.tolist()
    res_df.insert(0, "patient_id", patient_ids)
    res_df.insert(0, "real_survival_time", clinical["real_survival_time"].values.tolist())
    res_df.insert(0, "vital_status", clinical["vital_status"].values.tolist())
    save_path = os.path.join(save_root,disease)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_df.to_csv(os.path.join(save_path,"feature.csv"),index=False)

print("Done!")
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(features)
    # clinical = pd.read_csv(clinical_path, header=0)
    # s = clinical['real_survival_time'].values
    # d = clinical['vital_status'].values
    # # 划分训练集和测试集
    # X_train, X_test, s_train, s_test, d_train, d_test = train_test_split(X_scaled, s, d, test_size=0.2, stratify=d,
    #                                                                      random_state=seed)
    # # 将事件指示器和生存时间合并成结构化数组
    # y_train = np.core.records.fromarrays([d_train.astype(bool), s_train], names='event,time')
    # # 将事件指示器和生存时间合并成结构化数组
    # y_test = np.core.records.fromarrays([d_test.astype(bool), s_test], names='event,time')
    # # 拟合 Cox-EN 模型
    # cox_en = CoxnetSurvivalAnalysis()
    # cox_en.fit(X_train, y_train)
    # # 输出模型的性能
    # train_score = cox_en.score(X_train, y_train)
    # test_score = cox_en.score(X_test, y_test)
    # print("Training C-index: {:.3f}".format(train_score))
    # print("Testing C-index: {:.3f}".format(test_score))