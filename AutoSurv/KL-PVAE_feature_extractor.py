import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Standard
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # Return mean and standard

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Mean Square Erro
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta * KLD

class CancerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx, 1:].values.astype(np.float32), dtype=torch.float32)  # Ignoring patient_id column

data_dir = 'merged_data(miRNA_and_mRNA)'
cancer_types = os.listdir(data_dir)

# Hyper Parameters (KL-annealing learning strategy)
learning_rate = 1e-6
num_epochs = 5
M = 2
R = 0.3
steps_per_cycle = 3
patience = 3

for cancer in cancer_types:
    cancer_dir = os.path.join(data_dir, cancer)
    merged_file_path = os.path.join(cancer_dir, 'merged_data.csv')

    if os.path.exists(merged_file_path):
        df = pd.read_csv(merged_file_path)

        # Drop patient_id column
        df = df.drop(columns=['patient_id'])

        # Deal with Patient ID column
        df.rename(columns={'Patient ID': 'patient_id'}, inplace=True)

        df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: np.log2(x + 1))
        cancer_dataset = CancerDataset(df)
        train_loader = DataLoader(cancer_dataset, batch_size=128, shuffle=True)

        input_dim = df.shape[1] - 1
        total_steps = num_epochs * len(train_loader)

        model = VAE(input_dim=input_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        best_loss = torch.inf
        cnt = 0
        for t in range(total_steps):
            if cnt >= patience:
                break

            cycle = t // steps_per_cycle
            step_in_cycle = t % steps_per_cycle

            if cycle < M:
                if step_in_cycle < R * steps_per_cycle:
                    beta = step_in_cycle / (R * steps_per_cycle)  # Annealing
                else:
                    beta = 1.0  # Fixing
            else:
                beta = 1.0

            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar, beta)
                loss.backward()
                optimizer.step()
            if best_loss <= loss.item():
                cnt += 1
            else:
                cnt = 0
                best_loss = loss.item()
                # Saving Feature Matrix
                with torch.no_grad():
                    _, mu, _ = model(torch.tensor(df.values[:, 1:].astype(np.float32), dtype=torch.float32))
                    features_df = pd.DataFrame(mu.numpy(), columns=[f'feature_{i + 1}' for i in range(mu.shape[1])])
                    features_df['patient_id'] = df['patient_id'].values  # 添加 patient_id 列
                    output_file_path = os.path.join(cancer_dir, 'vae_features.csv')
                    features_df.to_csv(output_file_path, index=False)
                    print(f'Saved VAE features for {cancer} to {output_file_path}')

            if t % 3 == 0:
                print(f'Cancer Type: {cancer}, Step {t}, Loss: {loss.item()}')

print("Training completed for all cancer types.")
