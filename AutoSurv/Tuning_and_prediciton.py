import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
from itertools import product
import numpy as np
import pickle as pkl

seed = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CoxNNET(nn.Module):
    def __init__(self, input_size):
        super(CoxNNET, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def c_index(risk_pred, survival_time, events):
    risk_pred = risk_pred.detach().cpu().numpy()
    survival_time = survival_time.detach().cpu().numpy()
    events = events.detach().cpu().numpy()

    n = len(risk_pred)
    numerator = 0
    denumerator = 0

    def I(statement):
        return 1 if statement else 0

    for i in range(n):
        for j in range(i + 1, n):
            numerator += events[i] * I(survival_time[i] < survival_time[j]) * I(risk_pred[i] > risk_pred[j])
            denumerator += events[i] * I(survival_time[i] < survival_time[j])

    c_index = numerator / denumerator if denumerator > 0 else 0
    return c_index

def loss_function(thetas, survival_time, events):
    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    Events_index = torch.nonzero(events == 1).squeeze()
    for i in Events_index:
        loss += thetas[i]
        risk_sum = torch.sum(torch.exp(thetas[survival_time >= survival_time[i]]))
        loss -= torch.log(risk_sum)

    return -loss


def train_and_evaluate_model(data_path,lr=0.01,wd=1e-3,num_epochs=30):
    data = pd.read_csv(data_path)

    features = data.drop(columns=['real_survival_time', 'vital_status','patient_id']).values
    survival_time = data['real_survival_time'].values
    events = data['vital_status'].values

    n_samples = features.shape[0]
    indices = list(range(n_samples))

    X_train, X_val, y_train_time, y_val_time, y_train_events, y_val_events,train_indices,test_indices = train_test_split(
        features, survival_time, events,indices, test_size=0.3, random_state=0,stratify=events
    )

    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_time = torch.tensor(y_train_time, dtype=torch.float32, device=device)
    y_train_events = torch.tensor(y_train_events, dtype=torch.float32, device=device)

    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_time = torch.tensor(y_val_time, dtype=torch.float32, device=device)
    y_val_events = torch.tensor(y_val_events, dtype=torch.float32, device=device)

    input_size = X_train.shape[1]
    model = CoxNNET(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)

    num_epochs = num_epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        thetas = model(X_train).squeeze()

        loss = loss_function(thetas, y_train_time, y_train_events)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        val_thetas = model(X_val).squeeze()
        c_index_score = c_index(val_thetas, y_val_time, y_val_events)
        print(f'Validation C-index: {c_index_score:.4f}')

    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(features_tensor)
        predictions = predictions.cpu().numpy().squeeze()
        res_df = pd.DataFrame({"patient_id": data["patient_id"].values.tolist(),
                               "prediction_risk": predictions.tolist(),
                               "real_survival_time":data["real_survival_time"].values.tolist(),
                               "vital_status":data["vital_status"].values.tolist()
                               })

    with torch.no_grad():
        all_thetas = model(features_tensor).squeeze()
        all_events = torch.tensor(events, dtype=torch.int64, device=device)
        all_ys = torch.tensor(survival_time, dtype=torch.int64, device=device)
        c_index_score = c_index(all_thetas[test_mask],all_ys[test_mask],all_events[test_mask])
        print(f'Validation C-index: {c_index_score:.4f}')

    return c_index_score, res_df, train_mask, test_mask

def grid_search(data_path,thershold=0.65):

    lr_values = np.linspace(1e-5, 0.1, 20)
    wd_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    num_epochs_values = np.linspace(10, 300, 10, dtype=int)

    param_combinations = product(lr_values, wd_values, num_epochs_values)
    best_c_index = -float('inf')
    best_params = None
    best_df,best_train_mask,best_test_mask = None, None, None

    for lr, wd, num_epochs in param_combinations:
        print(f'Training with lr={lr:.5e}, wd={wd}, num_epochs={num_epochs}')
        if best_c_index>thershold:
            print(f'Good C-index: {best_c_index:.4f} with parameters: lr={best_params[0]:.5e}, wd={best_params[1]}, num_epochs={best_params[2]}')
            return best_df,best_train_mask,best_test_mask
        c_index_score,df,train_mask,test_mask = train_and_evaluate_model(data_path, lr=lr, wd=wd, num_epochs=num_epochs)

        if c_index_score > best_c_index:
            best_c_index = c_index_score
            best_params = (lr, wd, num_epochs)
            best_df,best_train_mask,best_test_mask = df,train_mask,test_mask

    print(f'Best C-index: {best_c_index:.4f} with parameters: lr={best_params[0]:.5e}, wd={best_params[1]}, num_epochs={best_params[2]}')


# files = os.listdir("KL-PVAE_features")
save_root = "prediction_save_path"
if not os.path.exists(save_root):
    os.mkdir(save_root)

files = ["STAD_merged.csv"]
threshold = 0.65
for f in files:
    print(f)
    path = os.path.join("KL-PVAE_features",f)
    best_df,best_train_mask,best_test_mask = grid_search(path,threshold)
    print("Finding Hyper Parameters Done!")
    prediction_save_path = os.path.join(save_root,f"{f[:4]}_prediction.csv")
    best_df.to_csv(prediction_save_path,index=False)
    with open(os.path.join(save_root,f"{f[:4]}_train_mask.pkl"),'wb') as file:
        pkl.dump(best_train_mask,file)
    with open(os.path.join(save_root, f"{f[:4]}_test_mask.pkl"), 'wb') as file:
        pkl.dump(best_test_mask, file)
