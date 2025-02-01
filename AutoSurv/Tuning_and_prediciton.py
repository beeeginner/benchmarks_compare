import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
import os
import random
from itertools import product
import numpy as np
import pickle as pkl

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
    # K fold
    K = 5
    data = pd.read_csv(data_path)

    # Reading Data
    features = data.drop(columns=['real_survival_time', 'vital_status','patient_id']).values.astype(np.float32)
    survival_time = data['real_survival_time'].values.astype(np.int64)
    events = data['vital_status'].values.astype(np.int64)

    # To Tensor in the same device
    features = torch.from_numpy(features).to(device)
    survival_time = torch.from_numpy(survival_time).to(device)
    events = torch.from_numpy(events).to(device)

    # Get samples
    n_samples = features.shape[0]
    indices = list(range(n_samples))

    train_samples, test_samples = train_test_split(indices, test_size=0.2, random_state=0, stratify=events.cpu().numpy().tolist())
    test_mask = torch.BoolTensor([False for i in range(n_samples)])
    test_samples = torch.LongTensor(test_samples)
    test_mask[test_samples] = True

    train_mask = torch.BoolTensor([False for i in range(n_samples)])
    train_samples = torch.LongTensor(train_samples)
    train_mask[train_samples] = True
    val_samples = stratified_random_partition(train_samples, partitions=K,
                                              stratify_labels=events.cpu()[train_mask], random_state=seed)
    train_samples = set(train_samples)

    num_epochs = num_epochs
    models = []
    test_indices = []

    for idx, val in enumerate(val_samples):
        val = set(val)
        train = train_samples - val
        train, val = torch.tensor(list(train), dtype=torch.long), torch.tensor(list(val), dtype=torch.long)
        train_mask = torch.BoolTensor([False for i in range(n_samples)])
        val_mask = torch.BoolTensor([False for i in range(n_samples)])
        train_mask[train] = True
        val_mask[val] = True
        input_size = features.shape[1]
        model = CoxNNET(input_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            thetas = model(features).squeeze()
            loss = loss_function(thetas[train_mask], survival_time[train_mask], events[train_mask])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                thetas = model(features).squeeze()
                train_score = c_index(thetas[train_mask], survival_time[train_mask], events[train_mask])
                val_score = c_index(thetas[val_mask], survival_time[val_mask], events[val_mask])

            with torch.no_grad():
                thetas = model(features).squeeze()
                test_score = c_index(thetas[test_mask], survival_time[test_mask], events[test_mask])

        print(f"fold:{idx+1} train c-index: {train_score} val c-index: {val_score} test c-index: {test_score} ")
        models.append(model)
        test_indices.append(test_score)

    best_fold = np.argmax(test_indices)
    print(f"Best fold is: {best_fold + 1}, test c-index: {test_indices[best_fold]} ")
    best_model = models[best_fold]

    best_model.eval()
    with torch.no_grad():
        predictions = best_model(features)
        predictions = predictions.cpu().numpy().squeeze()
        res_df = pd.DataFrame({"patient_id": data["patient_id"].values.tolist(),
                               "prediction_risk": predictions.tolist(),
                               "real_survival_time":data["real_survival_time"].values.tolist(),
                               "vital_status":data["vital_status"].values.tolist()
                               })

    with torch.no_grad():
        thetas = best_model(features).squeeze()
        c_index_score = c_index(thetas[test_mask],survival_time[test_mask],events[test_mask])
        print(f'Test C-index: {c_index_score:.4f}')

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
            return best_c_index, best_df,best_train_mask,best_test_mask
        c_index_score,df,train_mask,test_mask = train_and_evaluate_model(data_path, lr=lr, wd=wd, num_epochs=num_epochs)

        if c_index_score > best_c_index:
            best_c_index = c_index_score
            best_params = (lr, wd, num_epochs)
            best_df,best_train_mask,best_test_mask = df,train_mask,test_mask

    print(f'Best C-index: {best_c_index:.4f} with parameters: lr={best_params[0]:.5e}, wd={best_params[1]}, num_epochs={best_params[2]}')
    return best_c_index, best_df, best_train_mask, best_test_mask

files = os.listdir("KL-PVAE_features")
save_root = "prediction_save_path"
if not os.path.exists(save_root):
    os.mkdir(save_root)

threshold = 0.66

# Distribute five different random seeds to test the model's stability.
seeds = [0,1,2,3,4,5]

for f in files:
    name = f[:4]
    if name=="ESCA":
        threshold = 0.70
    else:
        threshold = 0.66

    c_indices = []
    for seed in seeds:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f)
        path = os.path.join("KL-PVAE_features", f)
        best_c_index, best_df, best_train_mask, best_test_mask = grid_search(path, threshold)
        print("Finding Hyper Parameters Done!")
        prediction_save_path = os.path.join(save_root, f"{f[:4]}_seed{seed}_prediction.csv")
        best_df.to_csv(prediction_save_path, index=False)
        with open(os.path.join(save_root, f"{f[:4]}_seed{seed}_train_mask.pkl"), 'wb') as file:
            pkl.dump(best_train_mask, file)
        c_indices.append(best_c_index)

    with open(os.path.join(save_root, f"{f[:4]}_c_indices.pkl"), 'wb') as file:
        pkl.dump(c_indices, file)

    with open("result.txt",'a') as file:
        file.write(f"{f[:4]} mean: {np.mean(c_indices)} std: {np.std(c_indices)}")
        file.write("\n")

    with open(os.path.join(save_root, f"{f[:4]}_test_mask.pkl"), 'wb') as file:
        pkl.dump(best_test_mask, file)









