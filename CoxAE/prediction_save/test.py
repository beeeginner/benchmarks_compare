import pandas as pd
import pickle as pkl
import os

def c_index(risk_pred, survival_time, events):

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

diseases = os.listdir("../gene_expression")
for d in diseases:
    file = f"{d}_prediction.csv"
    train_mask_path = f"{d}_train_mask.pkl"
    test_mask_path = f"{d}_test_mask.pkl"
    df = pd.read_csv(file, header=0)
    with open(train_mask_path, 'rb') as f:
        train_mask = pkl.load(f)
    with open(test_mask_path, 'rb') as f:
        test_mask = pkl.load(f)
    risk_pred = df["prediction_risk"].values
    survival_time = df["real_survival_time"].values
    events = df["vital_status"].values
    result = c_index(risk_pred[test_mask], survival_time[test_mask], events[test_mask])
    print(f"{d} cindex: {result}")
