import pandas as pd
import pickle as pkl

df = pd.read_csv("LUSC_prediction.csv",header=0)
with open("LUSC_train_mask.pkl",'rb') as f:
    train_mask = pkl.load(f)
with open("LUSC_test_mask.pkl",'rb') as f:
    test_mask = pkl.load(f)

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

risk_pred = df["prediction_risk"].values
survival_time = df["real_survival_time"].values
events = df["vital_status"].values

print(c_index(risk_pred[test_mask],survival_time[test_mask], events[test_mask]))