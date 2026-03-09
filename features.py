import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class electric_nn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
    def forward(self, x):
        return self.net.forward(x)
    


def load_datasets():
    df = pd.read_csv("electric_vehicle_analytics.csv")

    X = df[["Year", "Battery_Capacity_kWh", "Range_km", "Charging_Time_hr"]].values
    y = df["Resale_Value_USD"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val


def create_loaders(X_train, X_val, y_train, y_val):

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    return train_loader, val_loader
    
    
def neuralNetworkTrain(X_train, X_val, y_train, y_val):

    train_loader, val_loader = create_loaders(X_train, X_val, y_train, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = electric_nn(input_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):

        model.train()
        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

    # final validation RMSE
    model.eval()
    preds = []

    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).cpu())

    preds = torch.cat(preds).numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return rmse
        
        
        
# =============
# Random Forest
# =============

def RandomForestTrain(X_train, X_val, y_train, y_val):

    model = RandomForestRegressor(
        n_estimators=160,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return rmse
    