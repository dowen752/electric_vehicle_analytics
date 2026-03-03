import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class electric_nn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
    def forward(self, x):
        return self.net.forward(x)
    
    

def main():
    df = pd.read_csv("electric_vehicle_analytics.csv")
    X = df[["Year", "Battery_Capacity_kWh", "Range_km", "Charging_Time_hr"]].values
    Y = df["Resale_Value_USD"].values.reshape(-1, 1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    
    X_train = torch.tensor(X_train, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.float32)
    X_val = torch.tensor(X_val, dtype = torch.float32)
    y_val = torch.tensor(y_val, dtype = torch.float32)
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)
    
    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size = 32, shuffle = True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = electric_nn(input_dim = 4).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    model.train()
    
    print("Begginnig with device: ", device)
    
    for epoch in range(200):
        print("Epoch:", epoch, "/ 200")
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        
        train_loss /= len(train_loader)    
            
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                val_preds = model(X_batch)
                loss = criterion(val_preds, y_batch)
                
                val_loss += loss.item()
            
            val_loss /= len(val_loader)
        
        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        print(f"Current train_rmse: {train_rmse:.4f}")
        print(f"Current val_rmse: {val_rmse:.4f}")

    
    
    

if __name__ == "__main__":
    main()