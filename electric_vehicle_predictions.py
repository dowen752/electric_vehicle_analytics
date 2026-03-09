import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from features import load_datasets, neuralNetworkTrain, RandomForestTrain
    
    

def main():

    X_train, X_val, y_train, y_val = load_datasets()

    nn_rmse = neuralNetworkTrain(X_train, X_val, y_train, y_val)

    rf_rmse = RandomForestTrain(X_train, X_val, y_train, y_val)

    print("Neural Network RMSE:", nn_rmse)
    print("Random Forest RMSE:", rf_rmse)
    
    

if __name__ == "__main__":
    main()