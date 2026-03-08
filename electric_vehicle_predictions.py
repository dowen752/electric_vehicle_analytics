import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from features import neuralNetworkDatasets, neuralNetworkTrain
    
    

def main():
    neuralNetworkTrain()
    

    
    
    

if __name__ == "__main__":
    main()