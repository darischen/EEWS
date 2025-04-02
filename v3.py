import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    # Load all CSV files from "etfs" and "stocks" directories
    data_dir_etfs = './data/etfs'
    data_dir_stocks = './data/stocks'
    all_files = []
    for directory in [data_dir_stocks, data_dir_etfs]:
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                all_files.append(os.path.join(directory, file))

    # Concatenate all data into a single DataFrame
    data_list = []
    for file in all_files:
        df = pd.read_csv(file)
        data_list.append(df)
    data = pd.concat(data_list)

    print("First few rows of the combined data:")
    print(data.head())
    print("Checking for missing values in each column:")
    print(data.isnull().sum())
    data = data.dropna()
    print("Shape of the cleaned data:", data.shape)
    print("First few rows of the cleaned data:")
    print(data.head())

    # PreProcessing
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    print("Checking for NaN values in data:", np.isnan(scaled_data).any())
    print("Checking for infinity values in data:", np.isinf(scaled_data).any())
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    def create_dataset(dataset, look_back=1):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), :])
            y.append(dataset[i + look_back, :])
        return np.array(X), np.array(y)
    
    look_back = 20
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    
    print("Shapes of datasets:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    
    return X_train, y_train, X_test, y_test, scaler, look_back, data

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(256, output_size)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc(out)
        return out

def train_model(X_train, y_train, X_test, y_test, look_back):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    
    batch_size = 256
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers=0 for Windows
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    input_size = X_train.shape[2]
    output_size = X_train.shape[2]
    
    model = LSTMModel(input_size=input_size, hidden_size=256, output_size=output_size)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    l1_lambda = 1e-10
    l2_lambda = 7.25e-10
    
    best_val_loss = float('inf')
    best_epoch = -1
    num_epochs = 17
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False, colour='white')
        for batch_x, batch_y in train_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            l1_norm = sum(param.abs().sum() for param in model.parameters())
            l2_norm = sum(param.pow(2).sum() for param in model.parameters())
            loss = loss + l2_lambda * l2_norm  # L1 regularization is omitted here, but can be added similarly
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False, colour='white')
        with torch.no_grad():
            for batch_x, batch_y in val_bar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_bar.set_postfix(loss=loss.item())
        val_loss = val_loss / len(test_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.9e}, Val Loss: {val_loss:.9e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'mv3_best.pth')
    
    print(f"Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.9e}")
    return best_epoch, best_val_loss

def evaluate_and_plot(scaler, look_back, data, BEST_EPOCH, BEST_VAL_LOSS):
    # Load the NVDA CSV file
    nvda_df = pd.read_csv('./data/stocks/nvda.csv')
    nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])
    nvda_df.set_index('Date', inplace=True)
    
    scaled_nvda = scaler.fit_transform(nvda_df)
    print("Checking for NaN values in data:", np.isnan(scaled_nvda).any())
    print("Checking for infinity values in data:", np.isinf(scaled_nvda).any())
    
    def create_dataset(dataset, look_back=1):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), :])
            y.append(dataset[i + look_back, :])
        return np.array(X), np.array(y)
    
    X_nvda, y_nvda = create_dataset(scaled_nvda, look_back)
    X_nvda_tensor = torch.from_numpy(X_nvda).float()
    
    input_size = nvda_df.shape[1]
    output_size = nvda_df.shape[1]
    
    model = LSTMModel(input_size=input_size, hidden_size=256, output_size=output_size)
    model.load_state_dict(torch.load('mv3_best.pth'))
    model.eval()
    
    with torch.no_grad():
        predictions_tensor = model(X_nvda_tensor)
    nvda_predictions = predictions_tensor.numpy()
    
    nvda_predictions_full = np.zeros_like(scaled_nvda)
    nvda_actual_full = np.zeros_like(scaled_nvda)
    
    nvda_predictions_full[look_back:look_back+len(nvda_predictions), -nvda_predictions.shape[1]:] = nvda_predictions
    nvda_actual_full[look_back:look_back+len(y_nvda), -y_nvda.shape[1]:] = y_nvda
    
    nvda_predictions = scaler.inverse_transform(nvda_predictions_full)[look_back:look_back+len(nvda_predictions), -nvda_predictions.shape[1]:]
    nvda_actual = scaler.inverse_transform(nvda_actual_full)[look_back:look_back+len(y_nvda), -y_nvda.shape[1]:]
    
    time_section = slice(0, 30000)
    features = nvda_df.columns.tolist()
    n_features = len(features)
    
    plt.figure(figsize=(14, 3 * n_features))
    for i, feature in enumerate(features):
        plt.subplot(n_features, 1, i+1)
        plt.plot(nvda_actual[time_section, i], color='blue', label=f'Actual NVDA {feature}')
        plt.plot(nvda_predictions[time_section, i], color='red', label=f'Predicted NVDA {feature}')
        plt.title(f"Epoch {BEST_EPOCH} - Val Loss: {BEST_VAL_LOSS:.9e} - {feature}")
        plt.xlabel('Time')
        plt.ylabel(f'{feature} ($)')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Configure GPU visibility before any torch imports if necessary.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Run the whole pipeline
    X_train, y_train, X_test, y_test, scaler, look_back, data = load_and_preprocess_data()
    BEST_EPOCH, BEST_VAL_LOSS = train_model(X_train, y_train, X_test, y_test, look_back)
    evaluate_and_plot(scaler, look_back, data, BEST_EPOCH, BEST_VAL_LOSS)