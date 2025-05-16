#sales.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 1) Load & validate data
try:
    df = pd.read_csv("sales_data.csv")
    required_cols = ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
except FileNotFoundError:
    print("Error: 'sales_data.csv' not found.")
    exit(1)
except pd.errors.ParserError:
    print("Error: Invalid CSV format in 'sales_data.csv'.")
    exit(1)

if not all(col in df.columns for col in required_cols):
    print("Error: Missing required columns in 'sales_data.csv'.")
    exit(1)

df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors='coerce')
if df["Date"].isna().any():
    print("Error: Invalid date format in 'sales_data.csv'.")
    exit(1)

df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# 2) One-hot encode store
stores = sorted(df["Store"].unique())
store2idx = {s: i for i, s in enumerate(stores)}
df["StoreIdx"] = df["Store"].map(store2idx)
oh = np.eye(len(stores))[df["StoreIdx"]]
oh_df = pd.DataFrame(oh, columns=[f"store_{s}" for s in stores])
df = pd.concat([df, oh_df], axis=1)

# 3) Features & scaling
base_feats = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
store_feats = [f"store_{s}" for s in stores]
FEATURE_COLS = base_feats + store_feats

# Scale base_feats per store
scalers = {}
for s in stores:
    mask = df["Store"] == s
    sc = MinMaxScaler()
    df.loc[mask, base_feats] = sc.fit_transform(df.loc[mask, base_feats])
    scalers[s] = sc

# 4) Create sequences
WINDOW = 4
class MultiStoreDataset(Dataset):
    def __init__(self, data):
        seqs, labs = [], []
        for s in stores:
            sub = data[data["Store"] == s].reset_index(drop=True)
            arr = sub[FEATURE_COLS].values
            for i in range(len(arr) - WINDOW):
                seqs.append(arr[i : i + WINDOW])
                labs.append(arr[i + WINDOW, FEATURE_COLS.index("Weekly_Sales")])
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labs, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MultiStoreDataset(df)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

# 5) Model definition
class LSTMForecaster(nn.Module):
    def __init__(self, in_dim, hid_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True)
        self.fc   = nn.Linear(hid_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecaster(len(FEATURE_COLS)).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 6) Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred   = model(Xb)
        loss   = loss_fn(pred, yb)
        train_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_loss /= len(train_loader)
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            vp = model(Xb.to(device))
            val_losses.append(loss_fn(vp, yb.to(device)).item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 7) Save bundle
torch.save({
    "model_state": model.state_dict(),
    "scalers": scalers,
    "stores": stores,
    "feature_cols": FEATURE_COLS,
}, "sales_forecaster_multi.pt")
print("✅ Multi-store sales model saved to sales_forecaster_multi.pt")