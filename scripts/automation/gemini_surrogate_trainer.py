import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
"""
Surrogate Model Trainer for dFBA-RL
FBAの結果をニューラルネットワークで模倣するモデルを学習する
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import argparse
import json

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def train_for_species(df, species_name, output_dir):
    print(f"\n🧬 Training surrogate model for: {species_name}")
    
    # この種のみのデータを抽出
    species_df = df[df['species'] == species_name].copy()
    
    if len(species_df) < 100:
        print(f"  ⚠️ Too few samples ({len(species_df)}), skipping...")
        return

    # 特徴量 (X) とラベル (y) の選定
    input_cols = [col for col in df.columns if col.startswith('conc_')]
    all_flux_cols = [col for col in df.columns if col.startswith('flux_')]
    output_cols = ['growth_rate'] + all_flux_cols
    
    # 欠損値（この種に存在しない反応など）を0で埋める
    species_df[all_flux_cols] = species_df[all_flux_cols].fillna(0.0)
    
    # 常に0のフラックス（変動がないもの）を除外
    # 【科学的証明のための修正】変動の有無に関わらず、全ての交換反応を出力に含める
    useful_output_cols = output_cols
    
    print(f"  Input dim: {len(input_cols)}, Output dim: {len(useful_output_cols)}")
    print(f"  Samples: {len(species_df)}")
    
    X = species_df[input_cols].values
    y = species_df[useful_output_cols].values
    
    # 前処理
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Tensor化
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # モデル
    model = SurrogateMLP(len(input_cols), len(useful_output_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 学習
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    test_loss += criterion(outputs, batch_y).item()
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Test Loss: {test_loss/len(test_loader):.6f}")
            
    # 保存
    species_dir = Path(output_dir) / species_name
    species_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), species_dir / "model.pth")
    joblib.dump(X_scaler, species_dir / "X_scaler.pkl")
    
    metadata = {
        "input_cols": input_cols,
        "output_cols": useful_output_cols,
        "input_dim": len(input_cols),
        "output_dim": len(useful_output_cols)
    }
    with open(species_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  ✅ Model saved to {species_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, default='outputs/fba_data_collection.csv')
    parser.add_argument('--output-dir', type=str, default='models/surrogate')
    args = parser.parse_args()
    
    print("📊 Loading data (this may take a while)...")
    df = pd.read_csv(args.csv_path, low_memory=False)
    
    # クリーニング
    df = df[df['time'] != 'time'].copy()
    numeric_cols = [col for col in df.columns if col != 'species']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    species_list = [s for s in df['species'].unique() if isinstance(s, str)]
    print(f"Detected species: {species_list}")
    
    for species in species_list:
        train_for_species(df, species, args.output_dir)

if __name__ == "__main__":
    main()
