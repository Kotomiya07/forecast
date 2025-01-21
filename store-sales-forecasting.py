import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt
from accelerate import Accelerator
import argparse
import datetime

# ハイパーパラメータ
INPUT_SEQUENCE_LENGTH = 360  # 1年
OUTPUT_SEQUENCE_LENGTH = 30  # 3ヶ月
BATCH_SIZE = 64  # より小さいバッチサイズで細かく学習
EPOCHS = 20  # エポック数を増やして学習を深める
LEARNING_RATE = 5e-4  # より小さい学習率で安定した学習
MODEL_TYPE = "transformer"
TRANSFORMER_D_MODEL = 256  # モデルの表現力を向上
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 2  # より深いTransformer
TRANSFORMER_DROPOUT = 0.2  # やや小さいDropoutで情報をより保持
NUM_WORKERS = 16
PIN_MEMORY = True

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="時系列予測モデルの学習")
parser.add_argument("--seed", type=int, default=42, help="乱数シード")
args = parser.parse_args()

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

def fix_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def get_device():
    return device

def create_features(df):
    # 基本的な時間特徴量
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    
    # 周期性を捉える特徴量
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    # 季節性を捉える特徴量
    df['season'] = df['month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    # 週末フラグ
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 月末フラグ
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # 三角関数による周期性の表現
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    return df

def preprocess_data(data, input_len, output_len, sales_scaler):
    sequences = []
    targets = []
    features = []
    unique_stores = data['store_nbr'].unique()

    # 特徴量の選択
    feature_columns = [
        'dayofweek', 'month', 'year', 'day', 'dayofyear', 'weekofyear',
        'season', 'is_weekend', 'is_month_end', 'month_sin', 'month_cos',
        'day_sin', 'day_cos'
    ]

    for store in unique_stores:
        store_data = data[data['store_nbr'] == store].copy()
        
        # 売上データのスケーリング
        sales = store_data['sales'].values.astype(float).reshape(-1, 1)
        scaled_sales = sales_scaler.fit_transform(sales).flatten()
        
        # 特徴量の正規化
        store_features = store_data[feature_columns].values.astype(float)
        feature_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(store_features)

        # シーケンスの作成
        for i in range(len(scaled_sales) - input_len - output_len):
            # 売上データのシーケンス
            sequences.append(scaled_sales[i:i + input_len])
            targets.append(scaled_sales[i + input_len:i + input_len + output_len])
            
            # 対応する特徴量シーケンス
            features.append(scaled_features[i:i + input_len])

    # 特徴量とシーケンスの結合
    sequences = np.array(sequences)
    features = np.array(features)
    combined_sequences = np.concatenate([
        sequences.reshape(sequences.shape[0], sequences.shape[1], 1),
        features
    ], axis=2)

    return combined_sequences, np.array(targets)

class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class SalesTransformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        super(SalesTransformer, self).__init__()
        
        # より深い埋め込み層
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout/2)
        
        # 拡張されたTransformer層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=decoder_norm
        )
        
        # 出力層
        self.fc = nn.Sequential(
            nn.Linear(d_model * OUTPUT_SEQUENCE_LENGTH, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(d_model * 2, output_size)
        )
        
        self.d_model = d_model

    def forward(self, src):
        # 特徴量は既に正しい形状で入力される
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # エンコーダー処理
        memory = self.transformer_encoder(src)
        
        # デコーダー入力の準備
        tgt = torch.zeros(src.size(0), OUTPUT_SEQUENCE_LENGTH, self.d_model).to(src.device)
        tgt = self.pos_encoder(tgt)
        
        # デコーダー処理
        output = self.transformer_decoder(tgt, memory)
        
        # 最終的な予測
        output = self.fc(output.reshape(output.size(0), -1))
        
        return output

def create_model(model_type):
    if model_type == "transformer":
        # 入力サイズを特徴量の数（13）+ 売上データ（1）に更新
        input_size = 14
        return SalesTransformer(input_size, OUTPUT_SEQUENCE_LENGTH, TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT)
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {model_type}")

def save_predictions_plot(model, loader, epoch, sales_scaler, results_dir):
    model.eval()
    X_batch, y_batch = next(iter(loader))
    with torch.no_grad():
        predictions = model(X_batch).cpu().numpy()

    y_true = y_batch.cpu().numpy()

    y_true_rescaled = sales_scaler.inverse_transform(y_true)
    predictions_rescaled = sales_scaler.inverse_transform(predictions)
    input_rescaled = sales_scaler.inverse_transform(X_batch[0].cpu().numpy().reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH), input_rescaled, label="Input", color="blue")
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH, INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH), y_true_rescaled[0], label="True", color="green")
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH, INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH), predictions_rescaled[0], label="Predicted", color="red")
    plt.legend()
    plt.title(f"Epoch {epoch + 1} Predictions")
    plt.savefig(f"{results_dir}/epoch_{epoch + 1}_predictions.png")
    plt.close()

def evaluate_model(model, loader, sales_scaler):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    mape_sum = 0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())
            
            # バッチごとの損失を計算
            loss = nn.MSELoss()(predictions, y_batch)
            total_loss += loss.item() * len(y_batch)
            
            # MAPEの計算
            y_true_rescaled = sales_scaler.inverse_transform(y_batch.cpu().numpy())
            y_pred_rescaled = sales_scaler.inverse_transform(predictions.cpu().numpy())
            
            # ゼロ除算を防ぐ
            epsilon = 1e-10
            mape = np.mean(np.abs((y_true_rescaled - y_pred_rescaled) / (y_true_rescaled + epsilon))) * 100
            mape_sum += mape * len(y_batch)
            
            n_samples += len(y_batch)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    y_true_rescaled = sales_scaler.inverse_transform(y_true)
    y_pred_rescaled = sales_scaler.inverse_transform(y_pred)

    # 基本的な評価指標
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    rmsle = np.sqrt(mean_squared_log_error(np.maximum(y_true_rescaled, 0), np.maximum(y_pred_rescaled, 0)))
    
    # 平均のMAPE
    # 平均のMAPE
    
    # R2スコア
    r2 = 1 - np.sum((y_true_rescaled - y_pred_rescaled) ** 2) / np.sum((y_true_rescaled - np.mean(y_true_rescaled)) ** 2)

    # 平均損失
    avg_loss = total_loss / n_samples

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'rmsle': rmsle,
        'mape': mape,
        'r2': r2,
        'avg_loss': avg_loss
    }

    return metrics, y_pred_rescaled, y_true

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, accelerator, scheduler, sales_scaler, results_dir):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 5  # Early stoppingの閾値
    patience_counter = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        n_train_batches = 0
        
        # トレーニングループ
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            accelerator.backward(loss)
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / n_train_batches
        train_losses.append(avg_train_loss)

        # 検証ループ
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        # Early stopping チェック
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # ベストモデルの保存
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"{results_dir}/best_model.pth")
        else:
            patience_counter += 1

        # エポックごとの評価指標を計算
        train_metrics, _, _ = evaluate_model(model, train_loader, sales_scaler)
        val_metrics, _, _ = evaluate_model(model, val_loader, sales_scaler)
        
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Time: {elapsed_time:.2f}s")
        print("\nTraining Metrics:")
        print(f"Loss: {avg_train_loss:.4f}")
        print(f"MSE: {train_metrics['mse']:.4f}")
        print(f"RMSE: {train_metrics['rmse']:.4f}")
        print(f"MAE: {train_metrics['mae']:.4f}")
        print(f"MAPE: {train_metrics['mape']:.2f}%")
        
        print("\nValidation Metrics:")
        print(f"Loss: {avg_val_loss:.4f}")
        print(f"MSE: {val_metrics['mse']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"MAE: {val_metrics['mae']:.4f}")
        print(f"MAPE: {val_metrics['mape']:.2f}%")
        print(f"Best Val Loss: {best_val_loss:.4f}")

        # 予測プロットの保存
        if (epoch + 1) % 5 == 0:  # 5エポックごとにプロット
            save_predictions_plot(model, val_loader, epoch, sales_scaler, results_dir)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    return train_losses, val_losses, best_val_loss

def main():
    # CUDA高速化のための設定
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="時系列予測モデルの学習")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # シード値の固定 (Accelerator のランクを使用)
    fix_seed(args.seed, accelerator.process_index)

    # データの読み込み
    data_path = "data/store-sales/train.csv"
    data = pd.read_csv(data_path, parse_dates=['date'])
    data = data[data['family'] == 'GROCERY I'].copy()  # 'GROCERY I' のみにフィルタリング
    data = data.sort_values(['store_nbr', 'date']).reset_index(drop=True)

    # 特徴量エンジニアリング
    data = create_features(data)

    # スケーラーの初期化
    sales_scaler = MinMaxScaler()

    # 前処理
    sequences, targets = preprocess_data(data, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, sales_scaler)
    X_train_val, X_test, y_train_val, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=args.seed)

    # データセットとデータローダー
    train_dataset = SalesDataset(X_train, y_train)
    val_dataset = SalesDataset(X_val, y_val)
    test_dataset = SalesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # モデル、損失関数、最適化アルゴリズムの初期化
    model = create_model(MODEL_TYPE)
    
    # 複合損失関数の定義
    def custom_loss(pred, target):
        # MSE損失
        mse_loss = nn.MSELoss()(pred, target)
        
        # Huber損失（外れ値に対してより頑健）
        huber_loss = nn.HuberLoss(delta=1.0)(pred, target)
        
        # MAE損失（絶対誤差）
        mae_loss = nn.L1Loss()(pred, target)
        
        # 方向性の一致を評価する損失
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        direction_loss = -torch.mean(torch.sign(pred_diff) * torch.sign(target_diff))
        
        # 損失の重み付け結合
        total_loss = 0.4 * mse_loss + 0.3 * huber_loss + 0.2 * mae_loss + 0.1 * direction_loss
        return total_loss
    
    criterion = custom_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # One Cycle Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # データローダーの準備
    train_loader, val_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, test_loader, model, optimizer
    )

    # 予測プロットの保存用ディレクトリの作成
    results_dir = f"results/{MODEL_TYPE}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(f"{results_dir}", exist_ok=True)

    # 学習
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, optimizer, criterion, accelerator, scheduler, sales_scaler, results_dir)

    # 評価
    metrics, predictions, targets = evaluate_model(model, test_loader, sales_scaler)
    print("\nテスト結果:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"RMSLE: {metrics['rmsle']:.4f}")
    print(f"MAPE: {metrics['mape']:.4f}%")
    print(f"R2 Score: {metrics['r2']:.4f}")

    # テストデータでの予測プロットの保存
    save_predictions_plot(model, test_loader, EPOCHS, sales_scaler, results_dir)

if __name__ == "__main__":
    main()
