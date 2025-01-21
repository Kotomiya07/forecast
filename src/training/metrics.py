"""
モデルの評価指標に関する機能
"""
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

def evaluate_model(model, loader, sales_scaler):
    """モデルの評価を行う

    Args:
        model (nn.Module): 評価するモデル
        loader (DataLoader): データローダー
        sales_scaler (MinMaxScaler): 売上データのスケーラー

    Returns:
        tuple: (評価指標の辞書, 予測値, 真値)
    """
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
            loss = torch.nn.MSELoss()(predictions, y_batch)
            total_loss += loss.item() * len(y_batch)
            
            # MAPEの計算
            y_batch_np = y_batch.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            
            # ゼロ除算を防ぐ
            epsilon = 1e-10
            mape = np.mean(np.abs((y_batch_np - pred_np) / (y_batch_np + epsilon))) * 100
            mape_sum += mape * len(y_batch)
            
            n_samples += len(y_batch)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 評価指標の計算
    metrics = calculate_metrics(y_true, y_pred, total_loss, n_samples, mape_sum)

    return metrics, y_pred, y_true

def calculate_metrics(y_true, y_pred, total_loss, n_samples, mape_sum):
    """各種評価指標を計算する

    Args:
        y_true (np.ndarray): 真値
        y_pred (np.ndarray): 予測値
        total_loss (float): 総損失
        n_samples (int): サンプル数
        mape_sum (float): MAPEの合計

    Returns:
        dict: 評価指標の辞書
    """
    # 基本的な評価指標
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rmsle = np.sqrt(mean_squared_log_error(np.maximum(y_true, 0), np.maximum(y_pred, 0)))
    
    # 平均MAPE
    mape = mape_sum / n_samples
    
    # R2スコア
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

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

    return metrics
