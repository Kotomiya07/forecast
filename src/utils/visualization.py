"""
可視化に関する機能
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from config.config import INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH

def save_predictions_plot(model, loader, epoch, sales_scaler, results_dir):
    """予測結果をプロットして保存する

    Args:
        model: 予測モデル
        loader: データローダー
        epoch (int): 現在のエポック数
        sales_scaler: 売上データのスケーラー
        results_dir (str): 結果保存ディレクトリ
    """
    model.eval()
    X_batch, y_batch = next(iter(loader))
    with torch.no_grad():
        predictions = model(X_batch).cpu().numpy()

    y_true = y_batch.cpu().numpy()

    # 最初のバッチの売上データ（最初の特徴量）のみを取得
    input_sequence = X_batch[0, :, 0].cpu().numpy()

    plot_sequence_prediction(
        input_sequence=input_sequence,
        true_sequence=y_true[0],
        predicted_sequence=predictions[0],
        epoch=epoch,
        save_path=f"{results_dir}/epoch_{epoch + 1}_predictions.png"
    )

def plot_sequence_prediction(input_sequence, true_sequence, predicted_sequence, 
                           epoch, save_path):
    """時系列予測結果をプロットする

    Args:
        input_sequence (np.ndarray): 入力系列
        true_sequence (np.ndarray): 真の系列
        predicted_sequence (np.ndarray): 予測系列
        epoch (int): 現在のエポック数
        save_path (str): 保存パス
    """
    plt.figure(figsize=(12, 6))
    
    # 入力系列のプロット
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH), 
            input_sequence, 
            label="Input", 
            color="blue")
    
    # 真の系列のプロット
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH, 
                      INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH),
            true_sequence,
            label="True",
            color="green")
    
    # 予測系列のプロット
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH,
                      INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH),
            predicted_sequence,
            label="Predicted",
            color="red")
    
    plt.legend()
    plt.title(f"Epoch {epoch + 1} Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Sales")
    plt.grid(True)
    
    # プロットの保存
    plt.savefig(save_path)
    plt.close()

def plot_training_history(train_losses, val_losses, save_path):
    """学習履歴をプロットする

    Args:
        train_losses (list): 訓練損失の履歴
        val_losses (list): 検証損失の履歴
        save_path (str): 保存パス
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
