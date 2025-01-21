"""
モデルの学習に関する機能
"""
import time
import os
import torch
import datetime
from .metrics import evaluate_model
from ..utils.visualization import save_predictions_plot

class ModelTrainer:
    """モデルの学習を管理するクラス"""
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, 
                accelerator, scheduler, sales_scaler, results_dir):
        """
        Args:
            model: 学習するモデル
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            optimizer: オプティマイザ
            criterion: 損失関数
            accelerator: Acceleratorインスタンス
            scheduler: 学習率スケジューラ
            sales_scaler: 売上データのスケーラー
            results_dir: 結果保存ディレクトリ
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.sales_scaler = sales_scaler
        self.results_dir = results_dir

    def train(self, epochs, patience=5):
        """モデルの学習を実行

        Args:
            epochs (int): 学習エポック数
            patience (int, optional): Early stopping の閾値

        Returns:
            tuple: (学習損失のリスト, 検証損失のリスト, 最良の検証損失)
        """
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()
            
            # 訓練ループ
            train_loss = self._train_epoch()
            train_losses.append(train_loss)

            # 検証ループ
            val_loss = self._validate_epoch()
            val_losses.append(val_loss)

            # スケジューラの更新
            if self.scheduler:
                self.scheduler.step(epoch)

            # Early stopping チェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # ベストモデルの保存
                self._save_model(epoch, train_loss, val_loss)
            else:
                patience_counter += 1

            # エポックごとの評価指標を計算
            train_metrics, _, _ = evaluate_model(self.model, self.train_loader, self.sales_scaler)
            val_metrics, _, _ = evaluate_model(self.model, self.val_loader, self.sales_scaler)
            
            elapsed_time = time.time() - start_time
            self._print_epoch_results(epoch, epochs, elapsed_time, train_loss, val_loss,
                                   train_metrics, val_metrics, best_val_loss)

            # 予測プロットの保存
            if (epoch + 1) % 5 == 0:  # 5エポックごとにプロット
                save_predictions_plot(self.model, self.val_loader, epoch, 
                                   self.sales_scaler, self.results_dir)

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        return train_losses, val_losses, best_val_loss

    def _train_epoch(self):
        """1エポックの学習を実行

        Returns:
            float: エポックの平均損失
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for i, (X_batch, y_batch) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            self.accelerator.backward(loss)
            
            # 勾配クリッピング（より緩やかな制限）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches

    def _validate_epoch(self):
        """1エポックの検証を実行

        Returns:
            float: エポックの平均損失
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def _save_model(self, epoch, train_loss, val_loss):
        """モデルの保存

        Args:
            epoch (int): 現在のエポック
            train_loss (float): 訓練損失
            val_loss (float): 検証損失
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f"{self.results_dir}/best_model.pth")

    def _print_epoch_results(self, epoch, epochs, elapsed_time, train_loss, val_loss,
                           train_metrics, val_metrics, best_val_loss):
        """エポックの結果を表示

        Args:
            epoch (int): 現在のエポック
            epochs (int): 総エポック数
            elapsed_time (float): エポックの実行時間
            train_loss (float): 訓練損失
            val_loss (float): 検証損失
            train_metrics (dict): 訓練データの評価指標
            val_metrics (dict): 検証データの評価指標
            best_val_loss (float): 最良の検証損失
        """
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Time: {elapsed_time:.2f}s")
        print("\nTraining Metrics:")
        print(f"Loss: {train_loss:.4f}")
        print(f"MSE: {train_metrics['mse']:.4f}")
        print(f"RMSE: {train_metrics['rmse']:.4f}")
        print(f"MAE: {train_metrics['mae']:.4f}")
        print(f"MAPE: {train_metrics['mape']:.2f}%")
        
        print("\nValidation Metrics:")
        print(f"Loss: {val_loss:.4f}")
        print(f"MSE: {val_metrics['mse']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"MAE: {val_metrics['mae']:.4f}")
        print(f"MAPE: {val_metrics['mape']:.2f}%")
        print(f"Best Val Loss: {best_val_loss:.4f}")

def setup_training(model_name):
    """学習の準備を行う

    Args:
        model_name (str): モデルの名前

    Returns:
        str: 結果保存用ディレクトリのパス
    """
    results_dir = f"results/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
