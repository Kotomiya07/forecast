# ユーティリティモジュール

このディレクトリには、可視化や補助的な機能を提供するモジュールが含まれています。

## ファイル構成

### visualization.py
学習過程と予測結果の可視化を行うモジュール。

#### 主要機能

1. `plot_training_history()`
   - 学習曲線の描画
   - パラメータ：
     - `train_losses`: 訓練損失の履歴
     - `val_losses`: 検証損失の履歴
     - `save_path`: プロット保存パス
   - 特徴：
     - エポックごとの損失推移を表示
     - 訓練/検証の比較が可能
     - 自動保存機能

2. `save_predictions_plot()`
   - 予測結果の可視化
   - パラメータ：
     - `model`: 学習済みモデル
     - `val_loader`: 検証データローダー
     - `epoch`: 現在のエポック
     - `sales_scaler`: スケーラー
     - `results_dir`: 結果保存ディレクトリ
   - 特徴：
     - 実際の売上と予測値の比較
     - 予測区間の表示
     - 時系列での推移の可視化

## 可視化の詳細

### 学習曲線プロット
```python
plot_training_history(
    train_losses,
    val_losses,
    f"{results_dir}/training_history.png"
)
```

1. プロットの要素
   - x軸：エポック数
   - y軸：損失値
   - 2つの系列：
     - 訓練損失（青線）
     - 検証損失（オレンジ線）

2. プロットの特徴
   - グリッド表示
   - 凡例の表示
   - 軸ラベルの自動設定
   - 適切なスケーリング

### 予測結果プロット
```python
save_predictions_plot(
    model,
    val_loader,
    epoch,
    sales_scaler,
    results_dir
)
```

1. プロットの要素
   - x軸：時間
   - y軸：売上金額
   - 3つの系列：
     - 実際の売上（青線）
     - モデルの予測（赤線）
     - 予測区間（薄い赤の帯）

2. プロットの特徴
   - 自動スケーリング
   - 適切な時間軸の表示
   - 見やすい色使い
   - 重要なポイントの強調

## プロット保存と管理

1. ファイル名の規則
   - 学習曲線：`training_history.png`
   - 予測結果：`predictions_epoch_{epoch}.png`

2. 保存ディレクトリ構造
   ```
   results/
   ├── model_name_timestamp/
   │   ├── training_history.png
   │   ├── predictions_epoch_5.png
   │   ├── predictions_epoch_10.png
   │   └── ...
   ```

3. 画質設定
   - DPI: 300
   - フォーマット: PNG
   - 適切なフィギュアサイズ

## 使用例

1. 学習曲線の描画
   ```python
   # 学習後の損失履歴の可視化
   plot_training_history(
       train_losses=history['train_losses'],
       val_losses=history['val_losses'],
       save_path='results/training_history.png'
   )
   ```

2. 予測結果の可視化
   ```python
   # 5エポックごとに予測結果をプロット
   if (epoch + 1) % 5 == 0:
       save_predictions_plot(
           model, val_loader, epoch,
           sales_scaler, results_dir
       )
