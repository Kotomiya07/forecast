# 学習モジュール

このディレクトリには、モデルの学習プロセスと評価に関連するモジュールが含まれています。

## ファイル構成

### trainer.py
モデルの学習を管理するクラスと関連機能を実装。

#### 主要クラス

`ModelTrainer`
- 学習プロセス全体を管理
- 主な機能：
  1. エポック単位の学習制御
  2. 損失の計算と最適化
  3. 評価指標の計算
  4. モデルの保存
  5. 進捗の表示

#### 主要メソッド

1. `train()`
   - エポック単位の学習を実行
   - Early stoppingの制御
   - 学習履歴の記録
   - 最良モデルの保存

2. `_train_epoch()`
   - 1エポックの学習を実行
   - バッチ単位の処理
   - 勾配計算と最適化
   - 損失の計算

3. `_validate_epoch()`
   - 1エポックの検証を実行
   - 検証データでの評価
   - 損失の計算

### metrics.py
評価指標の計算を行うモジュール。

#### 主要機能

1. `evaluate_model()`
   - モデルの総合的な評価
   - 複数の評価指標の計算
   - 予測値と実際値の比較

2. 評価指標
   - MSE（Mean Squared Error）
   - RMSE（Root Mean Squared Error）
   - MAE（Mean Absolute Error）
   - MAPE（Mean Absolute Percentage Error）
   - R2スコア

## 学習プロセスの詳細

### 1. 初期化
```python
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    accelerator=accelerator,
    scheduler=scheduler,
    sales_scaler=sales_scaler,
    results_dir=results_dir
)
```

### 2. 学習ループ
```python
train_losses, val_losses, best_val_loss = trainer.train(
    epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE
)
```

### 3. エポックごとの処理
1. 訓練フェーズ
   - バッチ単位の学習
   - 勾配クリッピング（max_norm=3.0）
   - オプティマイザの更新
   - スケジューラの更新

2. 検証フェーズ
   - 予測精度の評価
   - 損失の計算
   - Early stoppingの判定

3. モデルの保存
   - 最良モデルの保存
   - チェックポイントの作成

4. 進捗の表示
   - 訓練損失
   - 検証損失
   - 各種評価指標
   - 経過時間

## 評価指標の詳細

### MSE（平均二乗誤差）
- 予測値と実際値の差の二乗の平均
- より大きな誤差に対してペナルティが大きい

### RMSE（二乗平均平方根誤差）
- MSEの平方根
- 元の単位での誤差を表現

### MAE（平均絶対誤差）
- 予測値と実際値の絶対差の平均
- 外れ値の影響を受けにくい

### MAPE（平均絶対パーセント誤差）
- 相対的な誤差を表現
- パーセンテージでの評価

### R2スコア
- モデルの説明力を表現
- 1に近いほど良好

## 学習の安定化機能

1. 勾配クリッピング
   - 勾配爆発の防止
   - 学習の安定化

2. Early Stopping
   - 過学習の防止
   - 計算資源の効率的利用

3. モデルの保存
   - 最良モデルの保持
   - 学習の再開が可能
