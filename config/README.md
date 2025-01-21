# 設定モジュール

このディレクトリには、プロジェクト全体で使用される設定とハイパーパラメータが含まれています。

## ファイル構成

### config.py
プロジェクトの主要な設定ファイル。以下の設定が含まれます：

#### データ設定
- `INPUT_SEQUENCE_LENGTH`: 入力シーケンスの長さ（360日）
- `OUTPUT_SEQUENCE_LENGTH`: 出力シーケンスの長さ（90日）
- `NUM_WORKERS`: データローダーのワーカー数
- `PIN_MEMORY`: GPUメモリのピン留め設定

#### トレーニング設定
- `BATCH_SIZE`: バッチサイズ（64）
- `EPOCHS`: 総エポック数（1000）
- `LEARNING_RATE`: 学習率（5e-5）
- `EARLY_STOPPING_PATIENCE`: Early Stopping の待機エポック数（10）

#### モデル設定
- `MODEL_TYPE`: モデルタイプ（"transformer"）
- `TRANSFORMER_CONFIG`: Transformerモデルの設定
  - `d_model`: モデルの次元数（256）
  - `nhead`: アテンションヘッド数（8）
  - `num_layers`: Transformerレイヤー数（4）
  - `dropout`: ドロップアウト率（0.1）

#### その他の設定
- `DEFAULT_SEED`: 乱数シード値（42）
- `TRAIN_DATA_PATH`: 学習データのパス
- `CACHE_DIR`: 前処理済みデータのキャッシュディレクトリ
- `PLOT_SAVE_INTERVAL`: プロット保存間隔

## 設定の変更方法

1. モデルの容量を調整する場合：
   - `d_model`を増減
   - `num_layers`を変更
   - `nhead`を調整（`d_model`の約数である必要あり）

2. 学習の安定性を調整する場合：
   - `LEARNING_RATE`を調整
   - `BATCH_SIZE`を変更
   - `dropout`を調整

3. 予測期間を変更する場合：
   - `INPUT_SEQUENCE_LENGTH`と`OUTPUT_SEQUENCE_LENGTH`を調整
   - 入力期間と出力期間のバランスを考慮
