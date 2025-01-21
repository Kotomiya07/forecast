# モデルモジュール

このディレクトリには、予測モデルの定義とモデル関連のユーティリティ機能が含まれています。

## ファイル構成

### transformer.py
売上予測用のTransformerモデルの実装。

#### 主要クラス

1. `PositionalEncoding`
   - 位置エンコーディングを行うモジュール
   - サブクラス: `nn.Module`
   - 主な機能：
     - 正弦波ベースの位置エンコーディング生成
     - 入力テンソルへの位置情報の付加

2. `SalesTransformer`
   - 売上予測用のTransformerモデル
   - サブクラス: `nn.Module`
   - 主なコンポーネント：
     - 深い埋め込み層
     - エンコーダー/デコーダー構造
     - 位置エンコーディング
     - 中間層による特徴抽出
     - 時点ごとの予測出力層

#### アーキテクチャの詳細

1. 入力処理
   ```
   入力 → 埋め込み層 → 位置エンコーディング → エンコーダー
   ```
   - 多層埋め込み（d_model*4 → d_model*2 → d_model）
   - スケーリングファクター（sqrt(d_model)）
   - Dropoutによる正則化

2. Transformer構造
   ```
   エンコーダー出力 → デコーダー入力生成 → マスク生成 → デコーダー
   ```
   - マルチヘッドアテンション
   - Feed-forward networks
   - Layer normalization
   - Residual connections

3. 出力生成
   ```
   デコーダー出力 → 中間層 → 出力層 → 予測値
   ```
   - 中間層での特徴抽出
   - 各時点での独立した予測

### utils.py
モデル関連のユーティリティ機能。

#### 主要機能

1. 最適化関連
   - `create_optimizer_and_scheduler()`: オプティマイザとスケジューラの作成
     - AdamWオプティマイザ
     - コサイン学習率スケジューリング
     - パラメータグループごとの設定

2. 学習制御
   - `fix_seed()`: 乱数シードの固定
   - `setup_training_device()`: 学習デバイスの設定
   - `create_loss_function()`: 複合損失関数の作成

3. デバイス管理
   - `get_device()`: 学習デバイスの取得
   - CUDA関連の設定最適化

## モデルの特徴

1. アーキテクチャ上の工夫
   - より深い埋め込み層による豊かな特徴表現
   - デコーダー入力の最適化
   - 適切なマスキングによる予測品質の向上
   - 中間層による特徴抽出の強化

2. 学習の安定化
   - パラメータグループごとの重み減衰
   - 適切な重み初期化
   - 勾配クリッピング
   - Layer Normalizationの活用

3. 予測精度の向上
   - 複合損失関数の使用
   - コサイン学習率スケジューリング
   - Dropoutによる正則化
   - 残差接続による勾配伝播の改善

## 使用例

```python
from config.config import TRANSFORMER_CONFIG
from src.models.transformer import create_model
from src.models.utils import create_optimizer_and_scheduler

# モデルの作成
model = create_model("transformer", input_size=14)  # 14は特徴量の次元数

# オプティマイザとスケジューラの設定
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    learning_rate=5e-5,
    num_epochs=1000,
    num_training_steps=len(train_loader)
)
