# 売上予測システム

## 概要
時系列データを使用して将来の売上を予測するシステムです。Transformerアーキテクチャを採用し、複雑なパターンの学習と予測を行います。

## 主な特徴
- Transformerベースの時系列予測モデル
- 豊富な時間的特徴量の自動生成
- データ前処理のキャッシング機能
- 効率的な学習プロセス
- 詳細な評価指標の計算
- 分散学習サポート（accelerate）

## データセット
このプロジェクトは[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)のデータセットを使用します。

### データの準備
1. Kaggleアカウントを作成（まだの場合）
2. 上記リンクからデータセットをダウンロード
3. ダウンロードしたファイルを`data/store-sales/`ディレクトリに配置：
   ```
   data/
   └── store-sales/
       ├── train.csv        # 学習データ
       ├── test.csv         # テストデータ
       └── sample_submission.csv
   ```

## プロジェクト構造
```
forecast/
├── config/
│   ├── config.py         # 設定とハイパーパラメータ
│   └── README.md         # 設定に関するドキュメント
├── src/
│   ├── data/            # データ処理関連
│   │   ├── dataset.py      # データセットクラス
│   │   ├── preprocessing.py # データ前処理
│   │   └── README.md       # データ処理のドキュメント
│   ├── models/          # モデル定義
│   │   ├── transformer.py  # Transformerモデル
│   │   ├── utils.py        # モデルユーティリティ
│   │   └── README.md       # モデルのドキュメント
│   ├── training/        # 学習関連
│   │   ├── trainer.py      # トレーニングループ
│   │   ├── metrics.py      # 評価指標
│   │   └── README.md       # 学習プロセスのドキュメント
│   └── utils/          # ユーティリティ
│       ├── visualization.py # 可視化機能
│       └── README.md       # ユーティリティのドキュメント
├── main.py             # メイン実行スクリプト
├── pyproject.toml      # プロジェクト設定
├── cache/             # 前処理済みデータのキャッシュ
└── results/           # 学習結果の保存先
```

## 主要コンポーネント

### モデルアーキテクチャ
- 深層Transformerモデル
- 豊富な中間層による高い表現力
- デコーダーでの適切なマスキング
- 各時点での独立した予測生成

### データ処理
- 時間的特徴量の自動生成
- スケーリングと正規化
- 効率的なデータキャッシング
- シーケンスデータの生成

### 学習プロセス
- AdamWオプティマイザ
- コサイン学習率スケジューリング
- 複合損失関数
- Early Stopping
- accelerateによる分散学習サポート

### 評価指標
- MSE（平均二乗誤差）
- RMSE（二乗平均平方根誤差）
- MAE（平均絶対誤差）
- MAPE（平均絶対パーセント誤差）
- R2スコア

## セットアップと実行

### 必要条件
- Python 3.8以上
- Rye (Pythonパッケージマネージャー)
- CUDA対応GPU（推奨）

### インストール
```bash
# Ryeがインストールされていない場合
curl -sSf https://rye-up.com/get | bash
# または
curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash

# プロジェクトのクローン
git clone <repository-url>
cd forecast

# Ryeによる環境のセットアップ
rye sync
```

### 実行方法

#### 単一GPUでの実行
```bash
# 仮想環境の有効化
rye shell

# 学習の実行
accelerate launch main.py

# 特定のシードを指定して実行
accelerate launch main.py --seed 42
```

#### 複数GPUでの実行
```bash
# accelerateの設定
accelerate config

# 分散学習の実行
accelerate launch --multi_gpu main.py
```

#### CPU実行
```bash
# CPU実行の設定
accelerate config default --cpu

# CPU上で実行
accelerate launch main.py
```

## 設定のカスタマイズ
`config/config.py`で以下の設定をカスタマイズできます：
- シーケンス長
- バッチサイズ
- 学習率
- モデルパラメータ
- Early Stoppingの設定
- その他の学習パラメータ

## キャッシュの利用
- 初回実行時：フルデータ処理とキャッシュ作成
- 2回目以降：キャッシュからの高速読み込み
- キャッシュの保存先：`cache/preprocessed_data/`

## 結果の保存
- モデルのチェックポイント
- 学習曲線
- 予測結果のプロット
- 評価指標のログ

## 開発ガイド

### 依存パッケージの追加
```bash
# 実行時依存の追加
rye add package_name

# 開発時依存の追加
rye add --dev package_name
```

### コード品質の維持
```bash
# フォーマッティング
rye run black .

# リンター
rye run flake8

# インポートの整理
rye run isort .

# テストの実行
rye run pytest
```

## ライセンス
MITライセンス
