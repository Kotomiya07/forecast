# データ処理モジュール

このディレクトリには、データの読み込み、前処理、特徴量生成に関連するモジュールが含まれています。

## データセットについて

このプロジェクトでは、Kaggleの[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)コンペティションのデータセットを使用します。

### データの入手方法

1. Kaggleアカウントの作成
   - [Kaggle](https://www.kaggle.com/)にアクセス
   - アカウントを作成（まだ持っていない場合）

2. データセットのダウンロード
   - 上記コンペティションページにアクセス
   - "Download All"ボタンをクリック
   - ダウンロードしたzipファイルを展開

3. データの配置
   ```
   data/
   └── store-sales/
       ├── train.csv        # 学習データ
       ├── test.csv         # テストデータ
       └── sample_submission.csv
   ```

## ファイル構成

### dataset.py
PyTorchのDatasetクラスを実装したファイル。

#### 主要クラス
- `SalesDataset`: 売上予測用のデータセットクラス
  - 入力シーケンスと目標値のペアを管理
  - PyTorchのDataLoaderと互換

### preprocessing.py
データの前処理と特徴量エンジニアリングを行うモジュール。

#### 主要機能
1. データ前処理
   - `preprocess_data()`: 生データから学習用シーケンスデータを生成
   - `create_features()`: 時間的特徴量の生成
   - `load_and_preprocess_data()`: データ読み込みと前処理の統合関数

2. 特徴量生成
   - 基本的な時間特徴量（曜日、月、年など）
   - 周期性特徴量（年間通算日、週番号）
   - 季節性特徴量（季節情報）
   - カスタム特徴量（週末フラグ、月末フラグ）
   - 三角関数特徴量（月と日の周期性表現）

3. キャッシュ管理
   - `get_cache_path()`: キャッシュファイルのパス生成
   - `check_cache_exists()`: キャッシュの存在確認
   - `save_to_cache()`: 前処理済みデータの保存
   - `load_from_cache()`: キャッシュからのデータ読み込み

## データ前処理の流れ

1. データ読み込み
   ```python
   data = pd.read_csv(data_path, parse_dates=['date'])
   ```

2. 特徴量生成
   ```python
   data = create_features(data)
   ```

3. シーケンスデータ作成
   ```python
   sequences, targets, scaler = preprocess_data(data, input_len, output_len)
   ```

4. キャッシュ管理
   ```python
   # キャッシュがある場合
   if check_cache_exists(data_path, input_len, output_len):
       return load_from_cache(data_path, input_len, output_len)
   
   # キャッシュがない場合
   save_to_cache(sequences, targets, scaler, data_path, input_len, output_len)
   ```

## 生成される特徴量

1. 時間的特徴量：
   - `dayofweek`: 曜日（0-6）
   - `month`: 月（1-12）
   - `year`: 年
   - `day`: 日（1-31）
   - `dayofyear`: 年間通算日（1-366）
   - `weekofyear`: 週番号（1-53）

2. カスタム特徴量：
   - `season`: 季節（1-4）
   - `is_weekend`: 週末フラグ（0/1）
   - `is_month_end`: 月末フラグ（0/1）

3. 周期性特徴量：
   - `month_sin`, `month_cos`: 月の周期性
   - `day_sin`, `day_cos`: 日の周期性

## データの正規化

- 売上データ: MinMaxScaler使用
- 特徴量: 各特徴量ごとにMinMaxScaler使用
- スケーラーは予測時の逆変換用に保存

## キャッシュシステム

### 目的
- データ前処理時間の短縮
- 計算リソースの効率的利用
- 一貫した前処理結果の保証

### 保存されるデータ
- 前処理済みシーケンス
- ターゲット値
- スケーラー

### キャッシュの場所
```
cache/
└── preprocessed_data/
    ├── train_360_90_sequences.npy    # 入力シーケンス
    ├── train_360_90_targets.npy      # ターゲット値
    └── train_360_90_scaler.joblib    # スケーラー
```

### キャッシュの更新
以下の場合にキャッシュは自動的に再生成されます：
- キャッシュファイルが存在しない
- 入力パラメータ（シーケンス長など）が変更された
- 元データが更新された
