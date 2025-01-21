"""
データの前処理とフィーチャーエンジニアリング
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from config.config import CACHE_DIR

def get_cache_path(data_path, input_len, output_len):
    """キャッシュファイルのパスを生成

    Args:
        data_path (str): 元データのパス
        input_len (int): 入力シーケンスの長さ
        output_len (int): 出力シーケンスの長さ

    Returns:
        tuple: (sequences_path, targets_path, scaler_path)
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 入力データのハッシュ値を計算（データパスと設定値から）
    base_name = f"{os.path.basename(data_path)}_{input_len}_{output_len}"
    
    return (
        os.path.join(CACHE_DIR, f"{base_name}_sequences.npy"),
        os.path.join(CACHE_DIR, f"{base_name}_targets.npy"),
        os.path.join(CACHE_DIR, f"{base_name}_scaler.joblib")
    )

def check_cache_exists(data_path, input_len, output_len):
    """キャッシュファイルが存在するかチェック

    Args:
        data_path (str): 元データのパス
        input_len (int): 入力シーケンスの長さ
        output_len (int): 出力シーケンスの長さ

    Returns:
        bool: すべてのキャッシュファイルが存在する場合True
    """
    seq_path, tgt_path, scaler_path = get_cache_path(data_path, input_len, output_len)
    return all(os.path.exists(p) for p in [seq_path, tgt_path, scaler_path])

def save_to_cache(sequences, targets, scaler, data_path, input_len, output_len):
    """前処理済みデータをキャッシュとして保存

    Args:
        sequences (np.ndarray): 入力シーケンス
        targets (np.ndarray): 目標値
        scaler (MinMaxScaler): スケーラー
        data_path (str): 元データのパス
        input_len (int): 入力シーケンスの長さ
        output_len (int): 出力シーケンスの長さ
    """
    seq_path, tgt_path, scaler_path = get_cache_path(data_path, input_len, output_len)
    
    np.save(seq_path, sequences)
    np.save(tgt_path, targets)
    joblib.dump(scaler, scaler_path)

def load_from_cache(data_path, input_len, output_len):
    """キャッシュから前処理済みデータを読み込む

    Args:
        data_path (str): 元データのパス
        input_len (int): 入力シーケンスの長さ
        output_len (int): 出力シーケンスの長さ

    Returns:
        tuple: (sequences, targets, scaler)
    """
    seq_path, tgt_path, scaler_path = get_cache_path(data_path, input_len, output_len)
    
    sequences = np.load(seq_path)
    targets = np.load(tgt_path)
    scaler = joblib.load(scaler_path)
    
    return sequences, targets, scaler

def create_features(df):
    """
    時系列データの特徴量を作成

    Args:
        df (pd.DataFrame): 日付カラムを含む元データ

    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
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

def preprocess_data(data, input_len, output_len, sales_scaler=None):
    """
    時系列データを学習用に前処理

    Args:
        data (pd.DataFrame): 前処理するデータフレーム
        input_len (int): 入力シーケンスの長さ
        output_len (int): 出力シーケンスの長さ
        sales_scaler (sklearn.preprocessing.MinMaxScaler, optional): 売上のスケーラー

    Returns:
        tuple: (シーケンスデータ, ターゲットデータ, 使用したスケーラー)
    """
    sequences = []
    targets = []
    features = []
    unique_stores = data['store_nbr'].unique()

    # スケーラーの初期化
    if sales_scaler is None:
        sales_scaler = MinMaxScaler()

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

    return combined_sequences, np.array(targets), sales_scaler

def load_and_preprocess_data(data_path, input_len, output_len):
    """
    データの読み込みと前処理を一括で行う

    Args:
        data_path (str): データファイルのパス
        input_len (int): 入力シーケンスの長さ
        output_len (int): 出力シーケンスの長さ

    Returns:
        tuple: (前処理済みデータ, スケーラー)
    """
    # キャッシュがある場合はそこから読み込む
    if check_cache_exists(data_path, input_len, output_len):
        print("Load cache data...")
        return load_from_cache(data_path, input_len, output_len)

    print("Preprocess data...")
    # データの読み込み
    data = pd.read_csv(data_path, parse_dates=['date'])
    data = data[data['family'] == 'GROCERY I'].copy()
    data = data.sort_values(['store_nbr', 'date']).reset_index(drop=True)

    # 特徴量エンジニアリング
    data = create_features(data)

    # データの前処理
    sequences, targets, scaler = preprocess_data(data, input_len, output_len)

    # 前処理済みデータをキャッシュとして保存
    print("Save cache data...")
    save_to_cache(sequences, targets, scaler, data_path, input_len, output_len)

    return sequences, targets, scaler
