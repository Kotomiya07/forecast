"""
設定値とハイパーパラメータの定義
"""

# データ設定
INPUT_SEQUENCE_LENGTH = 360  # 1年
OUTPUT_SEQUENCE_LENGTH = 90  # 3ヶ月
NUM_WORKERS = 16
PIN_MEMORY = True

# トレーニング設定
BATCH_SIZE = 64  # より小さいバッチサイズで細かく学習
EPOCHS = 1000  # エポック数を増やして学習を深める
LEARNING_RATE = 5e-5  # より小さい学習率で安定した学習
EARLY_STOPPING_PATIENCE = 20  # より長い忍耐値で学習を継続

# モデル設定
MODEL_TYPE = "transformer"
TRANSFORMER_CONFIG = {
    "d_model": 256,  # モデルの表現力を大幅に向上
    "nhead": 8,      # より多くのアテンションヘッド
    "num_layers": 4, # より深いTransformer
    "dropout": 0.1,  # 過学習を抑制しつつ、情報をより保持
}

# 乱数シード
DEFAULT_SEED = 42

# データパス
TRAIN_DATA_PATH = "data/store-sales/train.csv"

# キャッシュ設定
CACHE_DIR = "cache/preprocessed_data"

# 結果保存設定
PLOT_SAVE_INTERVAL = 1  # 1エポックごとにプロット
