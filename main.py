"""
売上予測モデルの学習実行スクリプト
"""
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from accelerate import Accelerator

from config.config import (
    INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, MODEL_TYPE, NUM_WORKERS, PIN_MEMORY,
    TRAIN_DATA_PATH, DEFAULT_SEED, EARLY_STOPPING_PATIENCE
)
from src.data.dataset import SalesDataset
from src.data.preprocessing import load_and_preprocess_data
from src.models.transformer import create_model
from src.models.utils import (
    fix_seed, setup_training_device, 
    create_optimizer_and_scheduler, create_loss_function
)
from src.training.trainer import ModelTrainer, setup_training
from src.training.metrics import evaluate_model
from src.utils.visualization import plot_training_history

def main():
    """メイン実行関数"""
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="時系列予測モデルの学習")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="乱数シード")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # シード値の固定 (Accelerator のランクを使用)
    fix_seed(args.seed, accelerator.process_index)

    # CUDA関連の設定
    setup_training_device()

    # データの読み込みと前処理
    sequences, targets, sales_scaler = load_and_preprocess_data(
        TRAIN_DATA_PATH, 
        INPUT_SEQUENCE_LENGTH, 
        OUTPUT_SEQUENCE_LENGTH
    )

    # データの分割
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=args.seed
    )

    # データセットとデータローダーの作成
    train_dataset = SalesDataset(X_train, y_train)
    val_dataset = SalesDataset(X_val, y_val)
    test_dataset = SalesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )

    # モデルの作成
    input_size = X_train.shape[2]  # 入力特徴量の次元数
    model = create_model(MODEL_TYPE, input_size)

    # オプティマイザ、スケジューラ、損失関数の設定
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        LEARNING_RATE, 
        EPOCHS, 
        len(train_loader)
    )
    criterion = create_loss_function()

    # データローダー、モデル、オプティマイザの準備
    train_loader, val_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, test_loader, model, optimizer
    )

    # 結果保存用ディレクトリの作成
    results_dir = setup_training(MODEL_TYPE)

    # トレーナーの初期化と学習の実行
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

    # モデルの学習
    train_losses, val_losses, best_val_loss = trainer.train(
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE
    )

    # 学習履歴のプロット
    plot_training_history(
        train_losses,
        val_losses,
        f"{results_dir}/training_history.png"
    )

    # テストデータでの評価
    test_metrics, predictions, targets = evaluate_model(
        model,
        test_loader,
        sales_scaler
    )

    # テスト結果の表示
    print("\nTest Results:")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"RMSLE: {test_metrics['rmsle']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}%")
    print(f"R2 Score: {test_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()
