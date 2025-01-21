"""
モデル関連のユーティリティ関数
"""
import random
import numpy as np
import torch
from accelerate import Accelerator
from timm.scheduler.cosine_lr import CosineLRScheduler

def fix_seed(seed, rank=0):
    """乱数シードを固定する

    Args:
        seed (int): 固定するシード値
        rank (int, optional): 分散学習時のランク. デフォルトは0
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def get_device():
    """学習に使用するデバイスを取得する

    Returns:
        torch.device: 使用するデバイス
    """
    accelerator = Accelerator()
    return accelerator.device

def setup_training_device():
    """CUDA関連の設定を行う"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False

def create_optimizer_and_scheduler(model, learning_rate, num_epochs, num_training_steps):
    """オプティマイザとスケジューラを作成する

    Args:
        model (nn.Module): 学習対象のモデル
        learning_rate (float): 学習率
        num_epochs (int): エポック数
        num_training_steps (int): 1エポックあたりの学習ステップ数

    Returns:
        tuple: (オプティマイザ, スケジューラ)
    """
    # パラメータグループごとに異なる設定を適用
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 改善されたCosine Learning Rate Scheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs * num_training_steps,
        lr_min=1e-7,
        warmup_lr_init=1e-6,
        warmup_t=int(0.15 * num_epochs * num_training_steps),  # より長いウォームアップ
        cycle_limit=1,  # サイクル数を減らして安定性を向上
        warmup_prefix=True
    )

    return optimizer, scheduler

def create_loss_function():
    """複合損失関数を作成する

    Returns:
        function: 損失関数
    """
    def custom_loss(pred, target):
        # MSE損失
        mse_loss = torch.nn.MSELoss()(pred, target)
        
        # Huber損失（外れ値に対してより頑健）
        huber_loss = torch.nn.HuberLoss(delta=1.0)(pred, target)
        
        # MAE損失（絶対誤差）
        mae_loss = torch.nn.L1Loss()(pred, target)
        
        # 方向性の一致を評価する損失
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        direction_loss = -torch.mean(torch.sign(pred_diff) * torch.sign(target_diff))
        
        # 損失の重み付け結合
        total_loss = 0.4 * mse_loss + 0.3 * huber_loss + 0.2 * mae_loss + 0.1 * direction_loss
        return total_loss

    return custom_loss
