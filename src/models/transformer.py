"""
Transformerモデルの実装
"""
import numpy as np
import torch
import torch.nn as nn
from config.config import TRANSFORMER_CONFIG, OUTPUT_SEQUENCE_LENGTH

class PositionalEncoding(nn.Module):
    """位置エンコーディング層"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 入力テンソル [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class SalesTransformer(nn.Module):
    """売上予測用Transformerモデル"""
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        """
        Args:
            input_size (int): 入力特徴量の次元数
            output_size (int): 出力系列の長さ
            d_model (int): モデルの隠れ層の次元数
            nhead (int): マルチヘッドアテンションのヘッド数
            num_layers (int): Transformerレイヤーの数
            dropout (float): ドロップアウト率
        """
        super(SalesTransformer, self).__init__()
        
        # より深い埋め込み層
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout/2)
        
        # 拡張されたTransformer層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=decoder_norm
        )

        # 中間層の追加
        self.intermediate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 出力層の改善
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)  # 各時点で1つの値を予測
        )
        
        self.d_model = d_model
        self.output_size = output_size

        # 重みの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """モデルの重みを初期化する関数"""
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                # Linearレイヤーの初期化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNormの初期化
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.TransformerEncoder) or isinstance(m, nn.TransformerDecoder):
                # Transformer層の初期化
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
    
        # モデル全体に初期化を適用
        self.apply(_weights_init)

    def _generate_square_subsequent_mask(self, sz):
        """後続の位置をマスクする"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        Args:
            src: 入力テンソル [batch_size, seq_length, input_size]
        Returns:
            出力テンソル [batch_size, output_sequence_length]
        """
        device = src.device
        
        # 入力の埋め込みと位置エンコーディング
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # エンコーダー処理
        memory = self.transformer_encoder(src)
        
        # デコーダー入力として最後の入力シーケンスを使用
        last_input = memory[:, -1:, :]
        tgt = last_input.expand(-1, self.output_size, -1)
        tgt = self.pos_encoder(tgt)
        
        # デコーダーのマスク生成
        tgt_mask = self._generate_square_subsequent_mask(self.output_size).to(device)
        
        # デコーダー処理
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 中間層での処理
        output = self.intermediate(output)
        
        # 各時点での予測
        output = self.fc(output).squeeze(-1)  # [batch_size, output_sequence_length]
        
        return output

def create_model(model_type, input_size):
    """モデルを作成する関数

    Args:
        model_type (str): モデルのタイプ
        input_size (int): 入力特徴量の次元数

    Returns:
        nn.Module: 作成されたモデル

    Raises:
        ValueError: サポートされていないモデルタイプが指定された場合
    """
    if model_type == "transformer":
        return SalesTransformer(
            input_size=input_size,
            output_size=OUTPUT_SEQUENCE_LENGTH,
            d_model=TRANSFORMER_CONFIG["d_model"],
            nhead=TRANSFORMER_CONFIG["nhead"],
            num_layers=TRANSFORMER_CONFIG["num_layers"],
            dropout=TRANSFORMER_CONFIG["dropout"]
        )
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {model_type}")
