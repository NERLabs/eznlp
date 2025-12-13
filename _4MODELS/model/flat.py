import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FLATModel(nn.Module):
    """
    FLAT (Flat-Lattice Transformer) 模型核心结构
    用于序列标注任务，特别是中文NER
    """
    
    def __init__(self, 
                 vocab_size,           # 词汇表大小
                 char_vocab_size,      # 字符表大小
                 bigram_vocab_size,    # 双字符表大小
                 hidden_size=256,      # 隐藏层大小
                 num_heads=8,          # 注意力头数
                 num_layers=6,         # Transformer层数
                 num_labels=10,        # 标签数量
                 max_seq_len=512,      # 最大序列长度
                 use_bigram=True,      # 是否使用双字符信息
                 use_rel_pos=True,     # 是否使用相对位置编码
                 use_abs_pos=True,     # 是否使用绝对位置编码
                 dropout=0.1):         # Dropout率
        super(FLATModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.use_bigram = use_bigram
        self.use_rel_pos = use_rel_pos
        self.use_abs_pos = use_abs_pos
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.lattice_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.bigram_embed = nn.Embedding(bigram_vocab_size, hidden_size//2, padding_idx=0)
        
        # 字符嵌入维度
        if use_bigram:
            self.char_input_size = hidden_size + hidden_size//2
        else:
            self.char_input_size = hidden_size
            
        # 词汇嵌入维度
        self.lex_input_size = hidden_size
        
        # 投影层
        self.char_proj = nn.Linear(self.char_input_size, hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size, hidden_size)
        
        # 位置编码
        if use_abs_pos:
            self.abs_pos_encode = AbsoluteSEPositionEmbedding(hidden_size)
            
        if use_rel_pos:
            self.rel_pos_embedding = RelativePositionEmbedding(max_seq_len, hidden_size)
            self.four_pos_fusion = FourPosFusion(hidden_size)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                lattice,      # 词序列 [batch_size, seq_len+lex_num]
                bigrams,      # 双字符序列 [batch_size, seq_len]
                seq_len,      # 序列长度 [batch_size]
                lex_num,      # 词汇数量 [batch_size]
                pos_s,        # 词汇起始位置 [batch_size, seq_len+lex_num]
                pos_e):       # 词汇结束位置 [batch_size, seq_len+lex_num]
        """
        前向传播
        """
        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)
        
        # 嵌入层
        raw_embed = self.lattice_embed(lattice)  # 词汇嵌入
        
        if self.use_bigram:
            bigrams_embed = self.bigram_embed(bigrams)
            # 扩展bigram嵌入以匹配lattice序列长度
            bigrams_embed = torch.cat([
                bigrams_embed,
                torch.zeros(
                    size=[batch_size, max_seq_len_and_lex_num - max_seq_len, bigrams_embed.size(-1)],
                    device=bigrams_embed.device
                )
            ], dim=1)
            raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)
        else:
            raw_embed_char = raw_embed
            
        # 投影层
        embed_char = self.char_proj(raw_embed_char)
        embed_lex = self.lex_proj(raw_embed)
        
        # 合并字符和词汇嵌入
        embedding = embed_char + embed_lex
        
        # 位置编码
        if self.use_abs_pos:
            embedding = self.abs_pos_encode(embedding, pos_s, pos_e)
            
        embedding = self.dropout(embedding)
        
        # Transformer编码器
        # 注意：这里需要根据实际序列长度创建合适的mask
        mask = self.generate_mask(seq_len, lex_num, max_seq_len_and_lex_num)
        encoded = self.transformer_encoder(embedding, src_key_padding_mask=~mask)
        
        # 输出层（只取字符部分）
        encoded = encoded[:, :max_seq_len, :]
        output = self.output(encoded)
        
        return output
    
    def generate_mask(self, seq_len, lex_num, max_len):
        """
        生成注意力mask
        """
        batch_size = seq_len.size(0)
        mask = torch.arange(max_len, device=seq_len.device).expand(batch_size, max_len) < (seq_len + lex_num).unsqueeze(1)
        return mask

class AbsoluteSEPositionEmbedding(nn.Module):
    """
    起始-结束位置编码
    """
    def __init__(self, hidden_size, max_len=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(torch.randn(2 * max_len + 1, hidden_size))
        
    def forward(self, x, pos_s, pos_e):
        """
        x: 输入张量 [batch_size, seq_len, hidden_size]
        pos_s: 起始位置 [batch_size, seq_len]
        pos_e: 结束位置 [batch_size, seq_len]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 获取位置嵌入
        pe_s = self.pos_embed[pos_s.view(-1) + self.pos_embed.size(0)//2].view(batch_size, seq_len, -1)
        pe_e = self.pos_embed[pos_e.view(-1) + self.pos_embed.size(0)//2].view(batch_size, seq_len, -1)
        
        # 添加位置信息
        return x + pe_s + pe_e

class RelativePositionEmbedding(nn.Module):
    """
    相对位置编码
    """
    def __init__(self, max_seq_len, hidden_size):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.pe = nn.Parameter(self.get_embedding(2 * max_seq_len + 1, hidden_size))
        
    def get_embedding(self, num_embeddings, embedding_dim):
        """生成正弦位置编码"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb
    
    def forward(self, pos_s, pos_e):
        """
        生成相对位置嵌入
        """
        batch = pos_s.size(0)
        max_seq_len = pos_s.size(1)
        
        # 计算四种相对位置
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)
        
        # 获取对应的嵌入
        pe_ss = self.pe[pos_ss.view(-1) + self.max_seq_len].view(batch, max_seq_len, max_seq_len, -1)
        pe_se = self.pe[pos_se.view(-1) + self.max_seq_len].view(batch, max_seq_len, max_seq_len, -1)
        pe_es = self.pe[pos_es.view(-1) + self.max_seq_len].view(batch, max_seq_len, max_seq_len, -1)
        pe_ee = self.pe[pos_ee.view(-1) + self.max_seq_len].view(batch, max_seq_len, max_seq_len, -1)
        
        return pe_ss, pe_se, pe_es, pe_ee

class FourPosFusion(nn.Module):
    """
    四种位置信息融合
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, pe_ss, pe_se, pe_es, pe_ee):
        """
        融合四种位置信息
        """
        # 拼接四种位置信息
        pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
        # 融合
        rel_pos_embedding = self.fusion(pe_4)
        return rel_pos_embedding

# 使用示例
"""
model = FLATModel(
    vocab_size=10000,
    char_vocab_size=5000,
    bigram_vocab_size=8000,
    hidden_size=256,
    num_heads=8,
    num_layers=6,
    num_labels=10,
    max_seq_len=512
)

# 假设输入数据
batch_size = 4
seq_len = 50
lex_num = 20

lattice = torch.randint(0, 10000, (batch_size, seq_len + lex_num))
bigrams = torch.randint(0, 8000, (batch_size, seq_len))
seq_len_tensor = torch.tensor([seq_len] * batch_size)
lex_num_tensor = torch.tensor([lex_num] * batch_size)
pos_s = torch.randint(0, seq_len, (batch_size, seq_len + lex_num))
pos_e = torch.randint(0, seq_len, (batch_size, seq_len + lex_num))

output = model(lattice, bigrams, seq_len_tensor, lex_num_tensor, pos_s, pos_e)
print("输出形状:", output.shape)  # 应该是 [batch_size, seq_len, num_labels]
"""
