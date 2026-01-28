import torch
import numpy as np
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# ==========================================
# 1. 节点定义 (Node Schema)
# ==========================================
class MugiNode(BaseModel):
    """
    Mugi 节点数据结构
    存储于 Neo4j，并在 Core 中用于语义对齐
    """
    id: int = Field(default=-1, description="唯一标识符，通常为一串数字编号")
    category: str = Field(default="", description="所属集合 V，如 Diagnosis, Symptoms")
    vector: List[float] = Field(..., description="Bert 生成的 768 维语义向量")
    
    # 工程辅助字段
    infer_count: int = Field(default=0, description="节点参与推理的次数")
    display_name: str = Field(default="", description="人类可读的中文名称")
    last_active: float = Field(default=0.0, description="节点最后一次参与推理的时间戳")
    node_rank: int = Field(default=0, description="节点连接的边数，用于计算边权")

# ==========================================
# 2. 边定义 (Edge Schema)
# ==========================================
class MugiEdge(BaseModel):
    """
    Mugi 边数据结构
    包含 Mugi 核心参数 r, w, b 以及用于主动学习的辅助参数
    """
    # 拓扑信息
    id: int = Field(default=-1, description="边 ID (e_ij)")
    source_id: int = Field(default=-1, description="源节点 ID (v_i)")
    target_id: int = Field(default=-1, description="目标节点 ID (v_j)")
    
    # Mugi 核心数学参数
    r_intrinsic: float = Field(default=0.5, description="本征相关性 r_ij")
    w_vector: List[float] = Field(..., description="条件权重向量 w_ij，长度对应 x 的维度 (m+n)")
    b_bias: float = Field(default=0.0, description="条件偏置值 b_ij")
    
    # 演化与主动学习参数
    psi_value: float = Field(default=0.0, description="可疑程度 Psi，用于主动学习推送")
    entropy: float = Field(default=0.0, description="模糊程度 H_ij，用于计算 Psi")
    variance: float = Field(default=0.0, description="推理结果的标准差 sigma_ij")
    
    # 工程管理字段
    infer_count: int = Field(default=0, description="边参与推理的次数")
    description: Optional[str] = Field(default="", description="人类可读的文本描述")
    last_labeled_time: float = Field(default=0.0, description="上次专家标注的时间，用于计算 C_ij")
    version: int = Field(default=1, description="参数演化版本号")
    status: str = Field(default="active", description="边状态: active(已激活), candidate(LLM生成的候选边)")

# ==========================================
# 3. 条件特征向量 (X Vector Schema)
# ==========================================
class PatientCondition(BaseModel):
    """
    条件特征向量 x
    """
    x1_quant: List[float] = Field(..., description="归一化后的可量化指标")
    x2_residual: List[float] = Field(..., description="文本语意残差向量")
    
    def to_Tensor(self) -> torch.Tensor:
        return torch.Tensor(self.x1_quant + self.x2_residual)

if __name__ == "__main__":
    # 测试节点和边的创建
    node = MugiNode(
        id="123",
        category="Diagnosis",
        vector=np.random.rand(768).tolist(),
        display_name="示例节点"
    )
    
    edge = MugiEdge(
        source_id="123",
        target_id="456",
        r_intrinsic=0.7,
        w_vector=np.random.rand(10).tolist(),
        b_bias=0.1
    )
    
    print(node)
    print(edge)