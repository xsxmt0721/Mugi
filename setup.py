from core.database import GraphStorageEngine
from datetime import datetime
from core.config import *
import json
import os

class SETUP_CONFIG:
    # 图谱存储元数据
    META_DATA = { 
        "name": "Mugi_Core", 
        "version": "1.0.1", 
        }

    # ---------------------------
    # 图谱存储配置
    # ---------------------------
    # 图谱最大节点数
    MAX_NODES = 10000 
    # 图谱最大边数
    MAX_EDGES = 500000 
    # 条件权重向量尺寸
    W_LENGTH = 32

if __name__ == "__main__":
    # 实例化引擎
    engine = GraphStorageEngine()

    # -----------------------------------
    # 1. 清除旧系统 (核心新增步骤)
    # -----------------------------------
    engine.clear_old_system(GRAPH_DATA_DIR)

    # -----------------------------------
    # 2. 建立新图谱存储系统
    # -----------------------------------
    # initialize_system 内部会重新创建文件和 Neo4j 约束
    engine.initialize_system(
        max_nodes=SETUP_CONFIG.MAX_NODES, 
        max_edges=SETUP_CONFIG.MAX_EDGES, 
        w_vector_length=SETUP_CONFIG.W_LENGTH, 
        system_meta=SETUP_CONFIG.META_DATA
    )
    
    # -----------------------------------
    # 3. 写入新的 Manifest
    # -----------------------------------
    manifest = {
        **SETUP_CONFIG.META_DATA,
        "max_nodes": SETUP_CONFIG.MAX_NODES,
        "max_edges": SETUP_CONFIG.MAX_EDGES,
        "w_length": SETUP_CONFIG.W_LENGTH,
        "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "storage_layout": {
            "node_vectors": NODE_VECTORS_FILE,
            "edge_weights": EDGE_WEIGHTS_FILE,
            "metadata_db": META_DB_NAME
        }
    }
    
    with open(os.path.join(GRAPH_DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    engine.close()
    print("\n✅ New Mugi system environment is ready.")