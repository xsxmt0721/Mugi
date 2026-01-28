import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from core.config import *
from utils.utils_neo4j import Neo4jHandler


class GraphStorageEngine:
    def __init__(self, data_dir: str = GRAPH_DATA_DIR):
        """
        图存储引擎：统一管理 Neo4j 拓扑、本地向量库和元数据
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径定义
        self.node_vec_path = self.data_dir / NODE_VECTORS_FILE
        self.edge_w_path = self.data_dir / EDGE_WEIGHTS_FILE
        self.meta_db_path = self.data_dir / META_DB_NAME
        self.system_meta_path = self.data_dir / SYSTEM_META_FILE

        # Neo4j 连接
        self.neo4j = Neo4jHandler()

    def initialize_system(self, 
                          max_nodes: int, 
                          max_edges: int, 
                          w_vector_length: int, 
                          system_meta: Dict[str, Any]):
        """
        【核心初始化函数】
        一键重置并初始化：Neo4j 图谱、向量库、元数据库
        """
        print(f"[*] Starting Mugi System Initialization...")
        print(f"    - Max Nodes: {max_nodes}")
        print(f"    - Max Edges: {max_edges}")
        print(f"    - W Vector Dim: {w_vector_length}")

        # ==========================================
        # 1. 初始化 Neo4j (拓扑层)
        # ==========================================
        print("[1/4] Initializing Neo4j Topology...")
        with self.neo4j.driver.session() as session:
            # 1.1 清空数据库 (危险操作，仅初始化用)
            session.run("MATCH (n) DETACH DELETE n")
            # 1.2 删除旧约束
            try:
                session.run("DROP CONSTRAINT node_id_unique IF EXISTS")
            except:
                pass 
            # 1.3 建立 ID 唯一约束 (关键索引)
            # 注意：Neo4j 5.x 语法可能略有不同，这是通用写法
            session.run("CREATE CONSTRAINT node_id_unique FOR (n:MugiNode) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT edge_id_unique FOR ()-[r:RELATED]-() REQUIRE r.id IS UNIQUE")

            # 1.4 常用检索索引（加速筛选/排序）
            # 节点：常用属性索引
            session.run("CREATE INDEX node_category_idx IF NOT EXISTS FOR (n:MugiNode) ON (n.category)")
            session.run("CREATE INDEX node_infer_count_idx IF NOT EXISTS FOR (n:MugiNode) ON (n.infer_count)")
            session.run("CREATE INDEX node_rank_idx IF NOT EXISTS FOR (n:MugiNode) ON (n.node_rank)")

            # 边：常用属性索引（关系属性索引）
            session.run("CREATE INDEX edge_r_intrinsic_idx IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.r_intrinsic)")
            session.run("CREATE INDEX edge_psi_value_idx IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.psi_value)")
            session.run("CREATE INDEX edge_infer_count_idx IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.infer_count)")
        
        # ==========================================
        # 2. 初始化 Tensor 库 (向量层 - NumPy Memmap)
        # ==========================================
        print("[2/4] Allocating Disk Space for Tensors...")
        
        # 2.1 节点语义向量库 (BERT Vectors) -> float32
        # 创建一个内存映射文件，填零初始化
        fp_nodes = np.memmap(self.node_vec_path, dtype='float32', mode='w+', shape=(max_nodes, 768))
        del fp_nodes # flush to disk

        # 2.2 边权重向量库 (W Vectors) -> float32
        fp_edges = np.memmap(self.edge_w_path, dtype='float32', mode='w+', shape=(max_edges, w_vector_length))
        del fp_edges # flush to disk

        # ==========================================
        # 3. 初始化 Metadata 库 (文本层 - SQLite)
        # ==========================================
        print("[3/4] Building Metadata SQL Database...")
        if self.meta_db_path.exists():
            os.remove(self.meta_db_path)
            
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        
        # 3.1 节点元数据表 (支持随机存取)
        cursor.execute('''
            CREATE TABLE node_meta (
                id INTEGER PRIMARY KEY,
                category TEXT,
                display_name TEXT,
                infer_count INTEGER DEFAULT 0,
                last_active REAL DEFAULT 0.0,
                node_rank INTEGER DEFAULT 0,
                full_json TEXT  -- 存储完整的 Schema JSON
            )
        ''')
        
        # 3.2 边元数据表
        cursor.execute('''
            CREATE TABLE edge_meta (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                target_id INTEGER,
                r_intrinsic REAL,
                b_bias REAL,
                psi_value REAL DEFAULT 0.0,
                entropy REAL DEFAULT 0.0,
                variance REAL DEFAULT 0.0,
                infer_count INTEGER DEFAULT 0,
                description TEXT,
                last_labeled_time REAL DEFAULT 0.0,
                version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                full_json TEXT
            )
        ''')

        # 3.3 ID 复用池（链表格式，记录连续空闲区间）
        cursor.execute('''
            CREATE TABLE node_id_reuse_pool (
                start INTEGER PRIMARY KEY,
                length INTEGER NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE edge_id_reuse_pool (
                start INTEGER PRIMARY KEY,
                length INTEGER NOT NULL
            )
        ''')

        # 初始化为全量空闲区间
        cursor.execute('INSERT INTO node_id_reuse_pool (start, length) VALUES (?, ?)', (0, max_nodes))
        cursor.execute('INSERT INTO edge_id_reuse_pool (start, length) VALUES (?, ?)', (0, max_edges))
        conn.commit()
        conn.close()

        # ==========================================
        # 4. 写入系统元数据 (System Meta)
        # ==========================================
        print("[4/4] Writing System Manifest...")
        manifest = {
            "meta_data": system_meta,
            "config": {
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "w_dim": w_vector_length,
                "node_vec_shape": [max_nodes, 768],
                "edge_w_shape": [max_edges, w_vector_length]
            },
            "status": "initialized"
        }
        with open(self.system_meta_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        print("\n✅ Mugi System Initialized Successfully!")
        print(f"   Data Location: {self.data_dir}")

    def close(self):
        self.neo4j.close()

    def clear_old_system(self, storage_dir: str):
        """
        彻底清除旧的存储系统，包括 Neo4j 拓扑和本地文件
        """
        print(f"[*] Warning: Clearing old system data in {storage_dir}...")

        # 1. 清理 Neo4j 拓扑与模式
        try:
            with self.neo4j.driver.session() as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                # 获取并删除所有约束 (Neo4j 5.x 兼容写法)
                constraints = session.run("SHOW CONSTRAINTS")
                for record in constraints:
                    session.run(f"DROP CONSTRAINT {record['name']}")
                print("    - Neo4j topology and constraints cleared.")
        except Exception as e:
            print(f"    - Neo4j clear warning: {e} (Maybe already empty)")

        # 2. 清理本地物理文件
        if os.path.exists(storage_dir):
            # 采用直接删除整个目录再重建的方式最为彻底
            try:
                shutil.rmtree(storage_dir)
                os.makedirs(storage_dir)
                print(f"    - Local storage directory {storage_dir} formatted.")
            except Exception as e:
                print(f"    - Local file clear error: {e}")
        else:
            os.makedirs(storage_dir)