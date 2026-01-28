import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
import time

from core.schema import MugiNode, MugiEdge
from utils.utils_neo4j import (
    Neo4jHandler,
    upload_node,
    upload_edge,
    delete_node_by_id,
    delete_edge_by_id,
    print_all_nodes_in_order,
    print_all_edges_in_order,
)

def test_storage_pipeline():
    print("\n[!] Starting Mugi Storage Pipeline Test...")
    handler = Neo4jHandler()
    # 1) 添加节点 A 并打印 id
    node_a = MugiNode(
        category="TestEntity",
        display_name="Node_A",
        vector=[random.random() for _ in range(768)],
        infer_count=10,
        last_active=time.time(),
        node_rank=1
    )
    id_a = upload_node(node_a, handler)
    print(f"[1] Node A ID: {id_a}")

    # 2) 添加节点 B 并打印 id
    node_b = MugiNode(
        category="TestEntity",
        display_name="Node_B",
        vector=[random.random() for _ in range(768)],
        infer_count=5,
        last_active=time.time(),
        node_rank=2
    )
    id_b = upload_node(node_b, handler)
    print(f"[2] Node B ID: {id_b}")

    # 3) 添加节点 C 并打印 id
    node_c = MugiNode(
        category="TestEntity",
        display_name="Node_C",
        vector=[random.random() for _ in range(768)],
        infer_count=8,
        last_active=time.time(),
        node_rank=3
    )
    id_c = upload_node(node_c, handler)
    print(f"[3] Node C ID: {id_c}")

    # 4) 删除节点 B
    delete_node_by_id(id_b, handler)
    print("[4] Deleted Node B")

    # 5) 按顺序打印当前图谱中的所有节点（期望：A, C）
    print("[5] Nodes in order (expect A, C):")
    print_all_nodes_in_order(handler)
    
    # 6) 添加节点 B 并打印 id（空闲池复用）
    node_b2 = MugiNode(
        category="TestEntity",
        display_name="Node_B",
        vector=[random.random() for _ in range(768)],
        infer_count=6,
        last_active=time.time(),
        node_rank=2
    )
    id_b2 = upload_node(node_b2, handler)
    print(f"[6] Node B (re-added) ID: {id_b2}")

    # 7) 添加关系 AB，AC，BC 并打印 id
    edge_ab = MugiEdge(
        source_id=id_a,
        target_id=id_b2,
        w_vector=[random.random() for _ in range(32)],
        r_intrinsic=0.88,
        b_bias=0.12,
        description="AB",
        status="active"
    )
    id_ab = upload_edge(edge_ab, handler)
    print(f"[7] Edge AB ID: {id_ab}")

    edge_ac = MugiEdge(
        source_id=id_a,
        target_id=id_c,
        w_vector=[random.random() for _ in range(32)],
        r_intrinsic=0.77,
        b_bias=0.11,
        description="AC",
        status="active"
    )
    id_ac = upload_edge(edge_ac, handler)
    print(f"[7] Edge AC ID: {id_ac}")

    edge_bc = MugiEdge(
        source_id=id_b2,
        target_id=id_c,
        w_vector=[random.random() for _ in range(32)],
        r_intrinsic=0.66,
        b_bias=0.10,
        description="BC",
        status="active"
    )
    id_bc = upload_edge(edge_bc, handler)
    print(f"[7] Edge BC ID: {id_bc}")

    # 8) 删除边 AC
    delete_edge_by_id(id_ac, handler)
    print("[8] Deleted Edge AC")

    # 9) 按顺序打印当前图谱中的所有边（期望：AB, BC）
    print("[9] Edges in order (expect AB, BC):")
    print_all_edges_in_order(handler)

    # 10) 添加关系 AC 并打印 id（空闲池复用）
    edge_ac2 = MugiEdge(
        source_id=id_a,
        target_id=id_c,
        w_vector=[random.random() for _ in range(32)],
        r_intrinsic=0.55,
        b_bias=0.09,
        description="AC",
        status="active"
    )
    id_ac2 = upload_edge(edge_ac2, handler)
    print(f"[10] Edge AC (re-added) ID: {id_ac2}")

    '''
    # 11) 删除节点 B
    delete_node_by_id(id_b2, handler)
    print("[11] Deleted Node B")

    # 12) 按顺序打印当前图谱中所有的边（期望：AC）
    print("[12] Edges in order (expect AC):")
    print_all_edges_in_order(handler)
    '''

    handler.close()
    print("\n[!] Pipeline Test Completed.")

if __name__ == "__main__":
    try:
        test_storage_pipeline()
    except Exception as e:
        print(f"Test crashed: {e}")