import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
from typing import Dict, Any, Tuple

import numpy as np

from utils.utils_neo4j import (
	Neo4jHandler,
	get_node_meta_by_id,
	get_node_vector_by_id,
	update_node_meta_by_id,
	update_node_vector_by_id,
	get_edge_meta_by_id,
	get_edge_vector_by_id,
	update_edge_meta_by_id,
	update_edge_vector_by_id,
	get_outgoing_edges_by_source_id,
	get_incoming_edges_by_target_id,
)


def _print_node_full(node_id: int) -> None:
	meta = get_node_meta_by_id(node_id)
	vector = get_node_vector_by_id(node_id)
	print(f"Node[{node_id}] meta=", meta)
	print(f"Node[{node_id}] vector(first10)=", vector[:10])


def _print_edge_full(edge_id: int) -> None:
	meta = get_edge_meta_by_id(edge_id)
	w_vector = get_edge_vector_by_id(edge_id)
	print(f"Edge[{edge_id}] meta=", meta)
	print(f"Edge[{edge_id}] w_vector(first10)=", w_vector[:10])


def test_query_update():
	handler = Neo4jHandler()

	node_id = 1
	print("[1] 获取节点元数据+向量")
	_print_node_full(node_id)

	print("[2] 修改节点元数据（随机）")
	node_updates = {
		"category": f"TestEntity_{random.randint(1, 9)}",
		"display_name": f"RandomNode_{random.randint(100, 999)}",
		"infer_count": random.randint(0, 1000),
		"last_active": random.random() * 1e6,
		"node_rank": random.randint(0, 100),
	}
	update_node_meta_by_id(node_id, node_updates, handler)

	print("[3] 重新获取节点数据并对比拓扑层")
	meta = get_node_meta_by_id(node_id)
	print("Meta:", meta)
	with handler.driver.session() as session:
		record = session.run(
			"""
			MATCH (n:MugiNode {id: $id})
			RETURN n.category AS category, n.display_name AS display_name,
				n.infer_count AS infer_count, n.last_active AS last_active,
				n.node_rank AS node_rank
			""",
			{"id": node_id},
		).single()
		neo = dict(record) if record else {}
	print("Topo:", neo)
	consistent = all(meta.get(k) == neo.get(k) for k in node_updates.keys())
	print("Meta/Topo一致:", consistent)

	print("[4] 修改节点向量（随机）")
	new_vector = np.random.rand(768).astype("float32").tolist()
	update_node_vector_by_id(node_id, new_vector)
	print("Vector updated (first10)=", new_vector[:10])

	print("[5] 重新获取向量并验证")
	updated_vector = get_node_vector_by_id(node_id)
	print("Vector verify:", np.allclose(updated_vector, new_vector, atol=1e-6))

	print("[6] 打印该节点作为 source/target 的边")
	outgoing = get_outgoing_edges_by_source_id(node_id, handler)
	incoming = get_incoming_edges_by_target_id(node_id, handler)
	print("Outgoing edges:")
	for edge_id, target_id in outgoing.items():
		print(f"[{node_id}, {target_id}, {edge_id}]")
	print("Incoming edges:")
	for edge_id, source_id in incoming.items():
		print(f"[{source_id}, {node_id}, {edge_id}]")

	print("[7] 选择第一条边并打印其全部数据")
	first_edge_id: int | None = None
	if outgoing:
		first_edge_id = next(iter(outgoing.keys()))
	elif incoming:
		first_edge_id = next(iter(incoming.keys()))
	if first_edge_id is None:
		print("No edges found for this node.")
		handler.close()
		return
	_print_edge_full(first_edge_id)

	print("[8] 修改边元数据并用拓扑层对比")
	edge_updates = {
		"r_intrinsic": random.random(),
		"b_bias": random.random(),
		"psi_value": random.random(),
		"entropy": random.random(),
		"variance": random.random(),
		"infer_count": random.randint(0, 1000),
		"description": f"rand_desc_{random.randint(100, 999)}",
		"last_labeled_time": random.random() * 1e6,
		"version": random.randint(1, 10),
		"status": random.choice(["active", "candidate"]),
	}
	update_edge_meta_by_id(first_edge_id, edge_updates, handler)
	meta_edge = get_edge_meta_by_id(first_edge_id)
	with handler.driver.session() as session:
		record = session.run(
			"""
			MATCH ()-[r:RELATED {id: $id}]->()
			RETURN r.r_intrinsic AS r_intrinsic, r.b_bias AS b_bias,
				r.psi_value AS psi_value, r.entropy AS entropy,
				r.variance AS variance, r.infer_count AS infer_count,
				r.description AS description, r.last_labeled_time AS last_labeled_time,
				r.version AS version, r.status AS status
			""",
			{"id": first_edge_id},
		).single()
		neo_edge = dict(record) if record else {}
	consistent_edge = all(meta_edge.get(k) == neo_edge.get(k) for k in edge_updates.keys())
	print("Edge Meta/Topo一致:", consistent_edge)

	print("[9] 修改边向量并验证")
	new_w_vector = np.random.rand(len(get_edge_vector_by_id(first_edge_id))).astype("float32").tolist()
	update_edge_vector_by_id(first_edge_id, new_w_vector)
	updated_w_vector = get_edge_vector_by_id(first_edge_id)
	print("Edge vector verify:", np.allclose(updated_w_vector, new_w_vector, atol=1e-6))

	handler.close()


if __name__ == "__main__":
	test_query_update()
