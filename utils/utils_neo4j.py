import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np
from neo4j import GraphDatabase

from core.schema import MugiNode, MugiEdge
from core.config import *

class Neo4jHandler:
    def __init__(self):
        # 这里的 URI 对应 yml 里的 bolt://mugi-db:7687
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 验证连接
            self.driver.verify_connectivity()
            print("Successfully connected to Mugi_Neo4j!")
        except Exception as e:
            print(f"Connection failed: {e}")

    def close(self):
        self.driver.close()

    def get_graph_stats(self, data_dir: str = GRAPH_DATA_DIR) -> Dict[str, Any]:
        """
        返回图谱节点/边数量，以及空闲池碎片程度。
        碎片程度 = 已存储数量 / (最后一个非空闲 id + 1)
        """
        data_path = Path(data_dir)
        node_vec_shape = _get_config_shape(data_path, "node_vec_shape")
        edge_w_shape = _get_config_shape(data_path, "edge_w_shape")
        meta_db_path = data_path / META_DB_NAME

        # 1) Neo4j 统计节点/边数量
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (n:MugiNode)
                WITH count(n) AS node_count
                MATCH ()-[r:RELATED]->()
                RETURN node_count, count(r) AS edge_count
                """
            ).single()
            node_count = int(record["node_count"]) if record else 0
            edge_count = int(record["edge_count"]) if record else 0

        # 2) 空闲池碎片程度
        conn = sqlite3.connect(meta_db_path)
        try:
            node_existing, node_last_id, node_ratio = _get_fragmentation_info(
                conn, "node_id_reuse_pool", node_vec_shape[0]
            )
            edge_existing, edge_last_id, edge_ratio = _get_fragmentation_info(
                conn, "edge_id_reuse_pool", edge_w_shape[0]
            )
        finally:
            conn.close()

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "node_fragmentation": node_ratio,
            "edge_fragmentation": edge_ratio,
            "node_existing": node_existing,
            "edge_existing": edge_existing,
            "node_last_id": node_last_id,
            "edge_last_id": edge_last_id,
        }


def _get_system_config(data_dir: Path) -> Dict[str, Any]:
    system_meta_path = data_dir / SYSTEM_META_FILE
    if not system_meta_path.exists():
        raise FileNotFoundError(f"System meta not found: {system_meta_path}")
    with open(system_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("config", {})


def _merge_reuse_pool(conn: sqlite3.Connection, pool_table: str) -> None:
    cursor = conn.cursor()
    rows = cursor.execute(
        f"SELECT start, length FROM {pool_table} ORDER BY start ASC"
    ).fetchall()
    if not rows:
        return

    merged = []
    cur_start, cur_len = rows[0]
    for start, length in rows[1:]:
        if start == cur_start + cur_len:
            cur_len += length
        else:
            merged.append((cur_start, cur_len))
            cur_start, cur_len = start, length
    merged.append((cur_start, cur_len))

    cursor.execute(f"DELETE FROM {pool_table}")
    cursor.executemany(
        f"INSERT INTO {pool_table} (start, length) VALUES (?, ?)",
        merged,
    )


def _get_fragmentation_info(
    conn: sqlite3.Connection,
    pool_table: str,
    max_count: int,
) -> tuple[int, int, float]:
    """
    根据空闲池计算：已存储数量、最后一个非空闲 id、碎片程度
    碎片程度 = 已存储数量 / (最后一个非空闲 id + 1)
    """
    cursor = conn.cursor()
    rows = cursor.execute(
        f"SELECT start, length FROM {pool_table} ORDER BY start ASC"
    ).fetchall()

    total_free = sum(int(length) for _, length in rows)
    existing_count = max(0, int(max_count) - total_free)

    if existing_count == 0:
        last_existing_id = -1
        ratio = 0.0
        return existing_count, last_existing_id, ratio

    last_existing_id = max_count - 1
    if rows:
        last_start, last_len = rows[-1]
        last_end = int(last_start) + int(last_len) - 1
        if last_end == max_count - 1:
            last_existing_id = int(last_start) - 1

    if last_existing_id < 0:
        ratio = 0.0
    else:
        ratio = existing_count / float(last_existing_id + 1)

    return existing_count, last_existing_id, ratio


def _allocate_id(conn: sqlite3.Connection, pool_table: str) -> int:
    cursor = conn.cursor()
    row = cursor.execute(
        f"SELECT start, length FROM {pool_table} ORDER BY start ASC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError(f"No available id in {pool_table}")

    start, length = row
    if length <= 1:
        cursor.execute(f"DELETE FROM {pool_table} WHERE start = ?", (start,))
    else:
        cursor.execute(
            f"UPDATE {pool_table} SET start = ?, length = ? WHERE start = ?",
            (start + 1, length - 1, start),
        )
    _merge_reuse_pool(conn, pool_table)
    return int(start)


def upload_node(node: MugiNode, handler: Any, data_dir: str = GRAPH_DATA_DIR) -> int:
    data_path = Path(data_dir)
    config = _get_system_config(data_path)
    node_vec_shape = tuple(config.get("node_vec_shape", []))
    if len(node_vec_shape) != 2 or node_vec_shape[1] != 768:
        raise ValueError("Invalid node vector shape in system_meta.json")

    meta_db_path = data_path / META_DB_NAME
    node_vec_path = data_path / NODE_VECTORS_FILE

    conn = sqlite3.connect(meta_db_path)
    try:
        conn.execute("BEGIN")
        node_id = _allocate_id(conn, "node_id_reuse_pool")

        # 1) 向量层写入
        node_vec = np.memmap(
            node_vec_path,
            dtype="float32",
            mode="r+",
            shape=node_vec_shape,
        )
        node_vec[node_id] = np.array(node.vector, dtype="float32")
        del node_vec

        # 2) 元数据层写入
        node_dict = node.dict()
        node_dict["id"] = node_id
        conn.execute(
            """
            INSERT INTO node_meta (
                id, category, display_name, infer_count, last_active, node_rank, full_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                node.category,
                node.display_name,
                node.infer_count,
                node.last_active,
                node.node_rank,
                json.dumps(node_dict, ensure_ascii=False),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # 3) 拓扑层写入
    try:
        with handler.driver.session() as session:
            session.run(
                """
                MERGE (n:MugiNode {
                    id: $id,
                    category: $category,
                    display_name: $display_name,
                    infer_count: $infer_count,
                    last_active: $last_active,
                    node_rank: $node_rank
                })
                """,
                {
                    "id": node_id,
                    "category": node.category,
                    "display_name": node.display_name,
                    "infer_count": node.infer_count,
                    "last_active": node.last_active,
                    "node_rank": node.node_rank,
                },
            )
    except Exception as e:
        print(f"Failed to create node in Neo4j: {e}")
    return node_id


def upload_edge(edge: MugiEdge, handler: Any, data_dir: str = GRAPH_DATA_DIR) -> int:
    data_path = Path(data_dir)
    config = _get_system_config(data_path)
    edge_w_shape = tuple(config.get("edge_w_shape", []))
    if len(edge_w_shape) != 2:
        raise ValueError("Invalid edge vector shape in system_meta.json")

    meta_db_path = data_path / META_DB_NAME
    edge_w_path = data_path / EDGE_WEIGHTS_FILE

    if len(edge.w_vector) != edge_w_shape[1]:
        raise ValueError("Edge w_vector length mismatch with system_meta.json")

    conn = sqlite3.connect(meta_db_path)
    try:
        conn.execute("BEGIN")
        edge_id = _allocate_id(conn, "edge_id_reuse_pool")

        # 1) 向量层写入
        edge_w = np.memmap(
            edge_w_path,
            dtype="float32",
            mode="r+",
            shape=edge_w_shape,
        )
        edge_w[edge_id] = np.array(edge.w_vector, dtype="float32")
        del edge_w

        # 2) 元数据层写入
        edge_dict = edge.dict()
        edge_dict["id"] = edge_id
        conn.execute(
            """
            INSERT INTO edge_meta (
                id, source_id, target_id, r_intrinsic, b_bias, psi_value, entropy, variance,
                infer_count, description, last_labeled_time, version, status, full_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                edge.source_id,
                edge.target_id,
                edge.r_intrinsic,
                edge.b_bias,
                edge.psi_value,
                edge.entropy,
                edge.variance,
                edge.infer_count,
                edge.description,
                edge.last_labeled_time,
                edge.version,
                edge.status,
                json.dumps(edge_dict, ensure_ascii=False),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # 3) 拓扑层写入
    try:
        with handler.driver.session() as session:
            session.run(
                """
                MATCH (s:MugiNode {id: $source_id}), (t:MugiNode {id: $target_id})
                MERGE (s)-[r:RELATED {
                    id: $id,
                    source_id: $source_id,
                    target_id: $target_id,
                    r_intrinsic: $r_intrinsic,
                    b_bias: $b_bias,
                    psi_value: $psi_value,
                    entropy: $entropy,
                    variance: $variance,
                    infer_count: $infer_count,
                    description: $description,
                    last_labeled_time: $last_labeled_time,
                    version: $version,
                    status: $status
                }]->(t)
                """,
                {
                    "id": edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "r_intrinsic": edge.r_intrinsic,
                    "b_bias": edge.b_bias,
                    "psi_value": edge.psi_value,
                    "entropy": edge.entropy,
                    "variance": edge.variance,
                    "infer_count": edge.infer_count,
                    "description": edge.description,
                    "last_labeled_time": edge.last_labeled_time,
                    "version": edge.version,
                    "status": edge.status,
                },
            )
    except Exception as e:
        print(f"Failed to create edge in Neo4j: {e}")

    return edge_id

def get_node_by_id(node_id: int, handler: Any) -> Dict[str, Any]:
    """通过 ID 查询 Neo4j 中的节点"""
    try:
        with handler.driver.session() as session:
            result = session.run(
                "MATCH (n:MugiNode {id: $id}) RETURN n", 
                {"id": node_id}
            ).single()
            return dict(result["n"]) if result else None
    except Exception as e:
        print(f"Failed to get node from Neo4j: {e}")
        return None

def get_edge_by_id(edge_id: int, handler: Any) -> Dict[str, Any]:
    """通过 ID 查询 Neo4j 中的边（关系）"""
    try:
        with handler.driver.session() as session:
            result = session.run(
                "MATCH ()-[r:RELATED {id: $id}]->() RETURN r", 
                {"id": edge_id}
            ).single()
            return dict(result["r"]) if result else None
    except Exception as e:
        print(f"Failed to get edge from Neo4j: {e}")
        return None


def get_node_meta_by_id(node_id: int, data_dir: str = GRAPH_DATA_DIR) -> Optional[Dict[str, Any]]:
    """
    根据节点 id 查询元数据层，返回字典。
    """
    data_path = Path(data_dir)
    meta_db_path = data_path / META_DB_NAME

    conn = sqlite3.connect(meta_db_path)
    try:
        row = conn.execute(
            "SELECT id, category, display_name, infer_count, last_active, node_rank, full_json "
            "FROM node_meta WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return None

        node_dict: Dict[str, Any] = {}
        if row[6]:
            try:
                node_dict = json.loads(row[6])
            except Exception:
                node_dict = {}

        node_dict["id"] = row[0]
        node_dict["category"] = row[1]
        node_dict["display_name"] = row[2]
        node_dict["infer_count"] = row[3]
        node_dict["last_active"] = row[4]
        node_dict["node_rank"] = row[5]
        return node_dict
    finally:
        conn.close()


def get_node_vector_by_id(node_id: int, data_dir: str = GRAPH_DATA_DIR) -> list[float]:
    """
    根据节点 id 查询向量层，返回节点向量。
    """
    data_path = Path(data_dir)
    node_vec_shape = _get_config_shape(data_path, "node_vec_shape")
    node_vec_path = data_path / NODE_VECTORS_FILE

    node_vec = np.memmap(
        node_vec_path,
        dtype="float32",
        mode="r",
        shape=node_vec_shape,
    )
    vector = node_vec[node_id].tolist()
    del node_vec
    return vector


def get_edge_meta_by_id(edge_id: int, data_dir: str = GRAPH_DATA_DIR) -> Optional[Dict[str, Any]]:
    """
    根据边 id 查询元数据层，返回字典。
    """
    data_path = Path(data_dir)
    meta_db_path = data_path / META_DB_NAME

    conn = sqlite3.connect(meta_db_path)
    try:
        row = conn.execute(
            "SELECT id, source_id, target_id, r_intrinsic, b_bias, psi_value, entropy, variance, "
            "infer_count, description, last_labeled_time, version, status, full_json "
            "FROM edge_meta WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if not row:
            return None

        edge_dict: Dict[str, Any] = {}
        if row[13]:
            try:
                edge_dict = json.loads(row[13])
            except Exception:
                edge_dict = {}

        edge_dict["id"] = row[0]
        edge_dict["source_id"] = row[1]
        edge_dict["target_id"] = row[2]
        edge_dict["r_intrinsic"] = row[3]
        edge_dict["b_bias"] = row[4]
        edge_dict["psi_value"] = row[5]
        edge_dict["entropy"] = row[6]
        edge_dict["variance"] = row[7]
        edge_dict["infer_count"] = row[8]
        edge_dict["description"] = row[9]
        edge_dict["last_labeled_time"] = row[10]
        edge_dict["version"] = row[11]
        edge_dict["status"] = row[12]
        return edge_dict
    finally:
        conn.close()


def get_edge_vector_by_id(edge_id: int, data_dir: str = GRAPH_DATA_DIR) -> list[float]:
    """
    根据边 id 查询向量层，返回 w_vector。
    """
    data_path = Path(data_dir)
    edge_w_shape = _get_config_shape(data_path, "edge_w_shape")
    edge_w_path = data_path / EDGE_WEIGHTS_FILE

    edge_w = np.memmap(
        edge_w_path,
        dtype="float32",
        mode="r",
        shape=edge_w_shape,
    )
    w_vector = edge_w[edge_id].tolist()
    del edge_w
    return w_vector


def get_outgoing_edges_by_source_id(node_id: int, handler: Any) -> Dict[int, int]:
    """
    通过拓扑层 Cypher 查询以该节点为 source_id 的边，返回 {edge_id: target_id}。
    """
    try:
        with handler.driver.session() as session:
            result = session.run(
                """
                MATCH (s:MugiNode {id: $id})-[r:RELATED]->(t:MugiNode)
                RETURN r.id AS edge_id, t.id AS target_id
                ORDER BY r.id
                """,
                {"id": node_id},
            )
            return {int(record["edge_id"]): int(record["target_id"]) for record in result}
    except Exception as e:
        print(f"Failed to query outgoing edges from Neo4j: {e}")
        return {}


def get_incoming_edges_by_target_id(node_id: int, handler: Any) -> Dict[int, int]:
    """
    通过拓扑层 Cypher 查询以该节点为 target_id 的边，返回 {edge_id: source_id}。
    """
    try:
        with handler.driver.session() as session:
            result = session.run(
                """
                MATCH (s:MugiNode)-[r:RELATED]->(t:MugiNode {id: $id})
                RETURN r.id AS edge_id, s.id AS source_id
                ORDER BY r.id
                """,
                {"id": node_id},
            )
            return {int(record["edge_id"]): int(record["source_id"]) for record in result}
    except Exception as e:
        print(f"Failed to query incoming edges from Neo4j: {e}")
        return {}


def update_node_meta_by_id(
    node_id: int,
    updates: Dict[str, Any],
    handler: Any,
    data_dir: str = GRAPH_DATA_DIR,
) -> None:
    """
    根据节点 id 修改元数据层与拓扑层（不可修改 id）。
    """
    if not updates:
        return

    forbidden = {"id"}
    allowed = {"category", "infer_count", "display_name", "last_active", "node_rank"}
    invalid_fields = set(updates.keys()) - allowed
    if invalid_fields:
        raise ValueError(f"Invalid node fields: {sorted(invalid_fields)}")
    if forbidden & set(updates.keys()):
        raise ValueError("id cannot be updated")

    data_path = Path(data_dir)
    meta_db_path = data_path / META_DB_NAME

    conn = sqlite3.connect(meta_db_path)
    try:
        row = conn.execute(
            "SELECT full_json FROM node_meta WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Node id not found: {node_id}")

        current_json: Dict[str, Any] = {}
        if row[0]:
            try:
                current_json = json.loads(row[0])
            except Exception:
                current_json = {}

        current_json.setdefault("id", node_id)
        for key in allowed:
            if key in updates:
                current_json[key] = updates[key]

        set_fields = list(updates.keys())
        set_clause = ", ".join([f"{k} = ?" for k in set_fields])
        params = [updates[k] for k in set_fields]
        params.append(json.dumps(current_json, ensure_ascii=False))
        params.append(node_id)
        conn.execute(
            f"UPDATE node_meta SET {set_clause}, full_json = ? WHERE id = ?",
            params,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    with handler.driver.session() as session:
        session.run(
            """
            MATCH (n:MugiNode {id: $id})
            SET n += $props
            """,
            {"id": node_id, "props": updates},
        )


def update_node_vector_by_id(
    node_id: int,
    vector: list[float],
    data_dir: str = GRAPH_DATA_DIR,
) -> None:
    """
    根据节点 id 修改向量层向量。
    """
    data_path = Path(data_dir)
    node_vec_shape = _get_config_shape(data_path, "node_vec_shape")
    node_vec_path = data_path / NODE_VECTORS_FILE

    if len(vector) != node_vec_shape[1]:
        raise ValueError("Node vector length mismatch with system_meta.json")

    node_vec = np.memmap(
        node_vec_path,
        dtype="float32",
        mode="r+",
        shape=node_vec_shape,
    )
    node_vec[node_id] = np.array(vector, dtype="float32")
    del node_vec


def update_edge_meta_by_id(
    edge_id: int,
    updates: Dict[str, Any],
    handler: Any,
    data_dir: str = GRAPH_DATA_DIR,
) -> None:
    """
    根据边 id 修改元数据层与拓扑层（不可修改 id/source_id/target_id）。
    """
    if not updates:
        return

    forbidden = {"id", "source_id", "target_id"}
    allowed = {
        "r_intrinsic",
        "b_bias",
        "psi_value",
        "entropy",
        "variance",
        "infer_count",
        "description",
        "last_labeled_time",
        "version",
        "status",
    }
    invalid_fields = set(updates.keys()) - allowed
    if invalid_fields:
        raise ValueError(f"Invalid edge fields: {sorted(invalid_fields)}")
    if forbidden & set(updates.keys()):
        raise ValueError("id/source_id/target_id cannot be updated")

    data_path = Path(data_dir)
    meta_db_path = data_path / META_DB_NAME

    conn = sqlite3.connect(meta_db_path)
    try:
        row = conn.execute(
            "SELECT full_json FROM edge_meta WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Edge id not found: {edge_id}")

        current_json: Dict[str, Any] = {}
        if row[0]:
            try:
                current_json = json.loads(row[0])
            except Exception:
                current_json = {}

        current_json.setdefault("id", edge_id)
        for key in allowed:
            if key in updates:
                current_json[key] = updates[key]

        set_fields = list(updates.keys())
        set_clause = ", ".join([f"{k} = ?" for k in set_fields])
        params = [updates[k] for k in set_fields]
        params.append(json.dumps(current_json, ensure_ascii=False))
        params.append(edge_id)
        conn.execute(
            f"UPDATE edge_meta SET {set_clause}, full_json = ? WHERE id = ?",
            params,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    with handler.driver.session() as session:
        session.run(
            """
            MATCH ()-[r:RELATED {id: $id}]->()
            SET r += $props
            """,
            {"id": edge_id, "props": updates},
        )


def update_edge_vector_by_id(
    edge_id: int,
    w_vector: list[float],
    data_dir: str = GRAPH_DATA_DIR,
) -> None:
    """
    根据边 id 修改向量层 w_vector。
    """
    data_path = Path(data_dir)
    edge_w_shape = _get_config_shape(data_path, "edge_w_shape")
    edge_w_path = data_path / EDGE_WEIGHTS_FILE

    if len(w_vector) != edge_w_shape[1]:
        raise ValueError("Edge w_vector length mismatch with system_meta.json")

    edge_w = np.memmap(
        edge_w_path,
        dtype="float32",
        mode="r+",
        shape=edge_w_shape,
    )
    edge_w[edge_id] = np.array(w_vector, dtype="float32")
    del edge_w

def _get_config_shape(data_dir: Path, key: str) -> tuple:
    config = _get_system_config(data_dir)
    shape = tuple(config.get(key, []))
    if len(shape) != 2:
        raise ValueError(f"Invalid {key} in system_meta.json")
    return shape


def _insert_free_range(conn: sqlite3.Connection, pool_table: str, start: int, length: int) -> None:
    if length <= 0:
        return
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT OR IGNORE INTO {pool_table} (start, length) VALUES (?, ?)",
        (int(start), int(length)),
    )
    _merge_reuse_pool(conn, pool_table)


def _get_existing_ids(conn: sqlite3.Connection, pool_table: str, max_count: int) -> list[int]:
    """
    通过空闲池计算当前存在的 id 列表（按升序）。
    空闲池记录的是“空闲区间”，存在的 id = [0, max_count) - 空闲池
    """
    cursor = conn.cursor()
    rows = cursor.execute(
        f"SELECT start, length FROM {pool_table} ORDER BY start ASC"
    ).fetchall()

    free = []
    for start, length in rows:
        free.append((int(start), int(start + length - 1)))

    existing = []
    cur = 0
    for start, end in free:
        if cur < start:
            existing.extend(range(cur, start))
        cur = end + 1
    if cur < max_count:
        existing.extend(range(cur, max_count))

    return existing


def _yield_batches(items: list[int], batch_size: int) -> Generator[list[int], None, None]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def delete_edge_by_id(edge_id: int, handler: Any, data_dir: str = GRAPH_DATA_DIR) -> None:
    """
    删除边：拓扑层 -> 元数据层 -> 向量层，并更新边空闲池
    """
    data_path = Path(data_dir)
    edge_w_shape = _get_config_shape(data_path, "edge_w_shape")
    max_edges = edge_w_shape[0]

    meta_db_path = data_path / META_DB_NAME
    edge_w_path = data_path / EDGE_WEIGHTS_FILE

    # 1) 拓扑层删除
    try:
        with handler.driver.session() as session:
            session.run(
                "MATCH ()-[r:RELATED {id: $id}]->() DELETE r",
                {"id": edge_id},
            )
    except Exception as e:
        print(f"Failed to delete edge in Neo4j: {e}")

    # 2) 元数据层删除 + 空闲池维护
    conn = sqlite3.connect(meta_db_path)
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM edge_meta WHERE id = ?", (edge_id,))
        _insert_free_range(conn, "edge_id_reuse_pool", edge_id, 1)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # 3) 向量层删除（归零）
    edge_w = np.memmap(
        edge_w_path,
        dtype="float32",
        mode="r+",
        shape=edge_w_shape,
    )
    if 0 <= edge_id < max_edges:
        edge_w[edge_id] = np.zeros(edge_w_shape[1], dtype="float32")
    del edge_w


def delete_node_by_id(node_id: int, handler: Any, data_dir: str = GRAPH_DATA_DIR) -> None:
    """
    删除节点：拓扑层 -> 元数据层 -> 向量层
    同时删除关联边，并维护边空闲池
    """
    data_path = Path(data_dir)
    node_vec_shape = _get_config_shape(data_path, "node_vec_shape")
    edge_w_shape = _get_config_shape(data_path, "edge_w_shape")
    max_nodes = node_vec_shape[0]
    max_edges = edge_w_shape[0]

    meta_db_path = data_path / META_DB_NAME
    node_vec_path = data_path / NODE_VECTORS_FILE
    edge_w_path = data_path / EDGE_WEIGHTS_FILE

    # 1) 拓扑层删除（先收集关联边 id，再删除边和节点）
    try:
        with handler.driver.session() as session:
            rels = session.run(
                """
                MATCH (n:MugiNode {id: $id})-[r:RELATED]-()
                RETURN r.id AS rid
                """,
                {"id": node_id},
            ).values()
            related_edge_ids = [row[0] for row in rels if row and row[0] is not None]

            session.run(
                """
                MATCH (n:MugiNode {id: $id})-[r:RELATED]-()
                DELETE r
                """,
                {"id": node_id},
            )
            session.run(
                "MATCH (n:MugiNode {id: $id}) DELETE n",
                {"id": node_id},
            )
    except Exception as e:
        print(f"Failed to delete node in Neo4j: {e}")
        related_edge_ids = []

    # 2) 元数据层删除 + 空闲池维护（节点 + 关联边）
    conn = sqlite3.connect(meta_db_path)
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM node_meta WHERE id = ?", (node_id,))
        _insert_free_range(conn, "node_id_reuse_pool", node_id, 1)

        if related_edge_ids:
            conn.executemany(
                "DELETE FROM edge_meta WHERE id = ?",
                [(int(eid),) for eid in related_edge_ids],
            )
            for eid in related_edge_ids:
                _insert_free_range(conn, "edge_id_reuse_pool", int(eid), 1)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # 3) 向量层删除（归零）
    node_vec = np.memmap(
        node_vec_path,
        dtype="float32",
        mode="r+",
        shape=node_vec_shape,
    )
    if 0 <= node_id < max_nodes:
        node_vec[node_id] = np.zeros(node_vec_shape[1], dtype="float32")
    del node_vec

    edge_w = np.memmap(
        edge_w_path,
        dtype="float32",
        mode="r+",
        shape=edge_w_shape,
    )
    for eid in related_edge_ids:
        if 0 <= eid < max_edges:
            edge_w[eid] = np.zeros(edge_w_shape[1], dtype="float32")
    del edge_w


def print_all_nodes_in_order(handler: Any, data_dir: str = GRAPH_DATA_DIR, batch_size: int = 1000) -> None:
    """
    按 id 顺序打印图谱中所有节点（通过空闲池推导已存在 id）。
    使用 Cypher UNWIND 批量查询，降低 I/O 次数。
    """
    data_path = Path(data_dir)
    node_vec_shape = _get_config_shape(data_path, "node_vec_shape")
    max_nodes = node_vec_shape[0]
    meta_db_path = data_path / META_DB_NAME

    conn = sqlite3.connect(meta_db_path)
    try:
        existing_ids = _get_existing_ids(conn, "node_id_reuse_pool", max_nodes)
    finally:
        conn.close()

    try:
        with handler.driver.session() as session:
            for batch in _yield_batches(existing_ids, batch_size):
                if not batch:
                    continue
                result = session.run(
                    """
                    UNWIND $ids AS id
                    MATCH (n:MugiNode {id: id})
                    RETURN n
                    ORDER BY id
                    """,
                    {"ids": batch},
                )
                for record in result:
                    node = dict(record["n"])
                    print(f"Node(id={node.get('id')}, display_name={node.get('display_name')})")
    except Exception as e:
        print(f"Failed to print nodes from Neo4j: {e}")


def print_all_edges_in_order(handler: Any, data_dir: str = GRAPH_DATA_DIR, batch_size: int = 1000) -> None:
    """
    按 id 顺序打印图谱中所有边（通过空闲池推导已存在 id）。
    使用 Cypher UNWIND 批量查询，降低 I/O 次数。
    """
    data_path = Path(data_dir)
    edge_w_shape = _get_config_shape(data_path, "edge_w_shape")
    max_edges = edge_w_shape[0]
    meta_db_path = data_path / META_DB_NAME

    conn = sqlite3.connect(meta_db_path)
    try:
        existing_ids = _get_existing_ids(conn, "edge_id_reuse_pool", max_edges)
    finally:
        conn.close()

    try:
        with handler.driver.session() as session:
            for batch in _yield_batches(existing_ids, batch_size):
                if not batch:
                    continue
                result = session.run(
                    """
                    UNWIND $ids AS id
                    MATCH ()-[r:RELATED {id: id}]->()
                    RETURN r
                    ORDER BY id
                    """,
                    {"ids": batch},
                )
                for record in result:
                    rel = dict(record["r"])
                    print(
                        f"Edge(id={rel.get('id')}, source_id={rel.get('source_id')}, target_id={rel.get('target_id')})"
                    )
    except Exception as e:
        print(f"Failed to print edges from Neo4j: {e}")

if __name__ == "__main__":
    handler = Neo4jHandler()
    handler.close()