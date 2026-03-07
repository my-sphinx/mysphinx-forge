#!/usr/bin/env python3
"""
从多个 Excel 文件读取 query，基于规则分类 + 向量 KMeans 生成 category，并输出 CSV。

功能:
1. 支持多个 Excel 顺序加载（通过脚本内字典配置）
2. 每个 Excel 读取一个指定列
3. 汇总到同一个 CSV，含 query/category 两列
4. 按 `分类规则.md` 的顺序执行规则；结合向量 KMeans 做分类增强
5. query 重复时，仅保留与其 category 语义中心最接近的一条
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# 在这里配置: Excel 文件路径 -> 要读取的列名
EXCEL_COLUMN_MAPPING: Dict[str, str] = {
    "data/log1.xlsx": "field_text"
}


def parse_rule_order(rules_file: Path) -> List[str]:
    """从分类规则文件中提取类别顺序（用于冲突优先级）。"""
    text = rules_file.read_text(encoding="utf-8")
    cats = re.findall(r"->\s*`([^`]+)`", text)
    if not cats:
        raise ValueError(f"无法从规则文件中提取类别: {rules_file}")
    # 去重并保持顺序
    ordered: List[str] = []
    for c in cats:
        if c not in ordered:
            ordered.append(c.strip())
    return ordered


def parse_rule_examples(rules_file: Path, rule_order: Sequence[str]) -> Dict[str, List[str]]:
    """
    从分类规则文件解析每个类别的“示例”文本，作为向量分类原型。
    只读取“示例”行，忽略“错误示例”。
    """
    text = rules_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    category_blocks: Dict[str, List[str]] = {c: [] for c in rule_order}

    current_cat = None
    cat_pattern = re.compile(r"->\s*`([^`]+)`")
    example_line_pattern = re.compile(r"^\s*示例[:：]")

    for raw in lines:
        line = raw.strip()
        m = cat_pattern.search(line)
        if m:
            c = m.group(1).strip()
            current_cat = c if c in category_blocks else None
            continue

        if current_cat is None:
            continue
        if "错误示例" in line:
            continue
        if example_line_pattern.search(line):
            examples = re.findall(r"`([^`]+)`", line)
            for ex in examples:
                ex = ex.strip()
                if ex:
                    category_blocks[current_cat].append(ex)

    # 若某类无示例，至少使用类名作为兜底原型
    for cat in rule_order:
        if not category_blocks[cat]:
            category_blocks[cat] = [cat]
    return category_blocks


def classify_by_rule_vectors(
    queries: Sequence[str],
    rule_order: Sequence[str],
    rule_examples: Dict[str, List[str]],
) -> Tuple[List[str], List[float], TfidfVectorizer, np.ndarray]:
    """
    基于规则示例向量进行分类。
    返回:
    - 每条 query 的规则类别
    - 每条 query 的规则相似度分数
    - 向量器
    - 归一化 query 向量（供后续 KMeans 复用）
    """
    seed_texts: List[str] = []
    seed_cats: List[str] = []
    for cat in rule_order:
        for ex in rule_examples.get(cat, []):
            seed_texts.append(ex)
            seed_cats.append(cat)

    corpus = list(queries) + seed_texts
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    mat = vectorizer.fit_transform(corpus)
    mat_norm = normalize(mat, norm="l2")

    q_count = len(queries)
    q_mat = mat_norm[:q_count]
    s_mat = mat_norm[q_count:]

    # 构建类别中心
    centroids: List[np.ndarray] = []
    for cat in rule_order:
        idxs = [i for i, c in enumerate(seed_cats) if c == cat]
        centroid = np.asarray(s_mat[idxs].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
    centroid_mat = np.vstack(centroids)

    # 相似度矩阵: [n_query, n_category]
    sim_mat = q_mat @ centroid_mat.T
    sim_mat = np.asarray(sim_mat)

    pred_cats: List[str] = []
    pred_scores: List[float] = []
    for sims in sim_mat:
        max_sim = float(np.max(sims))
        cand_idxs = np.where(np.isclose(sims, max_sim, atol=1e-8))[0]
        # 冲突按规则顺序执行（索引越小优先）
        best_idx = int(cand_idxs[0]) if len(cand_idxs) > 0 else int(np.argmax(sims))
        pred_cats.append(rule_order[best_idx])
        pred_scores.append(max_sim)

    return pred_cats, pred_scores, vectorizer, q_mat


def load_excel_queries(mapping: Dict[str, str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for idx, (file_path, col_name) in enumerate(mapping.items(), start=1):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"[{idx}] Excel 文件不存在: {file_path}")

        df = pd.read_excel(path)
        if col_name not in df.columns:
            raise ValueError(f"[{idx}] 列不存在: {col_name}, 文件: {file_path}, 可用列: {df.columns.tolist()}")

        part = pd.DataFrame({"query": df[col_name], "source_file": str(path)})
        rows.append(part)
        print(f"[加载] {path} -> 列 `{col_name}`: {len(part)} 条")

    if not rows:
        raise ValueError("EXCEL_COLUMN_MAPPING 为空，请先在脚本中配置 Excel 文件路径与列名映射")

    merged = pd.concat(rows, ignore_index=True)
    merged["query"] = merged["query"].astype(str).str.strip()
    merged = merged[merged["query"].notna() & (merged["query"] != "")].copy()
    merged.reset_index(drop=True, inplace=True)
    return merged


def vote_cluster_category(
    labels: np.ndarray,
    seed_categories: Sequence[str],
    category_order: Sequence[str],
) -> Dict[int, str]:
    """用规则初始标签给每个 KMeans 簇投票确定类别。"""
    order_idx = {c: i for i, c in enumerate(category_order)}
    cluster_to_category: Dict[int, str] = {}

    for cluster_id in np.unique(labels):
        idxs = np.where(labels == cluster_id)[0]
        votes: Dict[str, int] = {}
        for i in idxs:
            c = seed_categories[i]
            votes[c] = votes.get(c, 0) + 1
        # 票数降序，票数相同按规则顺序优先
        best = sorted(votes.items(), key=lambda x: (-x[1], order_idx.get(x[0], 10**9)))[0][0]
        cluster_to_category[int(cluster_id)] = best
    return cluster_to_category


def select_best_duplicate_rows(
    df: pd.DataFrame,
    vectors_norm,
    categories: Sequence[str],
) -> pd.DataFrame:
    """
    query 去重:
    - 若重复 query 的候选 category 不同，保留与其 category 中心余弦距离最小的那条。
    - 若 category 相同，保留第一条。
    """
    df = df.copy()
    df["category"] = list(categories)

    # 计算每个类别中心向量
    category_centroids: Dict[str, np.ndarray] = {}
    for cat in sorted(set(categories)):
        idxs = np.where(df["category"].values == cat)[0]
        if len(idxs) == 0:
            continue
        centroid = np.asarray(vectors_norm[idxs].mean(axis=0)).ravel()
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        category_centroids[cat] = centroid

    keep_indices: List[int] = []
    for _query, g in df.groupby("query", sort=False):
        if len(g) == 1:
            keep_indices.append(g.index[0])
            continue

        cats = g["category"].tolist()
        if len(set(cats)) == 1:
            keep_indices.append(g.index[0])
            continue

        # 按与其所属类别中心的距离选择最接近者
        best_idx = None
        best_dist = float("inf")
        for row_idx in g.index:
            cat = df.at[row_idx, "category"]
            centroid = category_centroids.get(cat)
            if centroid is None:
                dist = float("inf")
            else:
                vec = vectors_norm[row_idx]
                sim = float(vec @ centroid)
                dist = 1.0 - sim
            if dist < best_dist:
                best_dist = dist
                best_idx = row_idx

        if best_idx is None:
            best_idx = g.index[0]
        keep_indices.append(best_idx)

    deduped = df.loc[keep_indices, ["query", "category", "kmeans_score"]].copy()
    deduped.reset_index(drop=True, inplace=True)
    return deduped


def run_pipeline(
    output_csv: Path,
    rules_file: Path,
    n_clusters: int,
    random_state: int,
) -> None:
    print("=" * 68)
    print("Excel Query 聚类分类脚本")
    print("=" * 68)

    rule_order = parse_rule_order(rules_file)
    rule_examples = parse_rule_examples(rules_file, rule_order)

    df = load_excel_queries(EXCEL_COLUMN_MAPPING)
    print(f"[汇总] 总 query 数: {len(df)}")

    # 规则向量分类（冲突按规则文件顺序）
    rule_cats, rule_scores, _vectorizer, vectors_norm = classify_by_rule_vectors(
        df["query"].tolist(),
        rule_order,
        rule_examples,
    )
    df["rule_category"] = rule_cats
    df["rule_score"] = rule_scores

    k = max(1, min(n_clusters, len(df)))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(vectors_norm)

    # 计算每条样本与其所属簇中心的余弦相似度，作为 kmeans_score
    centers = normalize(kmeans.cluster_centers_, norm="l2")
    scores: List[float] = []
    for i, cluster_id in enumerate(labels):
        sim_arr = np.asarray(vectors_norm[i] @ centers[int(cluster_id)]).ravel()
        sim = float(sim_arr[0]) if sim_arr.size > 0 else 0.0
        scores.append(sim)
    df["kmeans_score"] = scores

    # 用规则标签给簇命名
    cluster_to_cat = vote_cluster_category(labels, df["rule_category"].tolist(), rule_order)
    df["cluster_category"] = [cluster_to_cat[int(x)] for x in labels]

    # 统计每个簇的多数票占比，用于判断簇类别是否可靠
    cluster_majority_ratio: Dict[int, float] = {}
    for cluster_id in np.unique(labels):
        idxs = np.where(labels == cluster_id)[0]
        votes: Dict[str, int] = {}
        for i in idxs:
            c = df["rule_category"].iat[i]
            votes[c] = votes.get(c, 0) + 1
        max_vote = max(votes.values()) if votes else 0
        ratio = max_vote / len(idxs) if len(idxs) > 0 else 0.0
        cluster_majority_ratio[int(cluster_id)] = ratio

    # 最终类别策略:
    # 1) 规则分数高 -> 直接用规则类别
    # 2) 规则分数低 -> 仅在“非单点簇且簇投票稳定”时使用簇类别，否则拒识
    final_categories: List[str] = []
    rule_score_threshold = 0.05
    min_cluster_size_for_fallback = 2
    min_majority_ratio_for_fallback = 0.6

    for i, cluster_id in enumerate(labels):
        rs = float(df["rule_score"].iat[i])
        if rs >= rule_score_threshold:
            final_categories.append(df["rule_category"].iat[i])
            continue

        idxs = np.where(labels == cluster_id)[0]
        c_size = len(idxs)
        c_ratio = cluster_majority_ratio.get(int(cluster_id), 0.0)
        if c_size >= min_cluster_size_for_fallback and c_ratio >= min_majority_ratio_for_fallback:
            final_categories.append(df["cluster_category"].iat[i])
        else:
            final_categories.append("拒识")

    df["category"] = final_categories

    # 按 query 去重，冲突时保留“类别最接近”样本
    result = select_best_duplicate_rows(df, vectors_norm, df["category"].tolist())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[输出] {output_csv} ({len(result)} 条)")
    print("[分布]")
    print(result["category"].value_counts(dropna=False).to_string())


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 Excel -> query/category CSV（规则 + KMeans）")
    parser.add_argument("--output_csv", type=Path, default=Path("query_category_result.csv"), help="输出 CSV 文件路径")
    parser.add_argument("--rules_file", type=Path, default=Path("分类规则.md"), help="分类规则文件路径")
    parser.add_argument("--n_clusters", type=int, default=11, help="KMeans 聚类数量（默认 11）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run_pipeline(
        output_csv=args.output_csv,
        rules_file=args.rules_file,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
