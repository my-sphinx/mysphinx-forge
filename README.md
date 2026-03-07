# Excel Query 分类脚本

基于 `分类规则.md` 对多个 Excel 中的 query 做向量分类，并输出统一 CSV 结果。

## 功能

1. 多个 Excel 顺序加载（脚本内字典配置）。
2. 每个 Excel 读取一个指定列，统一映射为 `query`。
3. 使用规则示例做向量分类（非纯关键词匹配）。
4. 使用 KMeans 聚类做类别辅助与打分。
5. `query` 重复时去重，仅保留更接近类别中心的记录。
6. 输出 `query,category,kmeans_score`。

## 文件说明

- 入口脚本: `excel_kmeans_clustering.py`
- 分类规则: `分类规则.md`
- 依赖清单: `requirements.txt`
- 默认输出: `query_category_result.csv`

## 环境准备

推荐使用你们已有的 `pingan` 环境。

```bash
/opt/anaconda3/envs/pingan/bin/python -m pip install -r requirements.txt
```

## 配置 Excel 输入

在脚本中修改 `EXCEL_COLUMN_MAPPING`：

```python
EXCEL_COLUMN_MAPPING = {
    "data/log1.xlsx": "field_text",
    # "data/log2.xlsx": "query_col",
}
```

说明：
- key 是 Excel 文件路径。
- value 是该 Excel 中需要读取的列名。
- 字典顺序即加载顺序。

## 执行

```bash
/opt/anaconda3/envs/pingan/bin/python excel_kmeans_clustering.py
```

可选参数：

```bash
/opt/anaconda3/envs/pingan/bin/python excel_kmeans_clustering.py \
  --output_csv query_category_result.csv \
  --rules_file 分类规则.md \
  --n_clusters 11 \
  --random_state 42
```

## 输出说明

输出 CSV 列：
- `query`: 归一化后的问题文本。
- `category`: 最终分类类别。
- `kmeans_score`: 当前样本与所属 KMeans 簇中心的余弦相似度（数值越高代表越贴近该簇中心）。

## 分类逻辑

1. 从 `分类规则.md` 读取类别顺序与每类“示例”文本。
2. 将 query 和规则示例共同向量化（TF-IDF 字符 n-gram）。
3. 用 query 向量与类别原型中心做余弦相似度，得到 `rule_category`。
4. 对 query 做 KMeans 聚类，并通过簇内规则标签投票得到 `cluster_category`。
5. 默认使用规则向量分类；当规则分数过低时使用簇类别兜底。
6. 对重复 query 做去重，冲突时保留更接近类别中心的记录。

## 常见问题

- 报错 `Excel 文件不存在`：检查 `EXCEL_COLUMN_MAPPING` 路径是否正确。
- 报错 `列不存在`：检查映射中的列名是否与 Excel 表头完全一致。
- 分类不符合预期：优先检查 `分类规则.md` 示例是否覆盖了该表达方式。
