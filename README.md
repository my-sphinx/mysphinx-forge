# MySphinx Forge

`MySphinx Forge` 是一个逐步演进的数据与模型工作流工具仓库。当前已实现 `数据清洗`、`数据去重`、`语义聚类`，以及显式流水线 `先清洗再去重`。

## 安装与打包

本项目已经配置为标准 Python 包，可以直接构建并发布到 PyPI。

本地安装：

```bash
uv sync
uv pip install -e .
```

构建分发包：

```bash
uv build
```

构建完成后会生成：

- `dist/*.tar.gz`
- `dist/*.whl`

建议先清理旧产物，再重新构建：

```bash
rm -rf build dist *.egg-info
uv build
```

建议先上传到 TestPyPI 验证：

```bash
uvx twine check dist/*
uvx twine upload --repository testpypi dist/*
```

然后用 TestPyPI 安装验证：

```bash
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mysphinx-forge
mysphinx-forge --help
```

确认没问题后再上传到正式 PyPI：

```bash
uvx twine upload dist/*
```

安装后可以直接使用命令行入口：

```bash
mysphinx-forge --help
```

执行清洗、去重或聚类时，终端会通过 `tqdm` 实时显示多阶段进度条；任务完成后会继续输出最终统计信息。`csv` 默认展示 `统计总行数 -> 分块处理 -> 写出结果`，`Excel` 展示 `读取文件 -> 执行处理 -> 写出结果`。清洗阶段完成时，进度条尾部会附带 `总数 / 删除 / 保留` 摘要，以及 `空行 / 符号 / 表情 / 乱码` 的删除分布；去重阶段会展示 `总数 / 重复 / 保留 / 唯一值` 摘要；聚类阶段会展示 `总数 / 簇数 / 噪声 / 入簇` 摘要。每次运行还会在输出文件同目录生成统一日志文件 `mysphinx-forge.log`，写入阶段日志、错误信息和最终统计；同时会为每个结果文件生成对应的 `*.meta.json` 元数据文件。

## 模块划分

- `mysphinx_forge/file_io.py`：集中处理 `csv` / `Excel` 的读取、分块读取和结果写出。
- `mysphinx_forge/cleaning.py`：只负责数据清洗规则与清洗统计，不再承担读写细节。
- `mysphinx_forge/deduplication.py`：负责标准化后的精确去重。
- `mysphinx_forge/semantic_deduplication.py`：负责基于 embedding + `faiss` 的语义去重。
- `mysphinx_forge/clustering.py`：负责基于 embedding 的 `HDBSCAN` / `KMeans` 文本聚类。
- `mysphinx_forge/cluster_reporting.py`：负责聚类分析报表与 HTML 可视化报告生成。
- `mysphinx_forge/cli.py`：负责编排命令行参数、阶段流程、日志和元数据写出。
- `mysphinx_forge/embedding.py`：集中处理本地 embedding 模型加载与输出抑制。

## 使用方式

先同步依赖：

```bash
uv sync
```

运行数据清洗：

```bash
uv run python main.py --action clean --input-file <输入文件路径>
```

指定清洗列：

```bash
uv run python main.py --action clean --input-file <输入文件路径> --target-column 用户问题
```

可选输出路径：

```bash
uv run python main.py --action clean --input-file <输入文件路径> -o <输出文件路径>
```

指定分块大小：

```bash
uv run python main.py --action clean --input-file <输入文件路径> --chunk-size 20000
```

运行精确去重：

```bash
uv run python main.py --action deduplicate --input-file <输入文件路径>
```

运行语义去重：

```bash
uv run python main.py --action deduplicate --input-file <输入文件路径> --dedupe-mode semantic
```

指定语义分类列：

```bash
uv run python main.py --action deduplicate --input-file <输入文件路径> --dedupe-mode semantic --category-column label
```

先清洗再去重：

```bash
uv run python main.py --action clean-deduplicate --input-file <输入文件路径>
```

先清洗再做语义去重：

```bash
uv run python main.py --action clean-deduplicate --input-file <输入文件路径> --dedupe-mode semantic
```

指定去重列：

```bash
uv run python main.py --action deduplicate --input-file <输入文件路径> --target-column 用户问题
```

指定语义模型和阈值：

```bash
uv run python main.py --action deduplicate --input-file <输入文件路径> --dedupe-mode semantic --embedding-model-path models/m3e-base --semantic-threshold 0.9
```

指定近似语义索引：

```bash
uv run python main.py --action deduplicate --input-file <输入文件路径> --dedupe-mode semantic --semantic-index-type hnsw --semantic-hnsw-m 32
```

运行语义聚类：

```bash
uv run python main.py --action cluster --input-file <输入文件路径>
```

指定聚类列：

```bash
uv run python main.py --action cluster --input-file <输入文件路径> --target-column 用户问题
```

使用 `KMeans` 固定簇数聚类：

```bash
uv run python main.py --action cluster --input-file <输入文件路径> --cluster-mode kmeans --num-clusters 12
```

调整 `HDBSCAN` 最小簇大小：

```bash
uv run python main.py --action cluster --input-file <输入文件路径> --cluster-mode hdbscan --min-cluster-size 8
```

使用 LLM 生成聚类摘要标签：

```bash
OPENAI_API_KEY=<你的密钥> uv run python main.py --action cluster --input-file <输入文件路径> --cluster-label-mode llm
```

执行简单模型测试：

```bash
uv run python main.py --action model-test --test-model-path models/your-chat-model
```

通过文件指定 system prompt 执行模型测试：

```bash
uv run python main.py --action model-test --test-model-path models/your-chat-model --system-prompt-file prompts/system.txt
```

基于清洗/去重后的文件批量执行模型测试：

```bash
uv run python main.py --action model-test --input-file data/input_deduplicated.csv --test-model-path models/your-chat-model
```

指定生成参数执行模型测试：

```bash
uv run python main.py --action model-test --test-model-path models/your-chat-model --max-new-tokens 128 --temperature 1.0 --top-p 1.0 --top-k 0 --repetition-penalty 1.05 --no-do-sample
```

当 `model-test` 提供 `--input-file` 时，程序会读取文件中的目标列作为用户输入，生成一个新的 `*_model_tested` 文件，并追加新列 `模型结果` 和 `模型调用时间`。如果原文件中存在 `预期结果` 列，则还会自动追加 `匹配预期` 列，按 `预期结果` 与 `模型结果` 的标准化文本是否一致输出 `True/False`。未提供 `--input-file` 时，仍会使用代码里的固定变量模拟单条“用户输入”，对应变量是 `mysphinx_forge/model_testing.py` 中的 `MODEL_TEST_USER_INPUT`。模型测试还内置了硬编码默认 `SYSTEM_PROMPT`，如果传入 `--system-prompt-file`（兼容别名 `--system-prompt-flie`），则文件内容优先覆盖默认值。这里的测试模型与 embedding 模型是两条独立链路。推荐使用 `--test-model-path`，`--model-path` 仅作为别名保留。当前默认生成策略为稳定模式：`do_sample=False`、`temperature=1.0`、`top_p=1.0`、`top_k=0`、`repetition_penalty=1.05`。

批量模型测试会优先按可见 GPU 数自动分配 worker，每个 worker 绑定单独 device，并在 worker 内按 batch 执行推理；没有 GPU 时会自动退化成单 worker。执行过程中会实时显示整体进度。

## 命令行参数

| 参数 | 是否必填 | 说明 | 支持的值 |
| --- | --- | --- | --- |
| `--action` | 是 | 指定要执行的功能。当前工具通过该参数选择不同处理动作。 | `clean`、`deduplicate`、`clean-deduplicate`、`cluster`、`model-test` |
| `--input-file` | 否 | 指定输入文件路径。`model-test` 可选传入该参数：传入时按文件批量测试，不传时执行单条模型测试；其它 action 会根据文件扩展名自动识别读取方式。 | 支持 `.csv`、`.xls`、`.xlsx`、`.xlsm` |
| `-o`, `--output` | 否 | 指定输出文件路径。未提供时，`clean` 默认生成 `*_cleaned` 文件，`deduplicate` 默认生成 `*_deduplicated` 文件。 | 任意合法输出路径，例如 `result.csv`、`result.xlsx` |
| `--chunk-size` | 否 | 指定 `csv` 分块流式处理时每块读取的行数。仅对 `csv` 生效，`Excel` 会忽略该参数。 | 大于 `0` 的整数，默认 `50000` |
| `--target-column` | 否 | 指定执行清洗或去重判断的目标列名。程序只根据这一列内容决定是否删除整行，其它列会随该行一并保留或删除。未显式传入时，会按候选列顺序自动探测。 | 任意存在于输入文件中的列名；默认按 `text -> 用户问题 -> 客户问题 -> 用户输入` 自动探测 |
| `--dedupe-mode` | 否 | 指定去重模式。`exact` 为标准化后精确匹配，`semantic` 为基于向量相似度的语义去重。仅对 `deduplicate` 生效。 | `exact`、`semantic`，默认 `exact` |
| `--category-column` | 否 | 指定语义去重时用于导出分类相关字段的来源列名。比如传 `label` 时，`*_matches.csv` 会导出 `label` / `matched_label` / `same_label`。输入文件没有该列时，不会导出这三列。 | 任意列名，默认 `category` |
| `--semantic-threshold` | 否 | 指定语义去重阈值。仅对 `--dedupe-mode semantic` 生效。阈值越高，判重越保守。 | `0` 到 `1` 之间的小数，默认 `0.9` |
| `--embedding-model-path` | 否 | 指定语义去重使用的本地 embedding 模型目录。仅对 `--dedupe-mode semantic` 生效。 | 合法本地模型目录路径，默认 `models/m3e-base` |
| `--train-model-path` | 否 | 指定模型训练使用的本地模型路径。当前版本先预留该参数，后续训练功能接入时使用。 | 合法本地模型目录路径 |
| `--batch-size` | 否 | 指定语义去重时 embedding 编码批大小。仅对 `--dedupe-mode semantic` 生效。 | 大于 `0` 的整数，默认 `64` |
| `--test-model-path` | 否 | 指定模型测试使用的本地模型目录。仅对 `--action model-test` 生效。 | 合法本地模型目录路径 |
| `--model-path` | 否 | `--test-model-path` 的别名，便于兼容和简写。 | 合法本地模型目录路径 |
| `--system-prompt-file` | 否 | 指定模型测试使用的 system prompt 文件路径；如果提供，文件内容优先于代码内置默认 system prompt。兼容别名 `--system-prompt-flie`。 | 合法文本文件路径 |
| `--max-new-tokens` | 否 | 模型测试时最大生成 token 数。 | 大于 `0` 的整数，默认 `64` |
| `--do-sample` | 否 | 模型测试时启用采样生成。 | 默认关闭 |
| `--no-do-sample` | 否 | 模型测试时关闭采样，改为确定性生成。 | 默认即关闭 |
| `--temperature` | 否 | 模型测试采样温度。 | 大于 `0` 的数值，默认 `1.0` |
| `--top-p` | 否 | 模型测试 nucleus sampling 的 top_p。 | `0` 到 `1` 之间的小数，默认 `1.0` |
| `--top-k` | 否 | 模型测试采样时的 top_k。 | 大于等于 `0` 的整数，默认 `0` |
| `--repetition-penalty` | 否 | 模型测试重复惩罚系数。 | 大于 `0` 的数值，默认 `1.05` |
| `--model-test-batch-size` | 否 | 批量模型测试时单个 worker 的推理批大小。 | 大于 `0` 的整数，默认 `8` |
| `--model-test-num-workers` | 否 | 批量模型测试 worker 数。 | `auto` 或大于 `0` 的整数，默认 `auto` |
| `--semantic-index-type` | 否 | 指定语义去重使用的向量索引类型。`flat` 为精确检索，`hnsw` 为近似检索。 | `flat`、`hnsw`，默认 `flat` |
| `--semantic-hnsw-m` | 否 | 指定 `hnsw` 索引的图连接度参数 `M`。仅对 `--semantic-index-type hnsw` 生效。 | 大于 `0` 的整数，默认 `32` |
| `--cluster-mode` | 否 | 指定聚类模式。`hdbscan` 为密度聚类，`kmeans` 为固定簇数聚类。仅对 `cluster` 生效。 | `hdbscan`、`kmeans`，默认 `hdbscan` |
| `--min-cluster-size` | 否 | 指定 `HDBSCAN` 的最小簇大小。仅对 `--cluster-mode hdbscan` 生效。 | 大于 `0` 的整数，默认 `5` |
| `--num-clusters` | 否 | 指定 `KMeans` 聚类簇数。仅对 `--cluster-mode kmeans` 生效。 | 大于 `0` 的整数，默认 `8` |
| `--cluster-selection-epsilon` | 否 | 指定 `HDBSCAN` 的 `cluster_selection_epsilon`。值越大，簇边界越宽松。 | 大于等于 `0` 的小数，默认 `0` |
| `--cluster-label-mode` | 否 | 指定聚类标签生成模式。`rule` 为“关键词 + 代表文本”的规则标签，`llm` 为基于簇样本生成的摘要标签。 | `rule`、`llm`，默认 `rule` |
| `--cluster-label-model` | 否 | 指定 LLM 聚类标签使用的模型名。仅对 `--cluster-label-mode llm` 生效。 | 任意兼容模型名，默认 `gpt-4.1-mini` |
| `--cluster-label-api-base` | 否 | 指定 LLM 聚类标签接口基地址。未指定时优先读取 `OPENAI_BASE_URL`，否则默认 `https://api.openai.com/v1`。 | 任意兼容 OpenAI Chat Completions 的基地址 |
| `--cluster-label-sample-size` | 否 | 指定每个簇送给 LLM 生成摘要标签的示例问题数量。 | 大于 `0` 的整数，默认 `8` |

## 参数示例

| 场景 | 命令 |
| --- | --- |
| 使用默认输出文件名执行清洗 | `uv run python main.py --action clean --input-file data.csv` |
| 指定输出文件路径执行清洗 | `uv run python main.py --action clean --input-file data.xlsx --output cleaned.xlsx` |
| 指定 csv 分块大小执行清洗 | `uv run python main.py --action clean --input-file data.csv --chunk-size 20000` |
| 指定清洗目标列执行清洗 | `uv run python main.py --action clean --input-file data.xlsx --target-column 客户问题` |
| 使用默认输出文件名执行去重 | `uv run python main.py --action deduplicate --input-file data.csv` |
| 指定去重目标列执行去重 | `uv run python main.py --action deduplicate --input-file data.xlsx --target-column 用户问题` |
| 使用语义去重 | `uv run python main.py --action deduplicate --input-file data.csv --dedupe-mode semantic` |
| 指定语义分类列 | `uv run python main.py --action deduplicate --input-file data.csv --dedupe-mode semantic --category-column label` |
| 指定语义模型路径和阈值 | `uv run python main.py --action deduplicate --input-file data.csv --dedupe-mode semantic --embedding-model-path models/m3e-base --semantic-threshold 0.9` |
| 指定近似语义索引 | `uv run python main.py --action deduplicate --input-file data.csv --dedupe-mode semantic --semantic-index-type hnsw --semantic-hnsw-m 32` |
| 先清洗再去重 | `uv run python main.py --action clean-deduplicate --input-file data.csv` |
| 先清洗再做语义去重 | `uv run python main.py --action clean-deduplicate --input-file data.csv --dedupe-mode semantic` |
| 使用默认参数执行聚类 | `uv run python main.py --action cluster --input-file data.csv` |
| 使用 `KMeans` 固定簇数聚类 | `uv run python main.py --action cluster --input-file data.csv --cluster-mode kmeans --num-clusters 12` |
| 调整 `HDBSCAN` 最小簇大小 | `uv run python main.py --action cluster --input-file data.csv --cluster-mode hdbscan --min-cluster-size 8` |
| 使用 LLM 生成聚类摘要标签 | `OPENAI_API_KEY=... uv run python main.py --action cluster --input-file data.csv --cluster-label-mode llm` |
| 执行简单模型测试 | `uv run python main.py --action model-test --test-model-path models/your-chat-model` |
| 通过文件指定 system prompt 执行模型测试 | `uv run python main.py --action model-test --test-model-path models/your-chat-model --system-prompt-file prompts/system.txt` |
| 基于去重结果文件批量执行模型测试 | `uv run python main.py --action model-test --input-file data/input_deduplicated.csv --test-model-path models/your-chat-model` |
| 指定 worker 数和 batch 大小执行批量模型测试 | `uv run python main.py --action model-test --input-file data/input_deduplicated.csv --test-model-path models/your-chat-model --model-test-num-workers auto --model-test-batch-size 8` |
| 自定义生成参数执行模型测试 | `uv run python main.py --action model-test --test-model-path models/your-chat-model --max-new-tokens 128 --temperature 1.0 --top-p 1.0 --top-k 0 --repetition-penalty 1.05 --no-do-sample` |

## 输出统计字段说明

程序执行完成后，会在终端输出清洗统计信息。各字段含义如下：

| 字段 | 说明 |
| --- | --- |
| `清洗完成，输出文件` | 清洗后的结果文件实际写入位置。 |
| `清洗前总行数` | 输入文件读取后的总行数，即参与清洗判断的原始数据行数。 |
| `删除空行` | 目标列被判定为空内容并删除的行数。 |
| `删除全符号/标点行` | 目标列内容全部由标点符号或符号字符组成的行数，例如 `!!!`、`###`、`***`。 |
| `删除全表情行` | 目标列内容全部由表情字符组成的行数，例如 `😂🤣`。 |
| `删除全乱码行` | 目标列内容被判定为乱码并删除的行数，例如大量替换字符或明显异常编码字符组成的内容。 |
| `共删除` | 本次清洗中删除的总行数。 |
| `清洗后总行数` | 清洗完成后保留下来的总行数。 |

## 去重统计字段说明

程序执行 `deduplicate` 完成后，会在终端输出去重统计信息。各字段含义如下：

| 字段 | 说明 |
| --- | --- |
| `去重完成，输出文件` | 去重后的结果文件实际写入位置。 |
| `去重模式` | 本次去重使用的模式，可能是 `exact` 或 `semantic`。 |
| `使用目标列` | 本次去重实际使用的列名。未显式传入时，可能是自动探测得到的列。 |
| `语义阈值` | 仅在语义去重时输出。达到该相似度阈值的文本会被判定为重复。 |
| `语义模型路径` | 仅在语义去重时输出。本次加载的本地 embedding 模型目录。 |
| `语义命中明细` | 仅在语义去重时输出。指向 `*_matches.csv`，记录每条被删除文本命中的代表文本及相似度。 |
| `去重前总行数` | 输入文件读取后的总行数，即参与去重判断的原始数据行数。 |
| `删除重复行数` | 基于当前去重模式判定为重复并删除的行数。 |
| `标准化后唯一值数量` | 精确去重时，表示标准化后最终得到的唯一值数量；语义去重时，表示最终保留下来的代表文本数量。 |
| `去重后总行数` | 去重完成后保留下来的总行数。 |

## 日志文件

每次执行 `clean`、`deduplicate`、`clean-deduplicate` 或 `cluster` 时，程序都会在输出文件同目录生成或追加写入 `mysphinx-forge.log`。日志会记录：

- 本次执行的 action、输入文件和输出文件
- 各处理阶段的开始和完成状态
- 运行错误信息
- 最终统计结果

## 元数据文件

每个输出结果都会生成一个同名元数据文件，例如：

- `input_cleaned.csv` 对应 `input_cleaned.meta.json`
- `input_deduplicated.csv` 对应 `input_deduplicated.meta.json`
- `input_clustered.csv` 对应 `input_clustered.meta.json`

元数据文件会记录：

- 生成时间
- action、输入文件、输出文件、日志文件
- 本次运行参数
- 清洗统计
- 去重统计
- 聚类统计
- 语义去重命中明细文件路径
- 聚类汇总文件路径
- 聚类二维投影文件路径
- 聚类分析报表路径
- 聚类 HTML 报告路径

## 聚类说明

- 聚类使用本地 embedding 模型生成句向量，默认复用 `models/m3e-base`。
- 默认聚类模式为 `hdbscan`，适合簇数未知、希望识别噪声点的数据。
- 可通过 `--cluster-mode kmeans --num-clusters N` 切换为固定簇数聚类。
- 聚类结果主文件会新增 `cluster_id`、`is_noise`、`cluster_size`、`cluster_representative_text`、`cluster_top_keywords`、`cluster_label` 字段。
- 聚类还会额外生成 `*_clusters.csv`，用于汇总每个簇的规模、主题标签、关键词、代表文本和示例文本，便于后续分析和可视化。
- 聚类还会生成 `*_projection.csv`，包含 `row_index`、目标文本、`cluster_id`、`is_noise`、二维坐标 `x/y`，可直接用于散点图可视化。
- 聚类还会生成 `*_analysis.csv`，按簇输出 `rank / size / ratio / label / keywords / representative_text`，便于表格分析和二次加工。
- 聚类还会生成 `*_report.html`，内置统计卡片、簇摘要表和二维散点图，并支持按簇筛选、仅看噪声点、点击散点查看原文详情，可直接在浏览器打开查看。
- 当前聚类会一次性读取文件到内存中执行；相比清洗和去重，它更适合已经过预处理的数据集。

## 语义去重说明

- 语义去重使用本地 `m3e-base` 模型生成句向量，并通过 `faiss` 做最近邻检索。
- 默认索引类型为 `flat`，即精确内积检索；如果更关注大规模性能，可以切换为 `hnsw` 近似检索。
- 默认模型路径为 `models/m3e-base`。你当前仓库里可以直接通过软链接访问本地模型。
- `semantic` 模式会额外生成 `*_matches.csv`，用于审计每条被删除文本命中了哪条代表文本，以及对应的相似度分数。
- `*_matches.csv` 中的分类相关列名会跟 `--category-column` 动态联动。默认读取输入里的 `category` 列，并导出 `category` / `matched_category` / `same_category`；如果你的输入使用 `label`，则可通过 `--category-column label` 导出 `label` / `matched_label` / `same_label`。
- 如果输入里不存在指定的分类列，`*_matches.csv` 会只保留基础字段，不输出分类相关列。
- `csv` 路径会按块读取、按批生成 embedding，并将 `*_matches.csv` 命中明细按块追加写盘，避免一次性堆积全量向量和命中明细。
- 即便如此，内存中仍会保留代表文本向量索引；数据越多、代表文本越多，内存占用也会随之增长。

## 流水线说明

- `clean-deduplicate` 是显式流水线，不会去猜输入文件是否已经清洗过。
- `csv` 路径会先流式清洗到临时中间文件，再对清洗结果做流式去重，最后删除中间文件。
- `Excel` 路径会在内存中先清洗、再去重，然后一次写出最终结果。
- 该 action 会同时输出清洗统计和去重统计。

## 当前支持

- 自动根据扩展名读取 `csv`、`xls`、`xlsx`、`xlsm`
- `csv` 在清洗和去重时都采用分块流式处理，更适合大文件场景
- 可通过 `--chunk-size` 调整 `csv` 分块大小
- 默认按 `text -> 用户问题 -> 客户问题 -> 用户输入` 自动探测处理列，也可通过 `--target-column` 显式指定
- 清洗和去重过程中实时显示多阶段终端进度条
- 基于目标列删除空行
- 基于目标列删除全是标点或符号的行
- 基于目标列删除全是表情的行
- 基于目标列删除全是乱码的行
- 基于目标列做标准化后精确去重
- 支持基于本地 `m3e-base` + `faiss` 的语义去重
- 支持通过 `--semantic-index-type` 在精确检索和近似检索之间切换
- 支持基于本地 embedding 的 `HDBSCAN` / `KMeans` 文本聚类
- 支持显式流水线 `clean-deduplicate`
- 去重标准固定为：去首尾空格、压缩连续空白、大小写归一
- 语义去重可通过 `--semantic-threshold` 调整判重保守程度
- 每次运行会在输出目录写入统一日志文件 `mysphinx-forge.log`
- 每个输出结果都会生成对应的 `*.meta.json` 元数据文件
- 输出清洗和去重统计信息
