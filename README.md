# data-process

一个逐步演进的数据处理工具仓库。当前已实现 `数据清洗`、`数据去重`，以及显式流水线 `先清洗再去重`。

执行清洗或去重时，终端会通过 `tqdm` 实时显示多阶段进度条；任务完成后会继续输出最终统计信息。`csv` 默认展示 `统计总行数 -> 分块处理 -> 写出结果`，`Excel` 展示 `读取文件 -> 执行处理 -> 写出结果`。清洗阶段完成时，进度条尾部会附带 `总数 / 删除 / 保留` 摘要，以及 `空行 / 符号 / 表情 / 乱码` 的删除分布；去重阶段会展示 `总数 / 重复 / 保留 / 唯一值` 摘要。每次运行还会在输出文件同目录生成统一日志文件 `data-process.log`，写入阶段日志、错误信息和最终统计；同时会为每个结果文件生成对应的 `*.meta.json` 元数据文件。

## 模块划分

- `data_process/file_io.py`：集中处理 `csv` / `Excel` 的读取、分块读取和结果写出。
- `data_process/cleaning.py`：只负责数据清洗规则与清洗统计，不再承担读写细节。
- `data_process/deduplication.py`：负责标准化后的精确去重。
- `data_process/semantic_deduplication.py`：负责基于 embedding + `faiss` 的语义去重。
- `data_process/cli.py`：负责编排命令行参数、阶段流程、日志和元数据写出。

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

## 命令行参数

| 参数 | 是否必填 | 说明 | 支持的值 |
| --- | --- | --- | --- |
| `--action` | 是 | 指定要执行的功能。当前工具通过该参数选择不同处理动作。 | `clean`、`deduplicate`、`clean-deduplicate` |
| `--input-file` | 是 | 指定输入文件路径。程序会根据文件扩展名自动识别读取方式。 | 支持 `.csv`、`.xls`、`.xlsx`、`.xlsm` |
| `-o`, `--output` | 否 | 指定输出文件路径。未提供时，`clean` 默认生成 `*_cleaned` 文件，`deduplicate` 默认生成 `*_deduplicated` 文件。 | 任意合法输出路径，例如 `result.csv`、`result.xlsx` |
| `--chunk-size` | 否 | 指定 `csv` 分块流式处理时每块读取的行数。仅对 `csv` 生效，`Excel` 会忽略该参数。 | 大于 `0` 的整数，默认 `50000` |
| `--target-column` | 否 | 指定执行清洗或去重判断的目标列名。程序只根据这一列内容决定是否删除整行，其它列会随该行一并保留或删除。未显式传入时，会按候选列顺序自动探测。 | 任意存在于输入文件中的列名；默认按 `text -> 用户问题 -> 客户问题 -> 用户输入` 自动探测 |
| `--dedupe-mode` | 否 | 指定去重模式。`exact` 为标准化后精确匹配，`semantic` 为基于向量相似度的语义去重。仅对 `deduplicate` 生效。 | `exact`、`semantic`，默认 `exact` |
| `--semantic-threshold` | 否 | 指定语义去重阈值。仅对 `--dedupe-mode semantic` 生效。阈值越高，判重越保守。 | `0` 到 `1` 之间的小数，默认 `0.9` |
| `--embedding-model-path` | 否 | 指定语义去重使用的本地 embedding 模型目录。仅对 `--dedupe-mode semantic` 生效。 | 合法本地模型目录路径，默认 `models/m3e-base` |
| `--batch-size` | 否 | 指定语义去重时 embedding 编码批大小。仅对 `--dedupe-mode semantic` 生效。 | 大于 `0` 的整数，默认 `64` |
| `--semantic-index-type` | 否 | 指定语义去重使用的向量索引类型。`flat` 为精确检索，`hnsw` 为近似检索。 | `flat`、`hnsw`，默认 `flat` |
| `--semantic-hnsw-m` | 否 | 指定 `hnsw` 索引的图连接度参数 `M`。仅对 `--semantic-index-type hnsw` 生效。 | 大于 `0` 的整数，默认 `32` |

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
| 指定语义模型路径和阈值 | `uv run python main.py --action deduplicate --input-file data.csv --dedupe-mode semantic --embedding-model-path models/m3e-base --semantic-threshold 0.9` |
| 指定近似语义索引 | `uv run python main.py --action deduplicate --input-file data.csv --dedupe-mode semantic --semantic-index-type hnsw --semantic-hnsw-m 32` |
| 先清洗再去重 | `uv run python main.py --action clean-deduplicate --input-file data.csv` |
| 先清洗再做语义去重 | `uv run python main.py --action clean-deduplicate --input-file data.csv --dedupe-mode semantic` |

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

每次执行 `clean`、`deduplicate` 或 `clean-deduplicate` 时，程序都会在输出文件同目录生成或追加写入 `data-process.log`。日志会记录：

- 本次执行的 action、输入文件和输出文件
- 各处理阶段的开始和完成状态
- 运行错误信息
- 最终统计结果

## 元数据文件

每个输出结果都会生成一个同名元数据文件，例如：

- `input_cleaned.csv` 对应 `input_cleaned.meta.json`
- `input_deduplicated.csv` 对应 `input_deduplicated.meta.json`

元数据文件会记录：

- 生成时间
- action、输入文件、输出文件、日志文件
- 本次运行参数
- 清洗统计
- 去重统计
- 语义去重命中明细文件路径

## 语义去重说明

- 语义去重使用本地 `m3e-base` 模型生成句向量，并通过 `faiss` 做最近邻检索。
- 默认索引类型为 `flat`，即精确内积检索；如果更关注大规模性能，可以切换为 `hnsw` 近似检索。
- 默认模型路径为 `models/m3e-base`。你当前仓库里可以直接通过软链接访问本地模型。
- `semantic` 模式会额外生成 `*_matches.csv`，用于审计每条被删除文本命中了哪条代表文本，以及对应的相似度分数。
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
- 支持显式流水线 `clean-deduplicate`
- 去重标准固定为：去首尾空格、压缩连续空白、大小写归一
- 语义去重可通过 `--semantic-threshold` 调整判重保守程度
- 每次运行会在输出目录写入统一日志文件 `data-process.log`
- 每个输出结果都会生成对应的 `*.meta.json` 元数据文件
- 输出清洗和去重统计信息
