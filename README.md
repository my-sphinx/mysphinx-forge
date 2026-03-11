# data-process

一个逐步演进的数据处理工具仓库。当前已实现首个功能：`数据清洗`。

执行清洗时，终端会通过 `tqdm` 实时显示多阶段进度条；任务完成后会继续输出最终统计信息。`csv` 默认展示 `统计总行数 -> 分块清洗 -> 写出结果`，`Excel` 展示 `读取文件 -> 清洗数据 -> 写出结果`。清洗阶段完成时，进度条尾部会附带 `总数 / 删除 / 保留` 摘要，以及 `空行 / 符号 / 表情 / 乱码` 的删除分布。

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

## 命令行参数

| 参数 | 是否必填 | 说明 | 支持的值 |
| --- | --- | --- | --- |
| `--action` | 是 | 指定要执行的功能。当前工具通过该参数选择不同处理动作。 | `clean` |
| `--input-file` | 是 | 指定输入文件路径。程序会根据文件扩展名自动识别读取方式。 | 支持 `.csv`、`.xls`、`.xlsx`、`.xlsm` |
| `-o`, `--output` | 否 | 指定输出文件路径。未提供时，默认在输入文件同目录下生成 `*_cleaned` 文件。 | 任意合法输出路径，例如 `result.csv`、`result.xlsx` |
| `--chunk-size` | 否 | 指定 `csv` 分块流式清洗时每块读取的行数。仅对 `csv` 生效，`Excel` 会忽略该参数。 | 大于 `0` 的整数，默认 `50000` |
| `--target-column` | 否 | 指定执行清洗判断的目标列名。程序只根据这一列内容判断是否删除整行，其它列会随该行一并保留或删除。 | 任意存在于输入文件中的列名，默认 `text` |

## 参数示例

| 场景 | 命令 |
| --- | --- |
| 使用默认输出文件名执行清洗 | `uv run python main.py --action clean --input-file data.csv` |
| 指定输出文件路径执行清洗 | `uv run python main.py --action clean --input-file data.xlsx --output cleaned.xlsx` |
| 指定 csv 分块大小执行清洗 | `uv run python main.py --action clean --input-file data.csv --chunk-size 20000` |
| 指定清洗目标列执行清洗 | `uv run python main.py --action clean --input-file data.xlsx --target-column 客户问题` |

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

## 当前支持

- 自动根据扩展名读取 `csv`、`xls`、`xlsx`、`xlsm`
- `csv` 在清洗时采用分块流式处理，更适合大文件场景
- 可通过 `--chunk-size` 调整 `csv` 分块大小
- 默认清洗列为 `text`，可通过 `--target-column` 指定例如 `用户问题`、`客户问题`
- 清洗过程中实时显示多阶段终端进度条
- 基于目标列删除空行
- 基于目标列删除全是标点或符号的行
- 基于目标列删除全是表情的行
- 基于目标列删除全是乱码的行
- 输出清洗统计信息
