# matchzhengdb

本仓库用于整理“福州国潮香氛产品问卷与综合分析”项目的代码、报告文本和方法说明。项目围绕问卷数据展开，形成六个分析分模块，并在此基础上生成综合报告、扩展报告和论文稿。

## 项目定位

- 研究对象：福州国潮香氛产品消费者问卷与扩展语料
- 分析目标：完成认知、购买行为、场景适配、结构方程模型、痛点主题和消费者细分的全链路分析
- 当前主报告口径：第四部分 SEM 已切换为“购买准备度合并模型（备选A）”

## 主要入口

推荐从以下脚本运行：

```bash
python code/run_all_parts.py
```

如果只想更新总报告，可分别运行：

```bash
python code/generate_overall_report.py
python code/generate_extended_analysis_report.py
```

说明：

- `main.py` 目前不是项目真实入口，只是 IDE 初始化脚本。
- 六个分模块的主脚本都位于 `code/第一部分` 到 `code/第六部分` 目录中。

## 目录结构

```text
.
├─ code/
│  ├─ run_all_parts.py
│  ├─ generate_overall_report.py
│  ├─ generate_extended_analysis_report.py
│  ├─ shared_analysis_utils.py
│  ├─ 第一部分/
│  ├─ 第二部分/
│  ├─ 第三部分/
│  ├─ 第四部分/
│  ├─ 第五部分/
│  └─ 第六部分/
├─ data/              # 本地数据目录，仓库中已排除
├─ output/            # 综合报告与导出文稿
└─ referpaper/        # 参考资料目录，PDF 已排除
```

## 六个分析流程

1. 第一部分：认知现状与认知转化漏斗
2. 第二部分：购买行为、价格敏感度与消费者聚类
3. 第三部分：使用场景生态位与资源匹配
4. 第四部分：购买意愿结构方程模型与稳健性分析
5. 第五部分：显性痛点、开放题与外部语料主题融合
6. 第六部分：消费者细分与人口统计映射

## 第四部分 SEM 说明

第四部分是项目的方法核心，当前保留了两类能力：

- 基线 SEM 流程：测量模型、题项审计、路径回归、bootstrap、交叉验证、分组比较
- 主报告优选模型：将高相关构念合并后的“购买准备度”备选模型

当前主报告采用的核心思路是：

- 冻结原始模型作为 baseline
- 构建合并高相关构念的候选模型
- 自动比较判别效度、CR/AVE 和交叉验证 RMSE
- 如果判别效度改善明显、解释力下降很小，则将该模型作为主报告口径

相关代码和输出主要在：

- `code/第四部分/第四部分分析.py`
- `code/第四部分/output/分析摘要.md`
- `code/第四部分/output/SEM_计算过程与审计说明.md`

## 运行环境

仓库内暂未提供 `requirements.txt` 或 `environment.yml`。当前代码主要依赖：

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `networkx`

建议在已有的 Anaconda/Python 3.9+ 环境中运行。

## 数据与大文件说明

为便于代码仓库分享，当前 Git 已排除以下内容：

- `data/` 目录
- `*.csv`
- `*.pdf`
- 常见图片文件：`*.png`、`*.jpg`、`*.jpeg`、`*.gif`、`*.bmp`、`*.webp`、`*.tif`、`*.tiff`、`*.svg`
- `.idea/`、`__pycache__/`、`*.pyc`

这意味着：

- 仓库保留了代码、Markdown 说明、Word 报告和配置文件
- 如果要在本地完整复现分析，需要自行补回原始数据和被排除的图表资产
- 默认数据路径仍是 `data/endalldata1.csv`

## 当前已导出的结果

根目录 `output/` 下保留了当前版本的主要文稿，例如：

- 综合分析报告
- 图文扩展分析报告
- 投稿版论文稿

第四部分的 SEM 说明、候选模型比较和分析摘要位于 `code/第四部分/output/`。

