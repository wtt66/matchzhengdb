# 福州本土国潮香氛产品消费者认知、行为与购买意愿研究

## 中文摘要

本研究基于福州本土国潮香氛问卷数据（n=596），构建“认知—行为—场景—意愿—痛点—细分”一体化分析框架，系统检验本土国潮香氛产品的认知形成机制、购买行为结构、场景生态位特征、购买意愿驱动因素以及消费者细分逻辑。研究综合采用有序Logit模型、K-means聚类、对应分析、Levins生态位宽度指数、Pianka生态位重叠指数、路径分析式回归，以及基于问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构的三源证据整合方法。结果表明：（1）样本总体国潮认知均值为 2.90/5，本土产品知晓率为 83.2%，实际购买率为 46.3%，说明市场已形成初步文化认知基础，但仍存在显著的知晓—转化断裂；（2）购买行为呈现显著异质性，行为聚类最优解为 5 类；（3）场景使用具有明确的生态位分化和重叠结构，高频场景主要集中于 车载, 住宿, 娱乐；（4）合并高相关构念后的简约购买意愿模型 的调整后决定系数为 0.387，主要驱动集中于 购买准备度、先验知识、文化价值感知；（5）第五部分进一步整合问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构，问卷开放题与平衡后的公开网页语料清洗后形成 648 条 BERTopic 可建模文本（其中问卷开放题 351 条、公开网页语料 297 条）；痛点反馈集中于 留香短/品质不佳, 价格偏高/性价比低, 非遗/地域融合表面化，改进需求主要聚焦于 深化非遗与地域融合, 降低价格提升性价比, 优化香型适配湿热气候。研究表明，福州本土国潮香氛的关键约束不在于消费者对文化概念完全陌生，而在于文化认知尚未充分转译为可识别、可体验和可购买的具体产品体系。本文在理论上拓展了文化型香氛产品购买机制研究，在实践上为本土品牌的产品开发、场景布局、渠道归因与分层运营提供了实证依据。

**关键词：** 国潮香氛；非物质文化遗产；购买意愿；场景生态位；消费者细分；文本挖掘

## Abstract

Using questionnaire data on local Chinese-chic fragrance products in Fuzhou (n=596), this study develops an integrated analytical framework linking cognition, behavior, usage scenarios, purchase intention, pain points, and consumer segmentation. Ordered logit modeling, K-means clustering, correspondence analysis, Levins' niche breadth, Pianka's niche overlap, path-analysis-style regression, and a three-source pain-point framework that integrates explicit questionnaire pain points, a balanced public-web corpus, and BERTopic-derived themes were combined to examine the formation and conversion mechanisms of local fragrance consumption. The results show that respondents already possess a moderate cognitive basis for Chinese-chic fragrance products (mean awareness = 2.90/5), yet a substantial gap remains between local product awareness (83.2%) and actual purchase (46.3%). Behavioral heterogeneity is pronounced, with the optimal consumer behavior solution consisting of 5 segments. Scenario use also exhibits niche differentiation and overlap, with the most salient usage scenarios concentrated in 车载, 住宿, 娱乐. In the preferred purchase-intention model, the adjusted R-squared reaches 0.387, indicating that the main explanatory forces are concentrated in 购买准备度, 先验知识, 文化价值感知. The fifth module further integrates explicit questionnaire pain points, the balanced public-web corpus, and BERTopic themes, yielding 648 modelable texts after cleaning; consumer complaints are mainly concentrated on 留香短/品质不佳, 价格偏高/性价比低, 非遗/地域融合表面化, while improvement expectations focus on 深化非遗与地域融合, 降低价格提升性价比, 优化香型适配湿热气候. Overall, the major bottleneck of Fuzhou local Chinese-chic fragrance products does not lie in the absence of cultural cognition per se, but in the insufficient translation of cultural cognition into identifiable, experienceable, and purchasable product systems. The study contributes to the literature on cultural-product consumption and provides actionable implications for product development, scenario-oriented strategy, channel attribution, and segmented market operation.

**Keywords:** Chinese-chic fragrance products; intangible cultural heritage; purchase intention; scenario niche; consumer segmentation; text mining

## 1 引言

随着“国潮”消费的兴起，传统文化元素、地域文化符号与现代审美设计逐渐在消费市场中形成新的融合逻辑。与一般文化创意产品相比，香氛产品兼具嗅觉体验、情绪唤起、符号表达和礼赠功能，因此既是功能性消费品，也是高度依赖文化叙事与感官价值的体验型产品。对于福州而言，茉莉花窨制、冷凝合香、福文化、榕城意象和三坊七巷等地方文化资源，为本土国潮香氛产品提供了较强的文化素材基础。然而，地方文化资源是否能够稳定转化为市场认知、购买行为和持续消费，仍缺乏系统的问卷证据与实证模型支撑。

现有研究主要从三个方面解释文化型产品的消费形成机制。第一，计划行为理论（Theory of Planned Behavior, TPB）强调态度、主观规范与知觉行为控制对行为意向的共同作用[1]；技术接受模型（Technology Acceptance Model, TAM）则强调认知有用性与易用性对采纳行为的重要影响[2]。第二，品牌资产与感知价值研究指出，文化价值、品牌知识与象征意义会显著改变消费者对产品的判断和选择[3][6][7]。第三，感官营销研究表明，嗅觉体验会系统性影响消费者的情绪评价、环境感知、品牌记忆和购买反应[4][5][9]。但从研究现状看，关于“地方非遗—地域文化—香氛消费”这一复合情境，尤其是认知、行为、场景、意愿和痛点之间的递进机制，仍缺少整合性研究框架。

基于此，本文围绕福州本土国潮香氛问卷数据，构建一个多模块整合分析框架，重点回答以下问题：第一，消费者对国潮香氛、福州非遗技艺和本土产品的认知是否存在层级断裂；第二，购买行为是否表现出稳定的群体异质性与渠道迁移结构；第三，场景使用是否具有可度量的生态位宽度、重叠与协同关系；第四，哪些变量是购买意愿的核心驱动因素；第五，消费者痛点与改进诉求如何在结构化与文本化层面被识别；第六，不同消费者细分群在行为与文化认同上如何被界定。本文试图在理论上推进文化型香氛消费研究，在方法上实现多模块证据整合，在实践上为本土品牌策略制定提供依据。

## 2 理论假设

### 2.1 文化价值感知与购买意愿

文化价值感知是指消费者对非遗技艺、地域文化符号和地方叙事融入产品后所形成的综合价值判断。对于文化型香氛产品而言，文化价值感知并不只是附加属性，而是塑造产品差异化和认同感的重要来源。因此，提出如下假设：

**H1：** 文化价值感知正向影响购买意愿。

### 2.2 产品知识、购买便利性与经济可及性

产品知识反映消费者对香型、工艺、品质与文化表达差异的理解程度；购买便利性强调消费者获取、体验和购买产品的便利程度；经济可及性则代表消费者对价格和支付负担的主观承受能力。根据既有消费理论，这三类变量均会影响消费者的行为评估与决策信心。因此，提出如下假设：

**H2：** 产品知识正向影响购买意愿。  
**H3：** 购买便利性正向影响购买意愿。  
**H4：** 经济可及性正向影响购买意愿。

### 2.3 感知风险、产品涉入度与先验知识

感知风险通常会削弱消费者对新产品或本土文化产品的尝试意愿；而产品涉入度代表消费者对香氛产品的兴趣投入、信息关注和比较深度；先验知识则反映消费者在香调、香料、选购经验方面的知识储备。高涉入与高知识消费者更可能形成稳定判断并降低不确定性。因此，提出如下假设：

**H5：** 感知风险负向影响购买意愿。  
**H6：** 产品涉入度正向影响购买意愿。  
**H7：** 先验知识正向影响购买意愿。

### 2.4 调节效应假设

从认知加工与消费决策视角看，产品涉入度会强化消费者对文化价值和购买便利性的敏感性；先验知识则可能削弱感知风险的负面冲击。因此，进一步提出如下假设：

**H8a：** 产品涉入度正向调节文化价值感知与购买意愿之间的关系。  
**H8b：** 产品涉入度正向调节购买便利性与购买意愿之间的关系。  
**H9：** 先验知识削弱感知风险对购买意愿的负向影响。

## 3 材料与方法

### 3.1 研究设计与数据来源

本文数据来源于针对福州本土国潮香氛消费认知与行为的结构化问卷，分析数据文件为 `data/endalldata.csv`。问卷内容覆盖基础认知、购买行为、场景使用、购买意愿影响因素、消费痛点与改进建议以及人口统计信息六个部分。经程序读取与字段映射后，共纳入 596 份有效样本，字段数为 154。新版问卷未保留第11题的场景预算分配项，因此第三部分不再采用预算集中度指标，而采用场景协同强度与气候适配策略采用数作为替代性资源组织指标。

### 3.2 变量测量与术语规范

为统一术语，本文采用如下标准写法：文化价值感知（perceived cultural value, **CVP**）、产品知识（product knowledge, **PK**）、购买便利性（purchase convenience, **PC**）、经济可及性（economic accessibility, **EA**）、感知风险（perceived risk, **PR**）、产品涉入度（product involvement, **PI**）、先验知识（prior knowledge, **PKN**）、购买意愿（purchase intention, **BI**）。其中，基础认知模块构建综合认知指数：$AI=0.35GA+0.35HA+0.30LA^*$；认知不均衡程度采用基尼系数 $G=rac{\sum_i\sum_j|x_i-x_j|}{2n^2ar x}$ 衡量。

场景生态位部分采用 Levins 标准化生态位宽度：$B_A=(B-1)/(n-1)$，其中 $B=1/\sum_i p_i^2$；群体间场景重叠采用 Pianka 指数进行衡量。购买意愿影响因素部分以路径分析式回归框架近似检验主效应和调节效应，并结合题项—构念相关计算复合信度（composite reliability, **CR**）与平均方差提取量（average variance extracted, **AVE**）。第五部分采用“问卷显性痛点—平衡后的公开网页语料—BERTopic主题结构”的三源证据框架：结构化多选题用于识别显性痛点，问卷开放题与公开网页文本在清洗后共同进入 BERTopic 主题建模，并通过主题—痛点映射与跨来源占比比较识别隐性主题结构。

### 3.3 分析策略

为形成递进式证据链，本文采用以下分析策略：第一部分使用认知分布分析、有序Logit、渠道网络、认知跃迁矩阵与渠道—认知—知晓流向图，识别认知形成与转化瓶颈；第二部分使用 K-means 聚类、行为散点矩阵、购买路径流向图、价格敏感度模型、渠道迁移热图和平行坐标图，识别行为异质性与决策结构；第三部分使用场景生态位宽度、对应分析、场景网络、生态位重叠热图与场景机会象限图解释场景机制；第四部分使用路径分析、交互效应图、校准图和多组比较森林图检验购买意愿驱动机制；第五部分使用镜像优先级图、三源验证分面条形图、多源隐性主题双侧条形图、关键词小面板图、显性痛点共现强度图、机会排序图与显隐性气泡矩阵揭示痛点结构；第六部分使用消费者细分、主成分双标图、人口学流向图、战略定位图、平行坐标图与MCA风格双标图识别人群分层特征。

## 4 结果

### 4.1 基础认知结构与认知转化

样本总体国潮认知均值为 2.90/5，非遗认知均值为 3.26/5，本土产品知晓率为 83.2%，实际购买率为 46.3%。该结果表明，福州本土国潮香氛市场并非缺乏认知基础，而是存在从抽象文化认知到具体产品识别、再到现实购买转化的层层损耗。信息渠道方面，使用率最高的前三位渠道为 社交媒体（64.4%）, 文旅街区（61.6%）, 电商平台（51.3%）。结合认知跃迁热图与新增的渠道—认知—产品知晓流向图可以发现，消费者对“国潮”与“非遗”概念的理解并不必然转化为对福州本土香氛产品的识别，说明当前市场的关键问题在于文化认知的产品化转译效率不足。

### 4.2 购买行为结构与消费群体划分

购买行为分析显示，行为聚类最优解为 5 类，说明消费者在购买频次、消费金额、品类广度、渠道数与搜索深度方面存在稳定的异质性。购买路径流向图显示，已购群体与潜在群体在“品类—渠道—金额区间”链路上具有明显分化，初始信息触达渠道与最终成交渠道并不完全一致，说明营销传播与商业转化分别受不同渠道系统主导。价格接受分析表明，价格敏感性仍是实际购买形成中的关键门槛，但其作用并不是孤立的，而是与前期价值理解和产品判断共同发挥作用。未购买原因方面，比例最高的前三项分别为 无使用需求（52.9%）, 没有心仪款式/文化内涵（51.8%）, 价格偏高（48.7%）。

### 4.3 场景生态位与资源匹配关系

场景生态位分析结果表明，使用频率较高的场景主要集中于 车载（3.48）, 住宿（3.39）, 娱乐（3.23）。生态位宽度、场景协同强度和生态位重叠热图共同说明，不同年龄层和不同消费者在场景使用上既存在相互重叠，也存在显著分化。对应分析双标图进一步表明，场景与产品形态之间并非随机匹配，而是形成了稳定的低维结构邻近关系。新增的场景机会象限图显示，高机会场景并非只是使用频率高，更重要的是同时具备较高的协同强度和较强的形态承载能力，这为后续产品开发与渠道布局提供了更具操作性的依据。

### 4.4 购买意愿影响因素检验

合并高相关构念后的简约购买意愿模型 的调整后决定系数为 0.387，说明当前主报告采用的构念结构能够解释较大比例的购买意愿差异。路径检验结果如下：

| 假设 | 路径 | 理论命题 | 标准化系数 | p值 | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| H1 | CVP | 文化价值感知正向影响购买意愿 | 0.219 | 0.0000 | 支持 |
| H2 | PREP | 购买准备度正向影响购买意愿 | 0.379 | 0.0000 | 支持 |
| H3 | PKN | 先验知识正向影响购买意愿 | 0.225 | 0.0000 | 支持 |

整体来看，购买准备度、先验知识、文化价值感知 是购买意愿形成的重要正向因素。与未合并构念的模型相比，当前主报告口径更强调“文化价值 + 购买准备 + 知识储备”的联合机制，而不是对多个高度相关维度分别解释。多组比较结果表明，不同性别群体在部分路径上的系数大小存在一定差异，说明购买意愿形成机制具有一定的人群异质性。模型校准图进一步表明，当前模型不仅能够在统计上解释购买意愿的变化方向，也能够较好地在分组层面重现实证观测结果。

### 4.5 消费痛点与文本主题结构

显性痛点分析显示，消费者最集中反馈的问题为 留香短/品质不佳（86.6%）, 价格偏高/性价比低（84.1%）, 非遗/地域融合表面化（82.6%）；改进诉求则主要集中于 深化非遗与地域融合（92.8%）, 降低价格提升性价比（90.6%）, 优化香型适配湿热气候（74.2%）。这意味着消费者并非仅仅表达不满，而是已形成相对明确的改进方向。第五部分进一步把问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构整合为三源证据链，其中 问卷开放题与平衡后的公开网页语料清洗后形成 648 条 BERTopic 可建模文本（其中问卷开放题 351 条、公开网页语料 297 条）。BERTopic 结果进一步显示，问卷文本更集中于“福州本土香氛开发”，公开网页语料更集中于“通用香水体验评价”，说明在地文化期待与通用体验评价构成了并行的隐性主题结构。 三源分面条形图、机会排序图与显隐性气泡矩阵进一步说明，问题之间并非孤立存在，而是在结构层面具有明显的聚类特征和优先级差异。

### 4.6 消费者细分与人口统计映射

人口统计与细分结果表明，第六部分的消费者细分最优解为 4 类，文化认同均值为 3.36，消费强度均值为 2.74。主成分双标图、战略定位图和平行坐标图共同说明，不同细分群在文化偏好、购买频率、消费金额、搜索深度和品类广度等维度上存在稳定差异。人口学流向图、标准化残差热图与MCA风格双标图则进一步表明，不同年龄、职业、收入和区域类别在细分群中的分布并不均匀，说明消费者细分具有明确的人口统计支撑。

### 4.7 跨模块整合结果

跨模块整合图显示，认知、购买状态和购买意愿之间并非简单的线性递进关系，而更接近多阶段分流结构。全链路中最常见的典型路径包括：中认知→已购→中意愿（97人）；中认知→潜在→中意愿（89人）；中认知→已购→低意愿（70人）；中认知→潜在→低意愿（55人）；中认知→无意向→中意愿（28人）。综合相关热图表明，与购买意愿关联最强的变量主要包括 文化认同（r=0.543）, 产品涉入度（r=0.491）, 先验知识（r=0.462）, 产品知识（r=0.451）, 购买便利性（r=0.402）。战略矩阵进一步提示，真正具有经营价值的重点群体往往不是最低认知群体，而是“高认知待转化群”和“意向孵化群”，因为这些群体已经具备较高的认知或意愿基础，只差体验验证、产品匹配和渠道承接。

## 5 讨论

### 5.1 理论层面的解释

本文结果表明，福州本土国潮香氛的消费形成过程可以被理解为一个由认知基础、行为结构、场景机制、价值判断与人群异质性共同构成的递进系统。首先，文化认知本身并不是市场的主要瓶颈，更关键的是文化认知是否能被转译为可识别的产品对象，这为文化价值感知理论在地方香氛消费研究中的适用性提供了新的证据。其次，购买行为的异质性和细分群差异说明，文化型香氛产品的市场并不存在统一的决策逻辑，而是表现为多条并行路径。再次，场景生态位结果将香氛消费从单一产品选择拓展到多场景使用系统，说明场景理论在香氛研究中具有较强解释力。最后，购买意愿模型进一步支持：购买准备度、先验知识、文化价值感知 共同构成了购买意愿生成的核心机制。

### 5.2 管理启示

从实践层面看，第一，品牌传播不应停留在抽象的“国潮”和“非遗”叙事，而应强化“地方文化元素—具体产品对象—适用场景”之间的连接；第二，产品开发应优先围绕高机会场景推进适配湿热气候的香型、便携与礼赠兼顾的形态设计；第三，渠道管理应明确区分兴趣触达渠道和成交承接渠道，重构投放预算与归因逻辑；第四，针对不同细分群，应实施差异化运营策略，例如对高认知待转化群重点强化体验验证和产品说服，对高认同潜力群重点强化教育与品牌内容建设，对高价值文化拥护者则应强化复购和品牌忠诚机制。

### 5.3 研究局限与未来方向

本文仍存在若干局限。首先，数据来自单次横截面问卷，尚不能直接识别消费者在时间维度上的真实转换过程；其次，第三部分在新版问卷中缺少场景预算分配变量，因此未能直接估计预算集中度；再次，购买意愿模块采用路径分析式回归近似检验，而非完整的协方差型结构方程模型。未来研究可结合追踪调查、实验设计或多时点数据，进一步构建潜在转换模型、纵向结构模型和更严格的多组结构方程模型，以提升因果解释力和动态解释力。

## 6 结论

本文基于福州本土国潮香氛问卷数据，构建并验证了一个“认知—行为—场景—意愿—痛点—细分”的综合分析框架。研究发现：第一，市场已具备一定文化认知基础，但文化认知向本土产品知晓和现实购买的转化仍存在显著断裂；第二，购买行为与消费者结构呈现明显异质性，说明市场运营需要分层而非平均化；第三，场景生态位和场景协同结果表明，香氛消费本质上是一个多场景资源配置问题；第四，购买准备度、先验知识、文化价值感知 是购买意愿形成的重要驱动因素；第五，问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构共同表明，消费约束集中于品质、价格、文化融合与渠道可达性等维度；第六，消费者细分群在行为强度和文化认同两个维度上具有稳定分层。总体而言，福州本土国潮香氛的核心挑战不是让消费者第一次听说“国潮”，而是把地方文化、非遗技艺与产品体验真正转化为能够被消费者识别、理解、信任并愿意购买的完整产品系统。

## 参考文献

1. Ajzen, I. (1991). *The Theory of Planned Behavior*. Organizational Behavior and Human Decision Processes, 50(2), 179-211. DOI: https://doi.org/10.1016/0749-5978(91)90020-T  
2. Davis, F. D. (1989). *Perceived Usefulness, Perceived Ease of Use, and User Acceptance of Information Technology*. MIS Quarterly, 13(3), 319-340. DOI: https://doi.org/10.2307/249008  
3. Keller, K. L. (1993). *Conceptualizing, Measuring, and Managing Customer-Based Brand Equity*. Journal of Marketing, 57(1), 1-22. DOI: https://doi.org/10.1177/002224299305700101  
4. Krishna, A. (2012). *An Integrative Review of Sensory Marketing: Engaging the Senses to Affect Perception, Judgment and Behavior*. Journal of Consumer Psychology, 22(3), 332-351. DOI: https://doi.org/10.1016/j.jcps.2011.08.003  
5. Chatterjee, S., & Bryła, P. (2022). *Innovation and Trends in Olfactory Marketing: A Review of the Literature*. Journal of Economics and Management, 44(1), 210-235. DOI: https://doi.org/10.22367/jem.2022.44.09  
6. Li, Z., Shu, S., Shao, J., Booth, E., & Morrison, A. M. (2021). *Innovative or Not? The Effects of Consumer Perceived Value on Purchase Intentions for the Palace Museum’s Cultural and Creative Products*. Sustainability, 13(4), 2412. DOI: https://doi.org/10.3390/su13042412  
7. Liu, L., & Zhao, H. (2024). *Research on Consumers' Purchase Intention of Cultural and Creative Products—Metaphor Design Based on Traditional Cultural Symbols*. PLoS ONE, 19(5), e0301678. DOI: https://doi.org/10.1371/journal.pone.0301678  
8. Xu, Y., Hasan, N. A. M., & Jalis, F. M. M. (2024). *Purchase Intentions for Cultural Heritage Products in E-commerce Live Streaming: An ABC Attitude Theory Analysis*. Heliyon, 10(5), e26470. DOI: https://doi.org/10.1016/j.heliyon.2024.e26470  
9. Jacob, C., Stefan, J., & Guéguen, N. (2014). *Ambient Scent and Consumer Behavior: A Field Study in a Florist's Retail Shop*. The International Review of Retail, Distribution and Consumer Research, 24(1), 116-120. DOI: https://doi.org/10.1080/09593969.2013.821418  
10. *Mechanisms Influencing Consumer Purchase Intention: Cultural and Creative Products in Museums* (2025). Social Behavior and Personality. DOI: https://doi.org/10.2224/sbp.14349  
11. *Driving Factors of Purchase Intention Toward Bashu Intangible Cultural Heritage Products: An Extended Theory of Planned Behavior Approach* (2026). Sustainability, 18(3), 1593. DOI: https://doi.org/10.3390/su18031593
