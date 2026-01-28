# Mugi：医学单元梯度迭代算法 (Medical Unit Gradient Iteration)

**Mugi reframes knowledge from a static collection of facts into a condition-conditioned, continuously adjustable structure that evolves by minimizing epistemic inconsistency under human feedback.**

**Mugi** 是一个创新的医学知识图谱动态演化系统。它旨在弥合大规模语言模型（LLM）的“黑盒”推理与医疗领域高可解释性需求之间的鸿沟。Mugi 能够深度提取并结构化 LLM 中蕴含的内生医学知识，将其转化为显式、可回溯的图谱拓扑结构。

**Mugi**的四大核心优势：

1. 内生知识重构：

   利用 DeepSeek-R1 等顶级大模型作为底层算力，提取其神经网络中隐藏的医学逻辑，打破“幻觉”黑盒，实现从“概率预测”到“逻辑结构”的转化。

2. 显式存储与高可解释性：

   所有提取出的医学单元（Nodes）与关系（Edges）均持久化存储于 Neo4j 知识图谱中。通过显式的拓扑路径展示诊断逻辑，确保每一项预测均有据可查，满足医疗场景严苛的合规性与可信任要求。

3. 极低专家标注压力：

   引入基于 $\Psi$ 值（信息增益/不确定性）的主动学习机制。系统仅在算法遇到逻辑瓶颈时，精准向专家端推送高价值的可疑边进行标记。这种模式将专家的介入降至最低，极大提升了知识标注的边际效能。

4. 自主梯度演化：

   系统通过前向传播验证与反向梯度更新，在无需人工干预的情况下，周期性地执行节点合并、边剪枝与权重修正。MUGI 不仅仅是一个静态数据库，而是一个能够随数据流持续成长、自我优化的“活”的医学数字生命体。



## Mugi 的基础数学结构

Mugi 的核心数据结构是Mugi的知识图谱 $G=(\{V\},f)$ 。Mugi的本质功能是通过操纵大模型API来提取知识，完善图谱 $G$。

$\{V\}$ 是图谱$G$中所有不同种类节点集合的集合，因此 $\{V\}$ 由一系列互斥子集$V$组成。在医学相关任务中，$\{V\}$ 的构成如下：
$$
\{V\}=\{\text{Diagnosis}, \text{Symptoms}, \text{Checks}, \text{Treatments}\}
$$
其分别表示诊断、症状、检测、处置。

对任意 $V \subset \{V\}$，$V$ 是单独的节点$v$构成的集合（例如 $V=\text{Diagnosis}$）。每一个节点 $v$ 表示某一个具备特定语义的对象，例如肺癌、CT、加强CT、穿刺、化疗等。但 $v$ 在Mugi中并不以字符串的形式存在。$v$ 是一个长度为768的向量，即 $v \in \mathbb{R}^{768}$。$v$ 实际上是具备特定医学语义的token经过Bert模型向量化得到的结果。即，若一个节点 $v$ 表示 “非小细胞肺癌”，那么它实际上满足：
$$
v_i = \text{Bert(input="非小细胞肺癌")}, v_i \in \mathbb{R}^{768}
$$
这里 Bert 模型需要使用 ClinicalBert 或 BioBert 等变体。很显然每个节点 $v$ 都对应一个表示其种类的父集合，我们规定运算 $\mathcal{T}$ 如下：
$$
\forall v \in V, \mathcal{T}(v) = V, V \subset \{V\}
$$
实际上 $\mathcal{T}$ 运算就是求一个节点 $v$ 的种类，例如，若 $v_i = \text{Bert("非小细胞肺癌")}$，则有 $\mathcal{T}({v_i})=\text{Diagnosis}$。

$f$ 是以从属于两个不同的集合 $V$ 的元素为自变量的映射，可以将其理解为：
$$
f: v_i, v_j \to r'_{ij}, \mathcal{T}(v_i) \ne \mathcal{T}(v_j), r'_{ij} \in [-1, 1]
$$
我们规定 $\mathcal{N}$ 运算表示与某个节点 $v_i$ 存在映射关系的所有节点 $v_j$ 的集合：
$$
\mathcal{N}(v_i) = \{v_j | v_j \in G \text{且} (f: v_i, v_j \to r_{ij}' \text{或} f:v_j, v_i \to r_{ji}' \text{存在})\}
$$
$f$ 的因变量 $r'$ 表示两个节点 $v_i, v_j$ 的**真实相关性**。相关性 $r'$ 越高则表示二者之间存在明确的关联。

真实相关性并不是一个固定的值，在具体的条件下，节点之间的相关性会有很大不同。例如节点 “胸痛” 与 “肺癌” 的关联性，在一个老年男性身上应该较高，毕竟老年人有更高的癌症患病率，出现相关症状也更意味着癌症风险。但对于年轻人，胸痛可能仅仅意味着普通的肺炎或下呼吸道的小疾病。因此，我们定义$r_{ij}$ 表明两个节点 $v_i, v_j$ 的**本征相关性**，这是一个不随条件转移的值，表示两个节点之间的本质联系。显然，真实的相关性 $r'$ 是一个与本征相关性 $r$ 和条件 $\mathbf{x}$ 都相关的函数。

## 节点生成

本章节讲述如何生成 Mugi 图谱中最初的节点，以及后续扩展节点的逻辑。

**初始节点生成**：

1. 构建骨架：利用肺癌诊疗的权威指南让 LLM 生成细分骨架。简易Prompt 逻辑：“请根据 NCCN 非小细胞肺癌指南，列出该领域下的所有二级子领域。”
2. 递归提取与结构化：针对每个子领域，递归构建子骨架（若存在），否则命令大模型输出该领域的实体，并将其归类到 $\{V\}$ 中。
3. 实体对齐与去重：参考标准医学术语词库，利用 LLM 将同义词映射为标准医学术语。
4. 向量碰撞检测：计算新候选词 $v_{new}$ 与图谱中已有节点 $v_{exist}$ 的余弦相似度。若 $\cos(v_{new}, v_{exist}) > \text{阈值 (如 0.95)}$，则判定为重复或高度近似，不创建新节点，由模型评估并选择合适的表述替代 $v_{exist}$。

**新节点扩展：**

1. 用户提交新节点：专家向 Mugi 反馈了一个新节点 $v_{new}$。
2. 节点匹配未命中：Mugi 在当前图谱中匹配新节点，若其和所有节点的相似度低于阈值则进入下一步。
3. LLM验证：LLM 对比 $v_{new}$ 和图谱中相似度 Top-K 的节点，重新验证该节点是否为新节点。
4. 零样本初始化：LLM 验证通过，选取图谱中同类别节点内在 Bert 空间相似度最高的节点，将其拓扑结构和边权复制给 $v_{new}$。 

## 定义：条件特征向量

首先，我们定义**条件特征向量** $\mathbf{x}$:
$$
\mathbf{x} = [\mathbf{x_1}, \mathbf{x_2}]^{T} \in \mathbb{R}^{1 \times (m+n)}, \mathbf{x_1} \in \mathbb{R}^{m}, \mathbf{x_2} \in \mathbb{R}^{n}
$$
$\mathbf{x_1}$ 表示 $\mathbf{x}$ 的中可直接量化的指标构成的向量，例如年龄、身高、体重、BMI、性别、血压等。它们本身就是数据，可以直接通过归一化的方式处理成向量。

以下是一个初步的 $\mathbf{x_1}$ 的设计方案，我们令 $\mathbf{x_1} = [a_1, a_2, ..., a_{10}]$

| **维度** | **指标名称**          | **单位** | **建议归一化范围 (min∼max)** | **说明**                       |
| -------- | --------------------- | -------- | ---------------------------- | ------------------------------ |
| $a_1$    | **年龄 (Age)**        | 岁       | $0 \sim 100$                 | 反映机体衰老程度               |
| $a_2$    | **性别 (Gender)**     | 类       | $\{0, 1\}$                   | 1 为男，0 为女                 |
| $a_3$    | **身高 (Height)**     | cm       | $100 \sim 220$               | 用于关联发育特征               |
| $a_4$    | **体重 (Weight)**     | kg       | $30 \sim 150$                | 基础代谢指标                   |
| $a_5$    | **BMI**               | $kg/m^2$ | $10 \sim 50$                 | 肥胖/消瘦的综合量化指标        |
| $a_6$    | **收缩压 (SBP)**      | mmHg     | $70 \sim 200$                | 高血压诊断核心指标             |
| $a_7$    | **舒张压 (DBP)**      | mmHg     | $40 \sim 130$                | 循环系统压力指标               |
| $a_8$    | **心率 (Heart Rate)** | bpm      | $40 \sim 180$                | 自律神经与心脏机能             |
| $a_9$    | **体温 (Temp)**       | °C       | $35 \sim 42$                 | 感染与炎症的强信号             |
| $a_{10}$ | **空腹血糖 (FBG)**    | mmol/L   | $2 \sim 25$                  | 代谢性疾病（如糖尿病）基础指标 |

显然我们不能直接将原始数据填入 $\mathbf{x_1}$ 中，需要进行归一化操作。假设我们获取到的原始数据是 $a_{raw}$，归一化方法如下：
$$
a = \frac{a_{raw} - a_{min}}{a_{max} - a_{min}}
$$
$\mathbf{x_2}$ 表示 $\mathbf{x}$ 中不可量化的指标构成的向量，包括患者的病史、既往史、接触史等等。接下来的步骤展示如何将这些信息处理成向量。

我们令 $\mathbf{x_2} = [S_{static}, C_{cluster}, R_{res}]$，分别表示静态核心槽位 Static、动态领域聚类池 Cluster 和语意残差向量 res。每个槽位对应的值是该因素的强度，强度越大表示该因素对患者的程度越深。

$S_{static}$ 向量表示最核心、诊断考虑频率最高的指标，令 $S_{static} = [s_1, s_2, ..., s_{10}]$：

| **维度**     | **静态槽位分量**         | **临床含义**       | **强度 α 定义示例**             |
| ------------ | ------------------------ | ------------------ | ------------------------------- |
| **行为嗜好** | $s_1$: 烟草暴露          | 主/被动吸烟强度    | 0:不吸；1:重度/长期/粉尘共存    |
|              | $s_2$: 乙醇摄入          | 饮酒频率与量级     | 0:不饮；1:长期酗酒/酒精依赖     |
| **环境职业** | $s_3$: 关键物理/化学暴露 | 是否接触公认致病源 | 0:无；1:长期职业性高危接触      |
| **生理代谢** | $s_4$: 睡眠剥夺          | 节律破坏程度       | 0:规律；1:长期失眠/严重昼夜倒错 |
|              | $s_5$: 慢性应激          | 心理/生存压力      | 0:平稳；1:高度焦虑/长期应激     |
| **遗传背景** | $s_6$: 早发慢病家系      | 血管/代谢遗传倾向  | 0:无；1:直系亲属多发早发病      |
|              | $s_7$: 家族肿瘤史        | 泛癌种易感背景     | 0:无；1:直系亲属有恶性肿瘤史    |
| **系统黄旗** | $s_8$: 不明原因消耗      | 体重/食欲异常      | 0:正常；1:半年内显著下降/厌食   |
|              | $s_9$: 持续性疲劳        | 能量代谢衰竭信号   | 0:无；1:静息状态下仍感重度疲劳  |
|              | $s_{10}$: 免疫背景       | 过敏或免疫亢进状态 | 0:正常；1:高度过敏体质/免疫疾病 |

$C_{cluster}$ 向量对应 5 大维度的语义聚类。当患者提到的因素不属于上述 10 个静态槽位时（例如“石棉暴露”、“极度久坐”），由大模型归类并计入该维度的最大强度值。令 $C_{cluster} = [c_{BA}, c_{EO}, c_{CM}, c_{DG}, c_{YF}]$

- **$c_{BA}$ (行为聚类)**：涵盖非核心嗜好（如偏食、药物滥用、缺乏运动）。
- **$c_{EO}$ (环境聚类)**：涵盖长尾暴露（如装修污染、电离辐射、特定地理病、噪音污染）。
- **$c_{CM}$ (节律聚类)**：涵盖其他代谢信号（如暴饮暴食习惯、久坐不动）。
- **$c_{DG}$ (遗传聚类)**：涵盖罕见综合征、特定遗传代谢缺陷。
- **$c_{YF}$ (黄旗聚类)**：涵盖全身性不适（如夜汗、反复低热、整体健康感骤降）。

计算公式：
$$
c_{Domain} = \max(\alpha_{factor1}, \alpha_{factor2}, \dots)
$$
这保证了即使患者有 10 种环境暴露，也不会因为累加导致该维度溢出，而是提取最严重的一个。

$R_{res}$ 捕捉其他信息，例如“语境”。

这是一组根据用户描述，由本地 BERT 降维生成的 16-32 维嵌入向量。它不负责解释具体是什么病，而是负责捕捉副词带来的“质感”。同时补充上述领域没有包含的部分。

## 定义：边

上文严格定义了条件特征向量 $\mathbf{x}$，条件特征向量用于描述一个特定的患者的状态。现在，我们利用条件特征向量来计算，对于特定患者，节点与节点之间的真实相关性。

对于节点 $v_i, v_j$，在条件 $\mathbf{x}$ 下的真实相关性 $r_{ij}'$ 定义为：
$$
r_{ij}' = r_{ij} \tanh (\gamma(\mathbf{w}_{ij}\mathbf{x} + b_{ij}))
$$
 其中 $\gamma$ 为**温度系数**，用于将计算结果压缩至双曲正切函数的灵敏区间内，是对所有边一致的超参数。$\mathbf{w}_{ij}$ 和 $b_{ij}$ 分别为**条件权重向量**和**条件偏置值**。我们令 $u_{ij} = \mathbf{w}_{ij}\mathbf{x} + b_{ij}$，令 $A_{ij} = \tanh(\gamma u_{ij})$。很显然，$A_{ij}$ 的物理意义是:
$$
A_{ij}: \text{当前条件与由节点 $v_i$ 推导到节点 $v_j$ 的条件的契合度}
$$
其中 $\mathbf{w} \in \mathbb{R}^{m + n}$，$b \in \mathbb{R}$。

在上述推导的基础上，我们可以定义图 $G$ 中两个节点 $v_i, v_j$ 之间的**边** $e_{ij}$ 为一个由以下元素构成的集合：
$$
e_{ij} = \{v_i, v_j, r_{ij}, \mathbf{w}_{ij}, b_{ij}\}
$$
所有边 $e$ 的集合被称为 $E$。由此，我们可以将 Mugi 的图谱 $G$ 重新表述为：
$$
G = (\{V\}, E)
$$
显然，在现有体系下，针对每个输入的条件特征向量 $\textbf{x}$，图谱中的每条边 $e$ 都会给出一个对应的真实相关性 $r'$。由此我们可以给真实相关性$r'$ 的完整的物理意义描述：
$$
r_{ij}': \text{当前条件下由节点 $v_i$ 推导到节点 $v_j$ 的可行性}
$$

## 边初始化

本章节讨论如何初始化一条边的参数：**本征相关性** $r$，**条件权重向量** $\mathbf{w}$ 和**条件偏置值** $b$。

首先，对于本正相关性 $r$，直接利用大模型生成初始值。对于其余二者，通过下面的流程来实现初始化。

其次，对于需要进行初始化的边 $e$，利用大模型给出一组正样本集合 $X^+$ 和一组负样本集合 $X^-$。每组集合包含若干典型的条件特征向量 $\mathbf{x}$，作为优化过程的数据来源。

我们定义一个优化目标 $J(\mathbf{w}, b)$ 如下：
$$
J(\mathbf{w}, b) = \frac{1}{|X^+|}\sum_{\mathbf{x}^+ \in X^+} \tanh(\gamma u^+) - \frac{1}{|X^-|}\sum_{\mathbf{x}^- \in X^-} \tanh(\gamma u^-) - \text{Reg}(\mathbf{w})
$$
其中，$u^+ = \mathbf{wx}^+ + b, u^- = \mathbf{wx}^- + b$。$\text{Reg}$ $(\mathbf{w})$ 是正则化项，防止权重 $\mathbf{w}$ 在迭代时过拟合或者梯度爆炸，$\lambda_1, \lambda_2$ 为对应正则化系数，属于超参数。$\text{Reg}$ $(\mathbf{w})$ 计算方式如下：
$$
\text{Reg$(\mathbf{w})$} = \lambda_1||\mathbf{w}||_1 + \frac{\lambda_2}{2}||\mathbf{w}||_2
$$
显然这个优化目标的物理意义是：
$$
J(\mathbf{w}, b): 当前边对正样本预测的真实相关性和对负样本预测的真实相关性的差值
$$
显然，通过优化 $J(\mathbf{w}, b)$，我们可以得到最能够区分正样本和负样本的权重。不妨令 $A_+ = \tanh(\gamma u^+)$，$A_- = \tanh(\gamma u^-)$，又因为$\nabla_bu = 1$，因此可以得到 $J$ 对 $b$ 的梯度如下：
$$
\nabla_bJ = \nabla_uJ =\frac{1}{|X^+|}\sum_{\mathbf{x}^+ \in X^+}\gamma(1-A_+^2) - \frac{1}{|X^-|}\sum_{\mathbf{x}^- \in X^-}\gamma(1-A_-^2)
$$
同时注意到 $\nabla_{\mathbf{w}}u = \mathbf{x}$，故在不考虑正则项的情况下，有：
$$
\nabla_\mathbf{w}J = \nabla_uJ \cdot \mathbf{x} =\frac{1}{|X^+|}\sum_{\mathbf{x}^+ \in X^+}\gamma(1-A_+^2)\mathbf{x}^+ - \frac{1}{|X^-|}\sum_{\mathbf{x}^- \in X^-}\gamma(1-A_-^2)\mathbf{x}^-
$$
加入正则项后：
$$
\nabla_\mathbf{w}J  =\frac{1}{|X^+|}\sum_{\mathbf{x}^+ \in X^+}\gamma(1-A_+^2)\mathbf{x}^+ - \frac{1}{|X^-|}\sum_{\mathbf{x}^- \in X^-}\gamma(1-A_-^2)\mathbf{x}^- - \lambda_1\text{sgn}(\mathbf{w}) - \lambda_2 \mathbf{w}
$$

在明确了上述梯度之后，只需要使用梯度上升算法最优化 $J(\mathbf{w}, b)$ 即可得出最优的条件权重向量 $\mathbf{w}$ 和条件偏置值 $b$。梯度上升算法的单次迭代过程如下：
$$
b \gets b + \theta_b \nabla_bJ
$$

$$
\mathbf{w} \gets \mathbf{w} + \theta_{\mathbf{w}}\nabla_{\mathbf{w}}J
$$

在上述算法中，$\theta_b, \theta_{\mathbf{w}}$ 为学习率，属于超参数。另外， 在完成了梯度上升算法之后，需要再额外对 $\mathbf{w}$ 进行一次软阈值处理，将较小的权重直接置0，使 $\mathbf{w}$ 整体更加稀疏，方法如下：
$$
\mathbf{w} \gets \text{sgn}(\mathbf{w}) \max(0, |\mathbf{w}| - \tau)
$$
其中超参数 $\tau$ 表示阈值，上述算法本质上是绝对值小于 $\tau$ 的权重直接置0。

## 前向传播

本章节描述Mugi智能体根据图谱进行推理的过程，推理过程所依照的主要凭据是真实相关性。

**示例情形：用户描述症状，询问是何种疾病**

1. 大模型收到用户Prompt，确认这是一个由症状集合向诊断集合推理的一个请求：$\text{Symptoms} \to \text{Diagnosis}$。

2. 大模型解析用户Prompt，解析出用户提供的症状集合 $S = \{\text{胸痛}, \text{咳嗽}, \text{发烧}, ...\}$。

3. 元素向量化（Text2Vector）：$S \gets \text{Bert(S)}$

4. 节点匹配（V-Match）：查找 $\text{Symptoms}$ 集合中每个元素在用户症状集中最接近的元素
   $$
   \forall v_i \in \text{Symptoms}, \alpha_i = \max_{s \in S}(\cos(s, v_i))
   $$
   寻找 $\alpha$ 最大的 $k_1$ 个节点，构造一个子集合 $V_{sub}' = \{v_1', v_2',..., v_{k_1}'\}$

5. 边推理（E-Infer）：对子集合 $V_{sub}'$ 中的每个元素 $v_i$，查找它们到 $\text{Diagnosis}$ 集合的边，并计算
   $$
   \forall v_j \in \text{Diagnosis} \text{且} v_j \in \mathcal{N}(v_i), \beta_{ij} = \alpha_i r_{ij}'
   $$
   对每个元素 $v_i \in V_{sub}'$，寻找 $\beta$ 值最大的 $k_2$ 个节点 $v_j$，将所有 $v_j$ 加入待选集合 $V_{result}$。

6. 能量聚合（A-Aggregation）：对待选集合 $V_{result}$ 中的每个元素，计算其接收到的来自集合 $V_{sub}'$ 的所有节点的 $\beta$ 值的总和。
   $$
   \forall v_j \in V_{result}, \epsilon_{j} = \sum_{v_i \in \mathcal{N}(v_j)\cap V_{sub}'} \beta_{ij}
   $$
   将所有计算得到的 $\epsilon_{j}$ 合并为一个向量后计算 $\text{Softmax}$ （单一诊断）或对每个元素单独使用 $\text{Sigmoid}$ （多诊断）来评分，其计算结果作为 $V_{result}$ 中每个元素（诊断）的概率。

7. 结果输出：对于诊断概率高于指定阈值的诊断，输出并返回。

## 反向传播

本章节描述Mugi智能体在接收到专家给出的标注信息时，依据梯度下降算法对权重进行迭代的过程。

核心目的：根据专家的反馈，修正每个边的**本征相关性** $r$，**条件权重向量** $\mathbf{w}$ 和**条件偏置值** $b$。

规定人工专家给出的标签 $y$ 表示真实相关性 $r'$ 的准确值，因此，我们可以构造损失函数：
$$
\mathcal{L} = \frac{1}{2}(r' - y)^2 + \text{Reg$(\mathbf{w})$}
$$
显然 $\mathcal{L}$ 对 $r'$ 的梯度：
$$
\delta_{out} = \frac{\partial \mathcal{L}}{\partial r'} = r' - y
$$
又因为：
$$
r' = r \tanh (\gamma(\mathbf{w}\mathbf{x} + b)) = r\tanh(\gamma u) = rA
$$
有：
$$
\frac{\partial \mathcal{L}}{\partial r} = \delta_{out}A
$$
同时不难发现 $\mathcal{L}$ 对 $u = \mathbf{wx} + b$ 的梯度为：
$$
\delta_u = \frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \mathcal{L}}{\partial r} \cdot \frac{\partial r}{\partial A} \cdot \frac{\partial A}{\partial u} = \delta_{out} r(1-A^2)
$$

$$
\nabla_{\mathbf{w}}u = \mathbf{x}, \nabla_bu = 1
$$

因此，针对本征相关性 $r$ 的梯度下降算法如下：
$$
r \gets r - \eta_r\delta_{out}A
$$
针对条件权重向量 $\mathbf{w}$ 和条件偏置值 $b$ 的梯度下降算法的单次迭代过程如下：
$$
\mathbf{w} \gets \mathbf{w} - \eta_{\mathbf{w}}(\delta_u \mathbf{x} + \lambda_1\text{sgn}(\mathbf{w}) + \lambda_2 \mathbf{w})
$$

$$
b \gets b - \eta_b \delta_u
$$

利用上述梯度下降算法进行多轮次的迭代，可以让条件权重向量回归到正确值，此时损失函数最小。

接下来是反向传播的**节点扩散**机制。节点扩散机制保证在专家对某一条边 $e_{ij}$ 进行标注后，模型可以自己推理出类似的边的权重和相关性，并自行修改，以降低专家标注压力。假设专家完成了某一条边 $e_{ij}$ 的标注。

节点扩散机制为，$\forall v_m \in \{v|\mathcal{T}(v)=\mathcal{T}(v_i)\}$，有：
$$
r_{mj} \gets r_{mj} - \eta_r\mathcal{K}(v_i, v_m) e^{-\lambda_c\mathcal{C}_{mj}^{-1}} \delta_{out}A
$$

$$
\mathbf{w}_{mj} \gets \mathbf{w}_{mj} - \eta_{\mathbf{w}}\mathcal{K}(v_i, v_m) e^{-\lambda_c\mathcal{C}_{mj}^{-1}} (\delta_u \mathbf{x} + \lambda_1\text{sgn}(\mathbf{w}) + \lambda_2 \mathbf{w})
$$

$$
b_{mj} \gets b_{mj} - \eta_b\mathcal{K}(v_i, v_m) e^{-\lambda_c\mathcal{C}_{mj}^{-1}} \delta_u
$$

$\forall v_n \in \{v|\mathcal{T}(v) = \mathcal{v_j}\}$,同样有类似步骤，此处略。

其中，$\mathcal{K}(v_i, v_j)$ 函数用于计算两个节点之间的相似度，计算方式如下：
$$
\mathcal{K}(v_i, v_j) = 
\begin{cases}
e^{-\lambda_k(1-\cos(v_i, v_j))}, & \cos(v_i, v_j) > \tau_k\\
0, & \cos(v_i, v_j) \le \tau_k
\end{cases}
$$
在上述梯度下降算法中，$\eta_r, \eta_{\mathbf{w}}, \eta_b$ 分别为对应的学习率，属于超参数。$\lambda_c, \lambda_k$ 为相关超参数且均大于0。$\tau_k$ 是一种阈值，相似度大于该 $\tau_k$ 时才会进行节点扩散。$\mathcal{C}_{ij}$ 是标注间隔因子，是一个与 “上次专家标注时间” 有关的大于0的值，距离上次标注时间越短，$\mathcal{C}$ 越小，设置这一参数用于防止专家的正确标注值受到节点扩散机制的影响而产生噪声。

## 主动学习和可疑边筛查

本章节探讨 Mugi 算法如何通过自学习查找低置信度的边并将其推送给专家进行标记。

定义：
$$
\Psi_{ij} =  \mathcal{C}_{ij} [\mathcal{H}_{ij}, \sigma_{ij}, \mathcal{R}_{ij}] \cdot \mathcal{w}
$$
为边 $e_{ij}$ 的**可疑程度**，$\Psi$ 值越高表示该边越需要被专家标注。$\mathcal{w} \in \mathbb{R}^{1 \times 3}$ 为四个子因素的加权值，属于超参数。四个子因素定义如下：

**模糊程度** $\mathcal{H}$，满足：
$$
\mathcal{H}_{ij} = - r_{ij}\log r_{ij}
$$
根据上述公式不难看出，当本征相关性 $r$ 越接近中间值0.5时，模糊程度越大，对应的 $\mathcal{H}$ 越大。

**标准差** $\sigma$：由大模型随机生成或从条件特征向量库中选取一组近似的条件特征向量 $X_n$，其长度为 $n$，规定：
$$
\forall \mathbf{x}_k \in X_n, r_k' = r_{ij}\tanh(\gamma(\mathbf{w}_{ij}\mathbf{x} + b_{ij}))
$$
有：
$$
\sigma_{ij} = \sqrt{\sum_{\mathbf{x}_k \in X_n}\frac{(r_k' - \bar{r})^2}{n-1}}
$$
**边权** $\mathcal{R}$，表示边 $e_{ij}$ 关联的非自身的边的重要性，边权越大表示越多的推理路径会经过该边，则越需要标注，边权计算方式如下：
$$
\mathcal{R}_{ij} = \sum_{v_k \in \mathcal{N}(v_i)}r_{ik} + \sum_{v_k \in \mathcal{N}(v_j)}r_{jk} - r_{ij}
$$
**标注间隔因子** $\mathcal{C}$，是与该边上次标注时间有关的值，计算方式如下：
$$
\mathcal{C}_{ij} = \frac{1}{1 + e^{-\lambda_t (\Delta t - T)}}
$$
在上述公式中，$\lambda_t$ 和 $T$ 属于超参数，$\lambda_t$ 控制增长的速度，$T$ 表示标注后的一个缓慢增长的冷却期长度，$\Delta t$ 表示距离上次标注的时间间隔。当 $\Delta t \ll \tau$ 时，指数项极大，$\mathcal{C}_{ij} \approx 0$。即使该边的其他指标很高，由于 $\mathcal{C}$ 的抑制，总 $\Psi$ 值也会很低，避免重复标注。当 $\Delta t > \tau$ 后，$\mathcal{C}$ 开始快速上升。当 $\Delta t \gg \tau$ 时，$\mathcal{C}_{ij} \to 1$。此时该因子不再抑制 $\Psi$ 值，该边是否被推送完全由其本身的模糊度和重要性决定。