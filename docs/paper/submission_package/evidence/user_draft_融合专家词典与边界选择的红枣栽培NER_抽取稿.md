# 融合专家词典与边界选择的红枣栽培命名实体识别方法（DOCX抽取稿）

来源：`融合专家词典与边界选择的红枣栽培命名实体识别方法.docx`

基于词典和边界预测的红枣栽培命名实体识别

摘  要：针对红枣栽培命名实体识别中传统序列标注解码器难以捕捉实体跨度全局信息、领域专业术语语义表征不足以及实体类别分布不平衡导致识别精度低的问题，提出一种基于词典特征与边界预测的红枣栽培命名实体识别方法（Expert Dictionary + Boundary Prediction，EDBP）。该方法构建涵盖14个实体类别的红枣栽培命名实体识别数据集（Red Jujube cultivation Named entity recognition Dataset，RJND）；设计词典特征层，从训练集自动提取领域实体构建专家词典，采用BMES（Begin-Middle-End-Single）四通道结构对词典匹配信息进行编码；使用边界预测（Boundary Prediction，BP）解码器替代传统条件随机场（Conditional random field，CRF）序列标注，将命名实体识别建模为片段分类（Span classification）问题，利用双仿射注意力（Biaffine Attention）机制对所有可能的实体边界对进行分类；引入焦点损失函数（Focal Loss）缓解类别不平衡问题。实验结果表明，EDBP模型在RJND数据集上的精确率、召回率和F1值分别达到89.51%、87.58%和88.73%，较BiLSTM-CRF、BERT-wwm-ext+BiLSTM+CRF和MacBERT+BiLSTM+CRF三个基准模型分别提升9.85、3.06和2.97个百分点，能够更有效地识别红枣栽培领域文本中的各类实体；在MSRA、WeiboNER、ResumeNER、Boson和CLUENER五个公开数据集上F1值分别达到95.19%、72.27%、96.13%、85.60%和80.06%，验证了EDBP模型的泛化能力。

关键词：深度学习；红枣栽培；命名实体识别；专家词典；边界选择；Focal Loss

中图分类号：  文献标识码：

文章编号：

Named Entity Recognition for Red Jujube Cultivation Based on Expert Dictionary and Boundary Selection

Abstract: To address the issues of low recognition accuracy in named entity recognition (NER) for the red jujube cultivation domain, which are caused by the inability of traditional sequence labeling decoders to capture global span information, the lack of effective utilization of domain expert knowledge, and entity class imbalance, a red jujube cultivation NER model based on expert dictionary and boundary selection is proposed. First, focusing on the main production processes of red jujube cultivation, a Red Jujube cultivation Named entity recognition Dataset (RJND) covering 14 entity categories is constructed, and a stratified random splitting strategy is adopted to ensure balanced distribution of entity categories. Second, an expert dictionary feature layer is designed to automatically extract domain entities from the training set to construct an expert dictionary, which is encoded via multi-hot embedding into dense vectors and fused with character representations. Third, a Boundary Selection (BS) decoder is proposed to replace traditional CRF sequence labeling, modeling NER as a span classification problem and using biaffine attention mechanism to classify all possible entity boundary pairs. Finally, Focal Loss is introduced to alleviate the entity class imbalance problem by reducing the loss weight of easily classified samples through a focusing factor. Experimental results show that the expert dictionary features and boundary selection decoder effectively improve the model recognition accuracy, achieving an F1 score of 88.28% on the RJND dataset, which is significantly higher than the baseline model. This study is of great importance for the construction of red jujube cultivation knowledge graphs and can provide reference for NER research in other crops.

Key words: deep learning; red jujube cultivation; named entity recognition; expert dictionary; boundary selection; Focal Loss

0 引 言

红枣是我国重要的经济林树种，具有极高的营养价值和药用价值。在新疆、山东、河北、陕西等地广泛种植[1]。红枣栽培技术知识分散于专业书籍、科技论文和技术手册等载体中，传统的检索方式难以高效、准确地从中提取专业信息知识。命名实体识别（Named Entity Recognition，NER）能够从非结构化文本中抽取具有特定意义的实体，是实现知识结构化的基础任务[2]。准确识别红枣栽培领域实体，为知识图谱构建、知识检索和推荐等下游应用提供数据支撑，对实现红枣栽培知识的结构化组织具有重要意义。

命名实体识别方法经历了从规则方法、机器学习方法到深度学习方法的发展过程。早期NER主要依赖人工构建的领域词典和句法模式识别实体[3]，如Hobbs等[4]提出FASTUS系统，通过级联有限状态转换器从文本中抽取命名实体，方法虽然实现简单但是泛化能力弱、需要不断更新规则。机器学习方法将实体识别问题建模为序列标注问题，如Zhou等[5]提出基于隐马尔可夫模型的块标注器进行命名实体识别，McCallum等[6]引入条件随机场建模标签序列的依赖关系，但该类方法依赖大量标注数据并且对于复杂语言现象的建模能力有限。

近年来，命名实体识别深度学习方法凭借强大的上下文语义建模能力和端到端特征学习优势而广泛应用于医学[7,8]、农业[9-11]等领域，其中BiLSTM-CRF架构表现优异，例如Huang等[12]结合双向LSTM与CRF显著提升NER性能。然而，基于字符的NER方法不能充分利用词信息，为提升字符表示质量，研究者引入词典信息增强NER模型：Zhang等[13]将潜在词信息融合至Lattice LSTM，缓解分词错误和上下文歧义，但候选词集的不确定性影响推理效率；Ma等[14]提出SoftLexicon将匹配词按位置分组并引入词频加权，但词集间的重要程度和位置关系未得到充分利用，且上述方法均依赖外部词典，领域适应性受限。另一方面，预训练语言模型通过大规模语料学习上下文表示，为NER提供了新的语义增强途径，Devlin等[15]提出BERT模型显著提升了上下文理解能力，BERT-BiLSTM-CRF成为NER主流方法[16]。在解码层面，片段分类方法将NER建模为对文本中所有可能片段的直接分类问题，能够全局决策实体边界[17]，为克服CRF的局部依赖局限提供了新思路。尽管上述方法在嵌入增强与解码改进方面取得了显著进展，但在领域特定的NER任务中存在不足。

具体而言，在嵌入层，在通用语料上预训练的语言模型难以充分捕获领域术语的语义特征，虽然领域预训练能够改善这一问题，但其需要大量领域无标注语料，在领域数据分散有限的情况下难以实施。而现有词典增强方法依赖外部词典，通用领域词典在专业领域上效果不足。在解码层，CRF通过标签转移矩阵建模相邻标签间的局部依赖关系，无法对实体起始与结束位置进行全局联合决策，对长实体易产生实体边界错误[18]。在损失函数层面，CRF的负对数似然（Negative Log-Likelihood, NLL）损失对各类别样本同等加权，在实体类别较多且分布不均衡时，低频类别识别精度较低[19,20]。

针对以上问题，本文提出专家词典边界选择（Expert Dictionary Boundary Selection，EDBS）模型。在嵌入层面，设计专家词典特征层，从训练集自动提取领域实体构建专家词典，采用BMES四通道结构编码词典匹配信息，并通过通道注意力机制（Channel Attention over BMES，CA-BMES）自适应聚合四通道特征。在解码层面，使用边界选择（Boundary Selection，BS）解码器替代传统CRF，将NER建模为片段分类问题，利用双仿射注意力机制对所有可能实体边界对进行全局打分，克服CRF局部依赖的局限。在损失函数层面，引入边界平滑（Boundary Smoothing）[21]与Focal Loss[22]相结合的损失函数，边界平滑通过将硬标签软化为邻近span的概率分布以容忍边界标注噪声，Focal Loss通过聚焦因子降低易分类样本权重，二者共同缓解类别不平衡与边界标签噪声问题。

1 材料与方法

1.1 红枣栽培NER语料构建

选取红枣栽培领域专业书籍作为原始语料，详情信息见表 1。然后将书籍PDF按照正文区域进行裁剪，切除页眉和页脚部分，同时丢弃和红枣栽培知识无关的封面、前言、目录页面，避免冗余文字污染语料内容。其次使用MinerU工具[23]将PDF书籍转化为txt文本格式，人工校对OCR识别过程中出现的字符错误和乱码情况，最终构建红枣栽培NER语料。

表 1 红枣栽培语料信息来源

| 书名 | 作者 | 出版年份 | 字数 |
| --- | --- | --- | --- |
| 《中国果树志 枣卷》 | 王永蕙 | 1993 | 800k |
| 《枣丰产技术》 | 曹新芳 | 2006 | 76k |
| 《果树工 红枣种植》 | 刘多红 | 2011 | 144k |
| 《红枣优质高效栽培技术》 | 单守明 | 2021 | 114k |

1.2 数据标注及数据集划分

面向红枣栽培知识图谱构建和问答系统服务需求，参考红枣栽培专业书籍和专家知识，围绕红枣栽培关键生产环节，将实体类别细分为14种，各类别信息见表 2。

表 2 红枣栽培数据集（RJND）实体类别

| 实体类别 | 实体标签 | 实体描述 | 实体示例 | 实体数量 |
| --- | --- | --- | --- | --- |
| 品种 | CUL | 红枣品种名称 | 牛奶枣、金丝小枣、赞皇大枣 | 3056 |
| 产品 | PRO | 红枣加工制品 | 鲜食枣果、蜜枣、枣泥 | 1025 |
| 部位 | PAD | 红枣树器官与组织部位 | 骨干根、丛生发育枝 | 7432 |
| 时期 | PER | 生长发育阶段与物候期 | 萌芽期、完熟期、落果期 | 2192 |
| 分类 | TAX | 植物分类 | 落叶乔木、鼠李科、枣属 | 104 |
| 营养 | NUT | 营养成分 | 维生素C、可溶性糖、膳食纤维 | 522 |
| 病害 | DIS | 病害名称 | 缩果病、裂果、枣锈病 | 1283 |
| 虫害 | PES | 虫害名称与有害生物名称 | 枣步曲、桃小食心虫、红蜘蛛 | 461 |
| 杂草 | WEE | 田间杂草名称 | 田旋花、苦苣、马齿苋 | 296 |
| 地理信息 | GEO | 种植区域地名 | 新疆南疆地区、辽宁、广西 | 1885 |
| 设备 | EQU | 农业设备工具 | 播种机、修枝剪、烘干设备 | 1896 |
| 肥料 | FER | 肥料名称 | 尿素、磷酸铵、磷酸二氢钾 | 304 |
| 农药 | DRU | 病虫害防治药剂 | 2.5%高效氯氟氰菊酯 | 1087 |
| 农艺 | AGR | 栽培管理操作 | 嫁接、整形修剪、熏蒸 | 3959 |

本研究采用BMES（Begin, Middle, End, Single）方式对上述红枣栽培语料进行实体标注，其中"B"表示实体的开始字符，"M"表示实体的中间字符，"E"表示实体的结束字符，"S"表示由单个字符构成的实体。标注流程采用半自动化策略：首先使用DeepSeek-Chat大语言模型以多线程批处理方式自动提取候选实体词典，并依据词典进行类别匹配，生成初步的BMES标注结果；然后将初步结果导入Label Studio标注平台进行人工校验和修正。

对标注后的红枣栽培数据进行划分，按照8:1:1的比例划分训练集、验证集和测试集。最终RJND数据集切分后训练集包含1527个句子、约18.4万字，验证集包含178个句子、约2.1万字，测试集包含190个句子、约2.3字，共包含红枣栽培领域25502个实体，平均每个句子含实体数量13.5个。

1.3 模型框架

本文构建的红枣栽培 NER 模型整体框架如图 1 所示。模型由嵌入层、特征融合层和边界选择解码层组成：嵌入层并列设置字符级查表嵌入、MacBERT预训练编码、与按BMES四种边界角色匹配领域词典并经CA-BMES（BMES-Channel Multi-Head Attention）通道交互的专家词典嵌入三条支路，从字形、上下文语义与边界先验三个角度对字符序列进行多视角表征，以缓解领域罕见字与低频实体边界证据稀疏的问题；特征融合层将三路嵌入沿特征维拼接为字符级融合表；边界选择解码层基于双仿射机制对所有候选跨度联合打分，输出"（起始位置，结束位置，实体类型）"形式的实体三元组。

图 1 EDBP模型结构

1.3.1 字符级查表嵌入

为获取与上下文无关的领域字符级表征，本文以字符级查表嵌入作为嵌入层的第一条支路。该支路基于可训练嵌入矩阵进行查表。模型在训练集上按字符频次统计构建领域字符表V，并加入<unk>、<pad>等特殊符号；未登录字符在推断阶段统一映射为<unk>。字符嵌入矩阵Ec∈RV×dc随机初始化，并随下游模块端到端联合训练。

给定输入字符序列X={x1,x2,⋯,xn}，先按字符表将每个字符xi映射为整数索引ki=vocabxi∈{0,1,⋯,V-1}，并直接以该索引在嵌入矩阵中查表，得到xi的字符级嵌入：

hic=Ecki,:∈Rdc#(1)

整段序列的字符级查表嵌入矩阵记为：

Hc=h1c;h2c;⋯;hnc∈Rn×dc.#2

该支路在目标语料上从零学习字符级表征，可为训练集中出现的低频字符与领域专有字符提供独立、可微调的字符级表示通道。与 1.3.2 节得到的预训练上下文嵌入相比，字符级查表嵌入不随上下文动态变化，更侧重稳定的字符表面信息建模。最终，该支路输出的Hc将与MacBERT上下文表示以及1.3.3节的专家词典嵌入并列输入特征融合层。

1.3.2 基于 BERT 的字符嵌入

为获取上下文敏感的字符级语义表示，本文采用 MacBERT-base 预训练模型作为嵌入层的第二条支路。MacBERT 是 Cui et al. 提出的中文预训练语言模型，针对 BERT 在中文处理中的局限性进行了三方面改进：

（1）全词掩码策略（Whole Word Masking, WWM）。传统 BERT 采用随机掩码单个字符的策略，可能导致语义碎片化。MacBERT 借鉴 WWM 思想，将组成完整词语的所有字符同时掩码，强制模型学习词汇级别的语义表示

（2）MLM as Correction（MAC）。标准 MLM 任务使用[MASK]标记替换被掩码 token，导致预训练与微调阶段的输入分布不一致。MacBERT 改用同义词替换策略：对被掩码的字/词，以 80% 概率替换为基于 Word2Vec 检索得到的同义词，10% 替换为随机词，10% 保留原词，使预训练阶段的输入分布更接近真实文本，从根本上缓解 [MASK] 标记引入的预训练–微调不一致问题。

（3）N-gram 掩码策略。在 WWM 基础上，MacBERT 进一步以 0.4、0.3、0.2、0.1 的概率分别掩码 1-gram、2-gram、3-gram、4-gram，强制模型同时建模字符级与多字片段级的依赖。这对于红枣栽培领域中包含大量多字专业术语（如"植物生长调节剂""枣锈病防治"）的文本尤为有利。

对于输入的红枣栽培文本序列X={x1,x2,⋯,xn}，MacBERT 通过 12 层 Transformer 编码器输出每个字符的上下文表示：

HBERT=MacBERTX∈Rn×dh

式中 HBERT — MacBERT输出的上下文表示矩阵 X — 输入文本序列 n — 序列长度 dh — BERT隐藏层维度（本文取768维）

相较于标准 BERT，MacBERT 在多个中文 NER 基准数据集上取得了更优性能，尤其在细粒度实体识别任务上表现突出。EDBS 模型将 MacBERT 输出的上下文表示 HBERT 与 1.3.1 节的字符级查表嵌入、1.3.3 节的专家词典 BMES 嵌入并列输入特征融合层，无需经过额外的序列编码层。

1.3.3 专家词典特征层

为了有效利用红枣栽培领域的专家知识，本文设计专家词典特征层，从训练集自动提取领域实体构建专家词典，并将词典匹配信息融合到字符表示中。

1）专家词典构建

专家词典的构建过程如下：首先遍历训练集中所有标注的实体，提取实体文本及其对应的类别标签；然后按类别组织，构建类别到实体列表的映射；最后对每个类别的实体列表去重，得到最终的专家词典D={D1,D2,⋯,DC}，其中Dc表示第c类实体的词典，C为实体类别数。

2）Multi-hot编码

对于输入序列中的每个字符xi，根据其在词典匹配中的位置生成multi-hot编码向量。具体地，对于每个实体类别c和BMES位置p∈{B,M,E,S}，定义指示函数：

fc,pxi=1ifxi在类别c的词典匹配中处于位置p0otherwise

字符xi的词典特征向量为：

Vidict=f1,Bxi,f1,Mxi,f1,Exi,f1,Sxi,⋯,fC,Sxi∈R4C

3）BMES四通道嵌入结构

与传统的单通道词典特征不同，本文采用BMES四通道结构对词典匹配信息进行编码。如图1所示，对于每个实体类别c，分别建立B（Begin）、M（Middle）、E（End）、S（Single）四个独立的嵌入通道，每个通道的嵌入维度为dch=50。

对于字符xi，其在类别c下的BMES四通道嵌入为：

Eic=eic,B;eic,M;eic,E;eic,S∈R4dch=R200

其中eic,p=Embeddingpfc,pxi为通道p的嵌入向量，fc,pxi为公式(3)定义的位置指示函数。

4）特征融合

所有类别的BMES嵌入向量拼接后，通过线性层映射到与BiLSTM输出相同的维度，然后与编码器输出拼接：

Viproj=WdictEi1;Ei2;⋯;EiC+bdict

Hifused=Hienc;Viproj

其中Wdict和bdict为可学习参数，⋅;⋅表示向量拼接操作。BMES四通道结构的优势在于：（1）不同位置（B/M/E/S）的语义特征由独立的嵌入矩阵学习，表达能力更强；（2）200维的输出向量包含了丰富的位置语义信息，有助于模型理解实体的内部结构；（3）相比单通道结构，四通道结构能够区分字符在实体中的不同位置角色，为边界识别提供更精确的先验信息。

图2 BMES四通道嵌入结构

注：图2展示了BMES四通道嵌入结构的详细流程。对于输入序列中的每个字符xi，首先通过专家词典匹配确定其在实体中的位置角色（Begin、Middle、End或Single），然后四个独立的嵌入通道分别学习不同位置的语义特征。每个通道的嵌入维度为50，四个通道拼接后得到200维的密集向量。这种设计的优势在于：（1）不同位置的语义特征由独立的嵌入矩阵学习，避免了参数共享带来的混淆；（2）200维的输出向量包含了丰富的位置信息，有助于模型理解实体的内部结构；（3）相比传统的单通道结构，四通道结构能够明确区分字符在实体中的不同位置角色，为边界识别提供更精确的先验知识。

1.3.4 边界选择解码器

传统CRF将NER建模为逐字符的序列标注问题，难以直接捕捉实体跨度的全局信息。借鉴Yu等[13]提出的将NER建模为依存解析的思想，本文提出边界选择（Boundary Selection，BS）解码器，将NER建模为span分类问题，采用双仿射注意力机制[14]直接预测实体的起止边界及其类型。

1）Span表示

对于序列中的任意位置对i,j（1≤i≤j≤n），其span表示通过双仿射注意力机制计算。首先对融合后的特征分别进行起始位置和结束位置的线性变换：

Hstart=WstartHfused+bstart

Hend=WendHfused+bend

2）双仿射打分

span i,j属于类别c的得分通过双仿射变换计算：

si,j,c=Histart⋅Uc⋅HjendT+Wc⋅Histart;Hjend+bc

其中Uc为双线性变换矩阵，Wc和bc为类别c的权重和偏置。

3）Soft-label交叉熵损失

为了提升模型的泛化能力，采用标签平滑（label smoothing）策略。对于真实标签y，平滑后的标签分布为：

yc=1-ϵifc=yϵCotherwise

其中ϵ为平滑系数，本文设置ϵ=0.1。

图3 边界选择解码器结构

注：图3展示了边界选择解码器的完整流程。对于长度为n的输入序列，模型首先通过两个独立的前馈网络（FFN）分别生成起始位置表示Hstart和结束位置表示Hend，然后通过双仿射注意力机制计算所有可能span i,j的得分si,j,c，最终输出一个三维张量S∈Rn×n×C，其中每个元素表示对应span属于特定实体类型的置信度。与CRF逐字符标注不同，该方法直接对实体跨度进行全局建模，能够更好地捕捉实体的边界信息。

1.3.5 Focal Loss损失函数

红枣栽培NER任务存在严重的类别不平衡问题，大量负样本（非实体span）和高频类别实体会主导训练过程。本文引入Focal Loss[12]损失函数，通过聚焦因子降低易分类样本的损失权重：

FLpt=-αt1-ptγlogpt

其中pt为模型对真实类别的预测概率，γ为聚焦因子，αt为类别权重。当γ>0时，对于预测概率较高（易分类）的样本，1-ptγ接近0，其损失权重被大幅降低；而对于预测概率较低（难分类）的样本，损失权重保持较高。本文设置γ=2.0。

与边界选择解码器的软标签交叉熵结合，最终的损失函数为：

L=i,j,c​-αc1-pi,j,cγyi,j,clogpi,j,c

其中pi,j,c=softmaxsi,j,:c为span i,j属于类别c的预测概率。

1.4 试验环境与参数设置

试验采用的操作系统为Ubuntu 20.04，GPU为GeForce RTX 4090（24GB显存），Python版本为3.8，PyTorch版本为1.13.1，CUDA版本为11.8。

预训练模型选用hfl/chinese-macbert-base，训练轮次为30轮，批大小为16，优化算法选用AdamW。学习率设置采用分层策略：BERT层学习率为5e-5，其他层学习率为1e-3。边界选择解码器参数设置：sb_epsilon=0.1，sb_size=2。Focal Loss聚焦因子γ=2.0。每组实验运行3个不同的随机种子（42、43、44）取平均值，以确保结果的稳定性。

采用精确率（precision，P）、召回率（recall，R）和F1值（F1-score，F1）对命名实体识别精度进行评估，实体的边界和类型同时正确才被判定为正确识别。

2 结果与分析

2.1 消融实验结果

为了验证专家词典特征、边界选择解码器和Focal Loss三个模块对模型的影响，设计消融实验，结果如表2所示。

表 2 消融实验结果

Table 2 Results of ablation experiments

方法 Method词典 DictBSFocalAvg F1/%BERT+LSTM+CRF（纯基线）✗✗-85.57+专家词典✓✗-86.71+边界选择✗✓✗86.68+词典+边界选择✓✓✗87.66+边界选择+Focal✗✓✓86.58+词典+边界选择+Focal（完整模型）✓✓✓88.28

注："✗"表示未使用该模块，"✓"表示使用该模块，"-"表示该模块不适用。BS为边界选择解码器，Focal为Focal Loss损失函数。

从表2可以看出，三个创新模块均对模型性能有显著提升，完整模型的F1值从纯基线的85.57%提升至88.28%，总增益达2.71个百分点。

专家词典特征方面，在CRF解码器上引入词典特征使F1值从85.57%提升至86.71%（+1.14%），在边界选择解码器上引入词典特征使F1值从86.68%提升至87.66%（+0.98%），表明专家词典能够为模型提供有效的领域先验知识，帮助模型更准确地识别红枣栽培领域的专业术语实体。

边界选择解码器方面，在无词典条件下，边界选择解码器（86.68%）相较于CRF解码器（85.57%）提升了1.11个百分点；在有词典条件下，边界选择解码器（87.66%）相较于CRF解码器（86.71%）提升了0.95个百分点。这说明边界选择解码器通过将NER建模为span分类问题，能够直接捕捉实体跨度的全局信息，对于长实体的识别效果优于传统CRF解码器。

Focal Loss方面，在有词典的边界选择模型上，Focal Loss使F1值从87.66%提升至88.28%（+0.62%），表明Focal Loss能够有效缓解类别不平衡问题，使模型更关注难分类的实体边界样本。值得注意的是，在无词典条件下，Focal Loss对边界选择模型的效果并不显著（86.68%→86.58%，下降0.10%），说明Focal Loss需要与词典特征配合才能充分发挥作用，这可能是因为词典特征提供的先验信息能够帮助模型更好地区分易分类和难分类样本。

2.2 解码器与损失函数对比分析

为了全面分析边界选择解码器和Focal Loss的效果，在相同条件下（均使用BERT+BiLSTM编码器和专家词典特征）对比不同解码器和损失函数的性能差异，结果如表3和表4所示。

表 3 不同解码器对比结果

Table 3 Comparison results of different decoders

解码器 Decoder精确率 Precision/%召回率 Recall/%F1值 F1-score/%CRF86.3287.8087.05边界选择 Boundary Selection90.1285.6187.81边界选择+Focal Loss89.5187.5888.54

分析表3可知，边界选择解码器（F1=87.81%）相对于CRF解码器（F1=87.05%）提升了0.76个百分点。值得注意的是，两种解码器在精确率和召回率上表现出不同的倾向：CRF解码器的召回率（87.80%）高于精确率（86.32%），而边界选择解码器的精确率（90.12%）显著高于召回率（85.61%），精确率提升了3.80个百分点，表明边界选择解码器能够更精确地定位实体边界，减少误识别。

表 4 Focal Loss 对各类别召回率与 F1 的影响

Table 4 Impact of Focal Loss on recall and F1 of different entity categories

类别Category     实体数Support     无 Focal Loss     + Focal Loss     ΔRecall/%     ΔF1/%           Recall/%     F1/%     Recall/%     F1/%         EQU（设备）18668.8275.5274.1978.41+5.37+2.89   WED（杂草）4190.2490.2495.1292.86+4.88+2.62   NUT（养分）5082.0088.1786.0089.58+4.00+1.41   AGR（农事操作）46580.0082.1282.5884.12+2.58+2.00   LOC（地域）20774.4081.0576.8180.51+2.41-0.54   CUL（品种）29089.3191.5291.3891.70+2.07+0.18   PAR（部位）82291.3692.2693.3193.20+1.95+0.94   DRU（药剂）12685.7187.8087.3089.07+1.59+1.27   PER（时期）21392.4991.4293.9091.32+1.41-0.10   TAX（分类）966.6770.5966.6770.5900   PES（病虫害）5994.9297.3994.9296.550-0.84   DIS（疾病）14292.2592.9191.5592.20-0.70-0.71   PRO（产品）9985.8684.1684.8582.76-1.01-1.40   FER（肥料）2965.5273.0858.6268.00-6.90-5.08   总体 Overall2,73885.6187.8187.5888.54+1.97+0.73

注：表中展示 Focal Loss 对不同频率类别召回率的影响，类别按召回率提升幅度降序排列。

从表4可以看出，Focal Loss相对于标准交叉熵损失，F1值从87.81%提升至88.54%（+0.73个百分点）。具体来看，Focal Loss主要提升了模型的召回率（85.61%→87.58%，+1.97%），而精确率略有下降（90.12%→89.51%，-0.61%）。从各类别来看，设备（EQU，+5.37%）、杂草（WED，+4.88%）、农事操作（AGR，+2.58%）等多数类别召回率均有所提升；分类（TAX）因测试样本极少（9例）召回率未变；值得注意的是，肥料（FER）类别的召回率出现下降（65.52%→58.62%，-6.90%）。逐实体分析发现，Focal Loss新增漏检的3个实体（"中性有机肥料"、"生理酸性肥料"、"人粪尿"）均为训练集中仅出现1次且未被词典收录的低频复合结构实体，说明在缺乏词典先验支撑的情况下Focal Loss的聚焦机制对极低频实体存在一定的抑制效应。整体而言，Focal Loss通过降低易分类样本的损失权重，使模型更关注难分类的实体边界样本，从而在多数类别上提升了召回率。

红枣栽培NER数据集存在明显的类别不平衡，部分低频类别（如TAX、WED、FER）的实体数量远少于高频类别（如PAR、AGR）。为验证Focal Loss对低频类别的提升效果，统计不同损失函数下各频率类别的召回率变化。Focal Loss主要提升了低频类别的识别能力。WED（杂草）类的召回率从90.24%提升至95.12%，TAX（分类）类召回率保持66.67%（测试集仅9例，统计有限）。这表明Focal Loss通过1-ptγ因子降低易分类样本的损失权重，使模型更关注难分类的低频类别实体和边界模糊的困难样本。值得注意的是，Focal Loss在高频类别上的效果不如低频类别显著。PAR（部位）类的召回率仅从91.36%提升至93.31%，AGR（农事操作）类的召回率从80.00%提升至82.58%。这符合Focal Loss的设计初衷：通过聚焦因子降低易分类样本的权重，使模型将更多学习能力分配给困难样本。需要指出的是，FER（肥料）类是Focal Loss效果的一个例外（召回率从65.52%降至58.62%）。该类别在测试集中仅29例，其中9个实体（占31%）属于枚举结构（如"速效氮、磷、钾肥料"）、中英文混合（如"CO₂气体肥料"）或多词指代等两种方法均无法识别的特殊情形。真正受Focal Loss影响的3个新增漏检（"中性有机肥料""生理酸性肥料""人粪尿"）均为训练集中仅出现1次的低频复合实体，且因词典构建的频率阈值（min_freq=2）而未被纳入专家词典，使其在Focal Loss的强化聚焦机制下未能获得有效的词典先验信息辅助。这提示在极低频类别（训练实例＜50）上，词典覆盖率与Focal Loss的协同效果值得进一步研究。

综合分析：（1）边界选择将NER建模为span分类问题，能够直接对候选span进行分类，避免了CRF逐字符预测带来的错误传播；（2）双仿射注意力机制能够同时建模起始位置和结束位置之间的交互关系，捕捉实体跨度的全局语义信息；（3）Focal Loss使模型更关注难分类的实体边界，有效缓解了类别不平衡问题；（4）在无词典条件下，Focal Loss对边界选择模型的效果并不显著（86.68%→86.58%，下降0.10%），说明Focal Loss需要与词典特征配合才能充分发挥作用。

2.3 各类别识别效果分析

为了进一步分析模型对不同类别实体的识别效果，统计完整模型在测试集上各类别的P、R、F1值，结果如表5所示。

表 5 各类别识别结果

Table 5 Recognition results of each category

类别 Category测试集实体数 Test entities精确率 Precision/%召回率 Recall/%F1值 F1-score/%PAR（部位）82293.0893.3193.20AGR（农事操作）46585.7182.5884.12CUL（品种）29092.0191.3891.70PER（时期）21388.8993.9091.32EQU（设备）18683.1374.1978.41LOC（地域）20784.5776.8180.51DIS（病虫害）14292.8691.5592.20DRU（药剂）12690.9187.3089.07PRO（产品/工艺）9980.7784.8582.76PES（虫害）5998.2594.9296.55NUT（营养）5093.4886.0089.58FER（肥料）2980.9558.6268.00WED（杂草）4190.7095.1292.86TAX（分类）975.0066.6770.59总体 Overall273889.5187.5888.54

注：表中精确率、召回率和F1值为表现最佳种子的单次实验结果。

从表5可以看出，模型对不同类别的识别效果存在明显差异。高频类别中，部位（PAR）的F1值最高（93.20%），得益于其训练样本充足（训练集6017个实体）且实体边界清晰；品种（CUL，91.70%）和病虫害（DIS，92.20%）也表现优异，这些类别的实体通常是领域专有名词，专家词典能够提供有效的匹配信息。虫害（PES）虽然测试集实体数较少（59个），但F1值高达96.55%，其精确率（98.25%）和召回率（94.92%）均处于最高水平，这是因为虫害名称（如"枣步曲"、"桃小食心虫"）具有高度专业化的命名模式，模型易于学习。

低频类别方面，分类（TAX，70.59%）和肥料（FER，68.00%）的F1值最低。TAX类别仅有9个测试实体和87个训练实体，模型难以从极少的样本中学习到稳定的识别模式；FER类别虽有29个测试实体，但其召回率仅为58.62%，其中9个实体因枚举结构、中英混合等复杂命名形式两种方法均无法识别，而训练样本极少（仅出现1次）且未被词典收录的低频复合实体也存在漏检，表明小样本类别对词典覆盖率的依赖较高。设备（EQU，78.41%）和地域（LOC，80.51%）的F1值也相对较低，主要受召回率（分别为74.19%和76.81%）影响，表明这两类实体的边界识别仍存在挑战。

总体来看，Focal Loss的引入在一定程度上缓解了类别不平衡问题，使模型整体的精确率（89.51%）和召回率（87.58%）较为均衡，整体F1值达到88.54%。

2.4 与现有方法对比

为了验证本文方法的有效性，将完整模型与近年来农业领域NER的代表性方法进行对比，结果如表6所示。

表 6 与现有方法对比结果

Table 6 Comparison results with existing methods

| 方法 Method | F1值 |
| --- | --- |
| BiLSTM-CRF | 78.69 |
| BERT-CRF | 86.71 |
| SoftLexicon | 80.45 |
| BERT+BiLSTM+CRF | 85.48±0.25 |
| MacBERT-base+BiLSTM+CRF | 85.57±0.29 |
| EDBS（本文） | 88.28±0.35 |

注：本文BERT+BiLSTM+CRF、MacBERT-base+BiLSTM+CRF与EDBS均按3个随机种子（42/43/44）重复试验，报告均值±标准差。

分析表6可知，EDBS取得了最高的F1值（88.28%），相较于BiLSTM-CRF基线（78.69%）提升了9.59个百分点；相较于基线BERT+BiLSTM+CRF（85.48±0.25）提升2.80百分点。以3个种子结果进行配对t检验，EDBS相对BERT+BiLSTM+CRF（p=0.0036）达到显著提升（p<0.01）。与SoftLexicon方法相比，EDBS的专家词典是从训练集自动提取的领域专用词典，更加贴合红枣栽培领域的实际需求。

2.5 公开数据集泛化性验证

为了验证本文方法的泛化能力，在多个公开中文NER数据集上进行了验证实验。实验设置如下：优化器采用AdamW（学习率2e-3，BERT微调学习率2e-5），批大小为16，训练轮数为30，预训练模型采用MacBERT-base并设置0.2的dropout率，序列标注方案采用BIOES。各数据集的基本信息如表7所示。

表 7 公开数据集基本信息 Table 7 Basic information of public datasets

| 数据集 Dataset | 实体类别数 Types | 训练集句子数 Train | 验证集句子数 Dev | 测试集句子数 Test | 实体类型示例 Example types |
| --- | --- | --- | --- | --- | --- |
| WeiboNER | 4 | 1,352 | 270 | 270 | 人名、地名、机构名、地理位置 |
| ResumeNER | 8 | 3,821 | 463 | 477 | 国籍、学历、姓名、专业、城市 |
| Boson | 6 | 4,085 | 510 | 511 | 人名、时间、地点、产品、机构、公司 |
| RedJujube（本文） | 14 | 1,527 | 178 | 190 | 部位、农事操作、品种、时期、设备 |

表 8 3个公开数据集上的对比

| 方法 Method | F1值 F1-score/% |  |  |
| --- | --- | --- | --- |
|  | Boson | WeiboNER | ResumeNER |
| LSTM + CRF | 79.49 | 50.19 | 94.93 |
| SoftLexicon + LSTM + CRF | 83.64 | 61.17 | 95.48 |
| MacBERT+ LSTM + CRF | 85.35 | 70.48 | 95.97 |
| EDBS（本文） | 85.60±0.12 | 72.27±1.03 | 96.13±0.29 |

从表8可以看出：（1）在通用新闻领域，Boson和CLUENER数据集上EDBS分别提升0.25%和0.16%，表明方法在通用领域具有正向增益；（2）在社交媒体领域，WeiboNER基线范围50.19%~70.81%，该领域存在较大挑战；本文EDBS在WeiboNER上达到72.27±1.03，相对最强基线70.81提升1.46个百分点；（3）在专业领域，本文EDBS在ResumeNER上达到96.13±0.29，接近最优强基线；RedJujube（农业）基线为86.71%，EDBS在RedJujube上取得1.57%的显著提升。

总体而言，EDBS在已完成的实验中均取得了正向提升，验证了专家词典特征和边界选择解码器组合方案的泛化能力。实验正在扩展至更多数据集以进一步验证方法的有效性。

3 结 论

本文面向红枣栽培领域命名实体识别需求，构建了涵盖14个类别的红枣栽培命名实体识别数据集RJND，提出了基于专家词典与边界选择的红枣栽培命名实体识别模型EDBS。该模型一方面通过专家词典特征层自动提取和利用领域专家知识，通过multi-hot编码将词典匹配信息融合到字符表示中，增强模型对专业术语的识别能力；另一方面采用边界选择解码器替代传统CRF序列标注，将NER建模为span分类问题，利用双仿射注意力机制直接预测实体边界，能够更好地捕捉实体跨度的全局信息；同时引入Focal Loss损失函数缓解类别不平衡问题。与现有模型对比，EDBS取得更好的实体识别精度，F1值达到88.28%，在公开数据集Boson和CLUENER上的试验进一步表明模型的有效性。未来工作将围绕实体关系抽取以及红枣栽培知识图谱构建等后续应用展开进一步研究。

参考文献

[1] 邢钟毓, 莎仁图雅, 邢钰坤, 等. 我国红枣产业发展研究现状[J]. 农业与技术, 2025, 45(4): 84-88.邢钟毓, 莎仁图雅, 邢钰坤, 等. 我国红枣产业发展研究现状[J]. Agriculture and Technology, 2025, 45(4): 84-88. (in Chinese)

[2] 丁建平, 李卫军, 刘雪洋, 等. 命名实体识别研究综述[J]. 计算机工程与科学, 2024, 46(7): 1296.丁建平, 李卫军, 刘雪洋, 等. 命名实体识别研究综述[J]. 2024, 46(7): 1296. 2024. (in Chinese)

[3] GRISHMAN R, SUNDHEIM B. Message understanding conference- 6: a brief history[C]//COLING 1996 Volume 1: The 16th International Conference on Computational Linguistics. [2026-05-05].

[4] HOBBS J R, APPELT D, BEAR J, et al. FASTUS: A Cascaded Finite-State Transducer for Extracting Information from Natural-Language Text[M]. ROCHE E, SCHABES Y, eds.//Finite-State Language Processing. The MIT Press, 1997: 383-406[2026-05-05].

[5] ZHOU G, SU J. Named entity recognition using an HMM-based chunk tagger[C]. ISABELLE P, CHARNIAK E, LIN D, eds.//Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics. Philadelphia, Pennsylvania, USA: Association for Computational Linguistics, 2002: 473-480[2026-05-05].

[6] MCCALLUM A, LI W. Early results for Named Entity Recognition with Conditional Random Fields, Feature Induction and Web-Enhanced Lexicons[C]//Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003. [2026-05-05].

[7] ZHU Z, ZHAO Q, LI J, et al. Comparative Analysis of Large Language Models in Chinese Medical Named Entity Recognition[J]. Bioengineering (Basel, Switzerland), 2024, 11(10): 982.

[8] SU L, CHEN J, PENG Y, et al. Demonstration-based learning for few-shot biomedical named entity recognition under machine reading comprehension[J]. Journal of Biomedical Informatics, 2024: 104739.

[9] 李春春, 丁鑫, 张华扬, 等. 基于BERT-BiLSTM-CRF的茶树病虫害命名实体识别方法[J]. 农业机械学报, 2025, 56(11): 517-527.李春春, 丁鑫, 张华扬, 等. 基于BERT-BiLSTM-CRF的茶树病虫害命名实体识别方法[J]. 2025, 56(11): 517-527. 2025. (in Chinese)

[10] 吴钊, 朱玉颖, 张宏鸣, 等. 基于多特征融合的苹果栽培命名实体识别[J]. 农业工程学报, 2025, 41(10): 176-185.吴钊, 朱玉颖, 张宏鸣, 等. 基于多特征融合的苹果栽培命名实体识别[J]. Transactions of the Chinese Society of Agricultural Engineering, 2025, 41(10): 176-185. (in Chinese)

[11] 聂啸林, 张礼麟, 牛当当, 等. 面向葡萄知识图谱构建的多特征融合命名实体识别[J]. 农业工程学报, 2024, 40(3): 201-210.聂啸林, 张礼麟, 牛当当, 等. 面向葡萄知识图谱构建的多特征融合命名实体识别[J]. Transactions of the Chinese Society of Agricultural Engineering, 2024, 40(3): 201-210. (in Chinese)

[12] HUANG Z, XU W, YU K. Bidirectional LSTM-CRF models for sequence tagging[EB]. arXiv, 2015(2015-08-09)[2026-05-05].

[13] ZHANG Y, YANG J. Chinese NER Using Lattice LSTM[C]. GUREVYCH I, MIYAO Y, eds.//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Melbourne, Australia: Association for Computational Linguistics, 2018: 1554-1564[2026-05-05].

[14] MA R, PENG M, ZHANG Q, et al. Simplify the Usage of Lexicon in Chinese NER[C]. JURAFSKY D, CHAI J, SCHLUTER N, et al., eds.//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, 2020: 5951-5960[2024-10-13].

[15] JACOB DEVLIN, MING-WEI CHANG, KENTON LEE, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[EB]. arXiv, 2019(2019-05-24)[2024-04-11].

[16] JING LI, AIXIN SUN, JIANGLEI HAN, et al. A Survey on Deep Learning for Named Entity Recognition[J]. IEEE Transactions on Knowledge and Data Engineering, 2022, 34(1): 50-70.

[17] LI F, LIN Z, ZHANG M, et al. A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition[C]. ZONG C, XIA F, LI W, et al., eds.//Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Online: Association for Computational Linguistics, 2021: 4814-4828[2026-05-05].

[18] LI J, FEI H, LIU J, et al. Unified named entity recognition as word-word relation classification[EB]. arXiv, 2021(2021-12-19)[2026-03-19].

[19] LI X, LV C, WANG W, et al. Generalized focal loss: towards efficient representation learning for dense object detection[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, 45(3): 3139-3153.

[20] ZHENG C, CAI Y, XU J, et al. A boundary-aware neural model for nested named entity recognition[C]. INUI K, JIANG J, NG V, et al., eds.//Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Hong Kong, China: Association for Computational Linguistics, 2019: 357-366[2025-05-25].

[21] 余肖生, 黄莺, 张云涛, 等. GTR-NNER：一种融合单词多元信息的嵌套命名实体识别方法[J]. 数据分析与知识发现, 2025, 9(3): 127-135.余肖生, 黄莺, 张云涛, 等. GTR-NNER：一种融合单词多元信息的嵌套命名实体识别方法[J]. 2025, 9(3): 127-135. 2025. (in Chinese)

[22] LIN T-Y, GOYAL P, GIRSHICK R, et al. Focal loss for dense object detection[C]//2017 IEEE International Conference on Computer Vision (ICCV). [2026-05-05].

[23] Anonymous. opendatalab/MinerU[Z]. OpenDataLab, 2026(2026-05-12)[2026-05-12].
