# 绿色印刷产业发展情况研究

大家好，我们是上海理工大学印刷工程专业的大三学生。这是一个用来存储研究绿色印刷产业发展项目相关文件的一个仓库。

我们使用Python3来对爬取绿色印刷相关原始文本，对文本预处理，计算TF-IDF值，并使用K-means方法进行了聚类操作，使用了PCA和TSNE降维方法，输出并可视化了结果，用于后续的研究分析。

主要使用了Jieba中文分词组件和Scikit-learn机器学习等模块来进行一些文本处理操作。

自己还在学习NLP的过程中，还有许多代码不完善不成熟的地方，需要多多学习！！！

本仓库目前包括:

文件名称|含义|相关备注
--|:--:|--:
**chineseStopWords.txt**|中文停用词|额外添加了部分地名和人名 UTF-8格式
**cluster_analysis.py**|聚类主文件|Kmeans和可视化
**corpus.txt**|过滤后的语料文件|用于检查
**printing_dict.txt**|自定义词典|额外添加印刷相关术语 UTF-8格式
**sliced_text**|按时间切分的文本|用于分析发展情况相关图片
**full_text.txt** |全文本|必须为UTF-8格式
**ex_text.csv** |补充的相关文本数据|添加时间，标题，关键词，文本多个信息

---
CCCCaO 2019.3