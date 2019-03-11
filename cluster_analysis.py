# encoding:utf-8
# Author: Sicheng Gu
# Data:2018.11-2019.3
# 主要用于通过文本探索研究来反应绿色印刷发展情况的项目
# 自己还在学习NLP的过程中，使用kmeans对TFIDF来聚类并且进行了可视化处理，还需要进一步学习！


import jieba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def csv_preproc(ori_csv_dir, proc_csv_dir):
    """
    csv_preproc函数 用于对csv文件进行预处理操作：去除重复行，并按年份时间升序排列
    并将处理后的csv文件输出，可以用于检查
    csv文件的格式为：首行是列索引Date,Title,Keywords,Abstract
    每一行分别为一片文章的发表日期，标题，关键词，摘要或全文

    参数：
        ori_csv_dir     原始csv文件的路径
        proc_csv_dir    输出处理后的csv文件的路径
    
    返回值：
        无
    """
    try:
        csv_data = pd.read_csv(ori_csv_dir)
    except IOError:
        print("csv数据加载失败！请检查目录！")
    else:
        print("原始数据尺寸为：{}".format(csv_data.shape))
        # 在原地去重
        csv_data.drop_duplicates(subset=['Abstract'], inplace=True)
        print("去重后数据尺寸为：{}".format(csv_data.shape))
        # 按照年份排序
        csv_data_distinct_ordered = csv_data.sort_values(by='Date')
        # 输出新的csv文件
        csv_data_distinct_ordered.to_csv(proc_csv_dir, index=False)
        print('去除重复且按时间升序排列的csv文件输出完成！')
        print("---")


def csv_slice(proc_csv_dir, tar_csv_dir, start_year=1500, end_year=2019):
    """
    csv_slice函数 用于从去重且排序好的csv文件中 切出从start年份至end年的数据 用于分析
    
    参数：
        proc_csv_dir    去重且排序好的csv文件路径
        tar_csv_dir     选取某年到某年的csv文件并输出的路径
        start_year      起始年份 默认1500
        end_year        末尾年份 默认为2019

    返回值：
        无

    参考：
        使用Pandas对数据进行筛选和排序 http://bluewhale.cc/2016-08-06/use-pandas-filter-and-sort.html
    """
    try:
        csv_data = pd.read_csv(proc_csv_dir)
    except IOError:
        print("处理后的csv数据加载失败！请检查目录！")
    else:
        print("读取数据尺寸为：{}".format(csv_data.shape))
        csv_xtoy = csv_data.loc[(csv_data['Date'] >= start_year) & (csv_data['Date'] <= end_year)]
        print("{}年到{}年 切片后的数据尺寸为：{}".format(start_year, end_year, csv_xtoy.shape))
        # 将结果输出至csv文件 不要列索引输出header取None 不要行索引index取False
        csv_xtoy[['Keywords', 'Abstract']].to_csv(tar_csv_dir, header=None, index=False)
        print("---")


def text_load(userdict_dir, text_dir):
    """
    text_Load函数：用于文本相关预处理操作
    本函数包括 读取用户自定义词典和待处理文本 以及 使用jieba中文分词组件进行文本切分
    返回分词好的语料 其中每个词用空格进行分隔 

    参数：
        userdict_dir    str类型    自定义词典的绝对路径 自定义词典应为uft-8编码的txt文件
        text_dir        str类型    待处理文本的绝对路径 待处理文本应为uft-8编码的txt或csv文件
    
    返回值：
        corpus_ori      list类型   切分好的原始语料列表
    """
    # 注：try关键词用的还不是很会 不清楚这么用对不对...
    try:
        # 加载用户自定义词典以提高切分的准确程度 
        jieba.load_userdict(userdict_dir)
    except IOError:
        print("Userdict File Input Error!!!")
    else:
        print("Sucessfully load the Userdict...")
        # 定义一个匿名函数segment_jieba 输入text文本 用空格作为分隔符来分隔切分好的文本
        segment_jieba = lambda text:" ".join(jieba.cut(text, cut_all=False))
        corpus_ori = []
        try:
            # 以utf-8编码打开待处理文本 txt或csv都可以
            with open(text_dir, 'r', encoding='utf-8') as f:
                # 文本格式是每一行一篇关键词+文章 如果文本格式改变的话 修改这部分即可！
                for line in f:
                    if line != ' ':
                        corpus_ori.append(segment_jieba(line.strip()))
        except IOError:
            print("Text File Input Error!!!")
        else:
            print("Cut complete...")
    print("---")
    return corpus_ori


def words_filt(stopwords_dir, corpus_dir, corpus_ori):
    """
    words_filt函数：用于过滤停用词
    读取停用词表文件 遍历切分好的语料 去除停用词表中出现的词
    返回过滤后的语料 并且将其输出至uft-8编码的txt文件用于检查

    参数：
        stopwords_dir   str类型     停用词表的绝对路径
        corpus_dir      str类型     输出语料库文件的绝对路径
        corpus_ori      list类型    切分好的原始语料文件，由text_load函数加载切分

    返回值：
        corpus          list类型    切分并过滤过停用词之后的语料
    """
    corpus = []
    word = ''
    new_line = ''
    flag = 0
    try:
        # 加载停用词表文件
        stopwords = [line.strip() for line in open(stopwords_dir, 'r', encoding='utf-8').readlines()]
        #print(stopwords[-1:])
    except IOError:
        print("Can't load Stopwords!!! ")
    else:
        # 因为切分好的语料的格式是一行是一个词
        # 所以这里先读取一行 然后读取每一个字母 合并成一个词 然后再去过滤
        for line in corpus_ori:
            for letter in line:
                if letter != ' ':
                    word += letter
                else:
                    if word in stopwords:
                        flag += 1
                    else:
                        if word != '\t':
                            new_line = new_line + word + ' '
                    word = ''
            corpus.append(new_line)
            new_line = ''
        # 将过滤后的语料输出至txt文件
        try:
            with open(corpus_dir, 'w',encoding='utf-8') as b:
                for line in corpus:
                    b.write(line)
        except IOError:
            print("Can't output the filterd corpus to txt!!!")
        else:
            print("Filtered corpus file saved...")    
            # 统计一下去除了多少停用词
            print("{} words are filtered".format(flag))
            print("Filtering complete...")
    print("---")
    return corpus


def tfidf(corpus):
    """
    tfidf函数 
    使用sklearn中的两个类 计算文本TF-IDF值以备后续使用
    参考：“文本数据预处理” https://blog.csdn.net/m0_37324740/article/details/79411651

    参数：
        corpus    list类型  切分且过滤好的语料
    
    返回值：
        tf-idf_weight   词语TF-IDF值的矩阵
        word            所有文本切分后的词
    """
    # 先使用CountVectorizer计算词频矩阵 然后使用TfidfTransformer对词频矩阵中每个词统计TF-IDF值
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋中所有文本切分后的所有关键词
    word = vectorizer.get_feature_names()
    print("Word feature length:{}".format(len(word)))
    print("TF-IDF array get...")
    # tfidf转换成array
    tfidf_weight = tfidf.toarray()
    print("---")

    return tfidf_weight, word


def kmeans_vis(n_clusters, tfidf_weight, word, topn_features=5, decomposition='PCA'):
    """
    kmeans_vis_TSNE函数 
    使用kmeans聚类对tf-idf矩阵进行聚类操作 
    使用TSNE降低到二维绘制散点图
    
    参数：
        n_clusters      int         聚类的类数（即k值）
        tfidf_weight    ndarray     文本TFIDF矩阵
        word            list        词袋模型中所有的词语
        topn_features   int         前N项关键特征词 默认是5
        decomposition   str         降维的方法 可选PCA和TSNE 默认PCA

    返回值：
        TSNE或PCA降维的聚类散点图

    参考:
        Sklearn的KMeans参数介绍 https://blog.csdn.net/github_39261590/article/details/76910689
        主成分分析PCA降维的运用实战 https://blog.csdn.net/brucewong0516/article/details/78666763
        流形学习-高维数据的降维与可视化 https://blog.csdn.net/u012162613/article/details/45920827    
    """

    kmeans = KMeans(n_clusters)
    kmeans.fit(tfidf_weight)
    # 打印簇的中心坐标
    #print(kmeans.cluster_centers_)
    # 将文章索引 和 其类标签输出至控制台
    for index, label in enumerate(kmeans.labels_, 1):
        print("index: {}, label: {}".format(index, label))
    # 打印inertia 
    print("inertia:{}".format(kmeans.inertia_))
    # 打印关键特征词
    ordered_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    for cluster_num in range(n_clusters):
        key_features = [word[index]
                            for index
                            in ordered_centroids[cluster_num, :topn_features]]
        print(key_features)
    
    # 可视化部分
    # 降维选择
    if(decomposition=='TSNE'):
        # 使用TSNE对TF-IDF进行降维 降低至2维
        tsne = TSNE(n_components=2, n_iter=1600, learning_rate=100)
        decomposition_data = tsne.fit_transform(tfidf_weight)
    elif(decomposition=='PCA'):
        # 使用PCA对TF-IDF进行降维 降低至2维
        pca = PCA(n_components=2)
        decomposition_data = pca.fit_transform(tfidf_weight)
    # 绘制
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    # 画虚线方格
    plt.grid(ls='--')
    # 用o表示某篇文章 坐标在拆分的数据中
    plt.scatter(decomposition_data[:, 0], decomposition_data[:, 1], c=kmeans.labels_+1, marker="o", alpha=0.8)
    print(type(kmeans.labels_))
    ax.set_title(u'Green Printing  K-means clustering')
    plt.colorbar()
    plt.show()
    plt.savefig('./sample.png', aspect=1)
    print(type(kmeans.labels_))
    
    
def best_k(tfidf_weight):
    """
    best_k函数  使用手肘法来寻找最佳K值
    绘制不同k和相应SSE(误差平方和）的函数图像来观察 其中SSE代表所有样本的聚合误差
    手肘法的核心思想是随着聚类k数值增大，样本更精细，每个簇聚合程度会变高，SSE会变小
    当k小于真实聚类数时，由于k增大会大幅度增加每个簇的聚合程度，所以SSE下降幅度很大；
    当k接近真实聚类数时，再增加k值，其每个簇聚合程度会变小，所以SSE下降幅度会骤减
    如果绘制一个k和SSE的关系图时，会形成一个手肘的形状，这个肘部对应的k值是最好的k值
    参考：
        K-means聚类最优k值的选取 https://blog.csdn.net/qq_15738501/article/details/79036255

    参数：
        tfidf_weight    tfidf值的矩阵
    
    返回值：
        K-SSE图像
    """
    SSE = []
    for k in range(1,10):
        estimator = KMeans(n_clusters=k)
        estimator.fit(tfidf_weight)
        SSE.append(estimator.inertia_)
    # 绘图
    X = range(1, 10)
    plt.xlabel('K')
    plt.ylabel('SSE(Sum of the Squared Errors)')
    plt.plot(X, SSE, 'o-')
    plt.show()
    print("K-SSE figure ... ok!")


def dbscan_vis(tfidf_weight, decomposition='PCA'):
    """
    尝试使用DBSCAN(基于密度的带有噪声的空间聚类法)来聚类
    """
    if(decomposition=='TSNE'):
        # 使用TSNE对TF-IDF进行降维 降低至2维
        tsne = TSNE(n_components=2, n_iter=2000, learning_rate=200)
        decomposition_data = tsne.fit_transform(tfidf_weight)
    elif(decomposition=='PCA'):
        # 使用PCA对TF-IDF进行降维 降低至2维
        pca = PCA(n_components=2)
        decomposition_data = pca.fit_transform(tfidf_weight)
    
    X = StandardScaler().fit_transform(decomposition_data)
    #print(type(X))
    #X = np.concatenate([x, y], axis=0)
    db = DBSCAN(eps=0.35, min_samples=10).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=db)
    plt.show()
    

if __name__ == '__main__':
    # 这里应该使用一个demo文件 作为模块的测试用
    # 而真正的运行文件可以另外写一个py文件

    # 自定义词典路径 额外添加了很多印刷相关的术语
    userdict_dir = "C:\\Users\\82460\\Documents\\GitHub\\green_printing\\printing_dict.txt"
    # 停用词表路径 在搜集自网络的停用词表基础上 根据实际结果 额外添加了许多人名与地名
    stopwords_dir = "C:\\Users\\82460\\Documents\\GitHub\\green_printing\\chineseStopWords.txt"
    # 切分好的词语输出的路径
    corpus_dir = "C:\\Users\\82460\\Documents\\GitHub\\green_printing\\corpus.txt"
    
    
    # csv文件的预处理和加载
    ori_csv_dir = 'C:\\Users\\82460\\Documents\\GitHub\\green_printing\\ex_text.csv'
    proc_csv_dir = 'C:\\Users\\82460\\Documents\\GitHub\\green_printing\\sliced_text\\proc_text.csv'
    csv_preproc(ori_csv_dir, proc_csv_dir)
    # 选取某时间到某时间
    #tar_csv_dir = 'C:\\Users\\82460\\Documents\\GitHub\\green_printing\\sliced_text\\fulltext.csv'
    #csv_slice(proc_csv_dir, tar_csv_dir)
    
    
    #----------------------------------------------------------------------------#
    # 加载文件 仍然在修改中...
    # 全文分析
    #text_dir = "C:\\Users\\82460\\Documents\\GitHub\\green_printing\\full_text.txt"
    # 按年份分析
    #for year in range(2015, 2020):
        #text_dir = "C:\\Users\\82460\\Documents\\GitHub\\green_printing\\sliced_text\\text"+ str(year) + ".txt"
        #print(text_dir)
    text_dir = "C:\\Users\\82460\\Documents\\GitHub\\green_printing\\full_text.txt"
    #----------------------------------------------------------------------------#
    # 加载文本
    corpus = text_load(userdict_dir, text_dir)
    # 停用词过滤
    corpus1 = words_filt(stopwords_dir, corpus_dir, corpus)
    # 获取tfidf值和所有关键词
    tfidf_weight, word = tfidf(corpus)
    # 绘制聚类结果图
    #kmeans_vis(8, tfidf_weight, word, decomposition='TSNE')
    # 绘制K-SSE关系图
    #best_k(tfidf_weight)
    # 使用DBSCAN聚类
    #dbscan_vis(tfidf_weight, 'PCA')