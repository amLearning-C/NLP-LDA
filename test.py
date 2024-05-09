
from gensim import corpora, models
import pandas as pd
import numpy as np
import os
import codecs
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# 训练LDA模型的方法
# 训练LDA模型的方法
def lda_train(num_topics, data_file, middatafolder, iterations=5000, passes=1, workers=1):
    time1 = datetime.now()
    num_docs = 0
    doclist = []
    with codecs.open(data_file, 'r', 'utf-8') as source_file:
        for line in source_file:
            num_docs += 1
            if num_docs % 100000 == 0:
                print('Processed %d documents' % num_docs)
            doclist.append(line.strip().split(','))

    dictionary = corpora.Dictionary(doclist)
    dictionary.save(os.path.join(middatafolder, 'dictionary.dictionary'))
    corpus = [dictionary.doc2bow(doc) for doc in doclist]
    corpora.MmCorpus.serialize(os.path.join(middatafolder, 'corpus.mm'), corpus)

    time2 = datetime.now()

    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics, iterations=iterations, passes=passes)
    lda_model.print_topics(num_topics, 30)
    print('lda training time cost is : %s, all time cost is : %s ' % (datetime.now() - time2, datetime.now() - time1))
    lda_model.save(os.path.join(middatafolder, 'lda_tfidf_%s_%s.model' % (num_topics, iterations)))

    return lda_model, dictionary


# 分类模型训练和预测
def train_and_predict(X, y, model='rf', n_neighbors=5):
    # # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if model == 'rf':
        # 创建随机森林模型
        classifier = RandomForestClassifier(random_state=42)
    elif model == 'knn':
        # 创建KNN模型
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError("Invalid model type. Use 'rf' for Random Forest or 'knn' for KNN.")

        # 使用10折交叉验证
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')

    # 输出交叉验证结果
    print("Cross-Validation Accuracy Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))

    return classifier

# 获取主题分布的平均值，使得每个主题分布长度相同
def get_average_topic_distribution(topic_distributions, num_topics):
    average_topic_distribution = []
    for distribution in topic_distributions:
        avg_distribution = [0] * num_topics
        for topic, weight in distribution:
            avg_distribution[topic] = weight
        average_topic_distribution.append(avg_distribution)
    return average_topic_distribution

# 使用LDA模型将每个段落表示为主题分布的方法
def get_topic_distributions(lda_model, doclist):
    topic_distributions = []
    for doc in doclist:
        doc_bow = lda_model.id2word.doc2bow(doc)
        topic_distribution = lda_model[doc_bow]
        topic_distributions.append(topic_distribution)
    return topic_distributions
# 打印每个主题的主题词
def print_topic_words(lda_model, num_words=10):
    for i in range(lda_model.num_topics):
        topic_words = lda_model.print_topic(i, topn=num_words)
        print(f"主题 {i+1} 的主题词：{topic_words}")


def compare_lda_rf(num_topics_list, data_file, middatafolder, iterations=5000, passes=1, workers=1):
    for num_topics in num_topics_list:
        print(f"\nExperimenting with {num_topics} topics:")
        # 训练LDA模型
        lda_model, dictionary = lda_train(num_topics, data_file, middatafolder, iterations, passes, workers)

        # 加载数据并将每个段落表示为词列表
        data_words = pd.read_csv(data_file)
        doclist = [doc.split(',') for doc in data_words['段落']]

        # 使用训练好的LDA模型获取每个段落的主题分布
        topic_distributions = get_topic_distributions(lda_model, doclist)

        # 获取主题分布的平均值，使得每个主题分布长度相同
        X = get_average_topic_distribution(topic_distributions, num_topics)

        # 获取标签
        y = data_words['标签']

        # 训练和预测分类模型（使用随机森林）
        print("Training and evaluating Random Forest model...")
        train_and_predict(X, y, model='rf')

# 调用训练方法
num_topics = 50
data_file = "chars_data.csv"
middatafolder = r"D:/Users/Vincent W/Desktop/研究生/学习/研一下NIP/作业2/"
iterations = 5000
passes = 1
workers = 1


# 加载数据并将每个段落表示为词列表
data_words = pd.read_csv(data_file)
doclist = [doc.split(',') for doc in data_words['段落']]


# 获取标签
data = pd.read_csv(data_file)
y = data['标签']


# 定义不同的主题数量列表
num_topics_list = [10,20,50,100]

# 执行对比实验
compare_lda_rf(num_topics_list, data_file, middatafolder)