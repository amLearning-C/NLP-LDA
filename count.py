# -*- coding: utf-8 -*-
import jieba
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
from gensim import corpora, models
num = 000

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体，这里是宋体


# 预处理函数：删除隐藏符号、非中文字符和标点符号
def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # 删除停用词
    with open('cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # 删除所有的标点符号
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text


folder_path = r"D:/Users/Vincent W/Desktop/研究生/学习/研一下NIP/测试用/"

# 初始化数据列表
data_words = []
data_chars = []

# 初始化总段落数
total_paragraphs = 500

# 计算每个文件中应该抽取的段落数
file_paragraphs = {}

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):  # 只处理以 .txt 结尾的文件
        file_path = os.path.join(folder_path, file_name)
        print("处理文件:", file_path)

        # 读取文本内容并进行预处理和分词
        with open(file_path, "r", encoding='ansi') as file:
            text = file.read()
            preprocessed_text = preprocess_text(text)
            words = jieba.lcut(preprocessed_text)
            chars = list(preprocessed_text)
            file_paragraphs[file_name] = len(words) // num  # 每500个词为一个段落

# 确定每个文件中应该抽取的段落数
total_file_paragraphs = sum(file_paragraphs.values())
for file_name, paragraphs_to_extract in file_paragraphs.items():
    file_ratio = paragraphs_to_extract / total_file_paragraphs
    file_paragraphs[file_name] = int(file_ratio * total_paragraphs)

# 确保总段落数为1000个
current_total_paragraphs = sum(file_paragraphs.values())
if current_total_paragraphs < total_paragraphs:
    remaining_paragraphs = total_paragraphs - current_total_paragraphs
    if "鹿鼎记.txt" in file_paragraphs:
        file_paragraphs["鹿鼎记.txt"] += remaining_paragraphs
    else:
        print("文件夹中缺少鹿鼎记.txt，无法补充段落。")

# 抽取每个文件中的段落，并存储到数据列表中
for file_name, paragraphs_to_extract in file_paragraphs.items():
    file_path = os.path.join(folder_path, file_name)
    print("抽取文件:", file_name)
    # 读取文本内容并进行预处理和分词
    with open(file_path, "r", encoding='ansi') as file:
        text = file.read()
        preprocessed_text = preprocess_text(text)
        words = jieba.lcut(preprocessed_text)
        chars = list(preprocessed_text)

        # 随机抽取段落并添加到数据列表中
        sampled_indices = random.sample(range(len(words) // num), paragraphs_to_extract)
        for i in sampled_indices:
            start_index = i * num
            end_index = start_index + num
            paragraph_words = words[start_index:end_index]
            paragraph_chars = chars[start_index:end_index]
            # 将词列表连接成字符串，并以逗号分隔
            data_words.append({'段落': ', '.join(paragraph_words), '标签': file_name})
            data_chars.append({'段落': ', '.join(paragraph_chars), '标签': file_name})
#print(data_words)
# 将数据列表转换为 DataFrame
df_words = pd.DataFrame(data_words)
df_chars = pd.DataFrame(data_chars)

# 将 DataFrame 写入 CSV 文件
words_csv_path = "words_data.csv"
chars_csv_path = "chars_data.csv"

df_words.to_csv(words_csv_path, index=False)
df_chars.to_csv(chars_csv_path, index=False)
print("数据已写入 CSV 文件")




