import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
from tqdm.auto import tqdm

# 下载必要的nltk数据
nltk.download('punkt')


def preprocess_text(text):
    """文本预处理函数，先兜底保证是字符串"""
    import math

    # 处理 None / NaN / float 等情况，统一转成干净的字符串
    if text is None:
        text = ""
    elif isinstance(text, float):
        if math.isnan(text):
            text = ""
        else:
            text = str(text)
    else:
        text = str(text)

    # 转换为小写
    text = text.lower()
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词
    tokens = word_tokenize(text)
    return tokens


def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 合并标题和评论
    df['text'] = df.iloc[:, 1].astype(str).fillna('') + " " + df.iloc[:, 2].astype(str).fillna('')

    # 预处理所有文本，增加进度条显示
    corpus = []
    for text in tqdm(df['text'], desc="构建语料库 (分词)"):
        tokens = preprocess_text(text)
        corpus.append(tokens)

    return corpus, df.iloc[:, 0].values, df  # 返回处理后的文本、标签和带 text 列的 df


def train_word2vec(corpus, sg=0):
    """训练Word2Vec模型
    :param sg: 0 for CBOW, 1 for Skip-gram
    """
    model = Word2Vec(sentences=corpus,
                     vector_size=100,  # 词向量维度
                     window=5,  # 上下文窗口大小
                     min_count=1,  # 词频阈值
                     workers=4,  # 训练的线程数
                     sg=sg)  # 0 for CBOW, 1 for Skip-gram
    return model


def get_document_vector(text, model):
    """获取文档的词向量表示（取平均）"""
    tokens = preprocess_text(text)
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def main():
    # Use absolute path to ensure the file is found regardless of working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'train.csv')

    # 加载数据（同时拿到带 text 列的 df）
    corpus, labels, df = load_and_preprocess_data(csv_path)

    # --- 训练 CBOW 模型 (sg=0) ---
    print("正在训练 CBOW 模型...")
    model_cbow = train_word2vec(corpus, sg=0)
    model_cbow.save("word2vec_cbow.model")
    print("CBOW 模型已保存。")

    # --- 训练 Skip-gram 模型 (sg=1) ---
    print("正在训练 Skip-gram 模型...")
    model_skipgram = train_word2vec(corpus, sg=1)
    model_skipgram.save("word2vec_skipgram.model")
    print("Skip-gram 模型已保存。")

    # 获取文档向量 (这里以 CBOW 模型为例，也可以分别获取)
    # 如果后续需要启用，可以取消注释，并带有进度条显示
    # doc_vectors = []
    # for text in tqdm(df['text'], desc="计算文档向量 (CBOW)"):
    #     doc_vector = get_document_vector(text, model_cbow)
    #     doc_vectors.append(doc_vector)
    # X = np.array(doc_vectors)
    # y = labels
    # print("文档向量形状:", X.shape)
    # print("标签形状:", y.shape)

    # 示例：查看某些词的相似词，对比两个模型
    word = "great"
    
    print(f"\n--- 对比词 '{word}' 的相似词 ---")
    
    if word in model_cbow.wv:
        print(f"\n[CBOW 模型] 与'{word}'最相似的词:")
        similar_words = model_cbow.wv.most_similar(word)
        for w, score in similar_words:
            print(f"{w}: {score:.4f}")
    
    if word in model_skipgram.wv:
        print(f"\n[Skip-gram 模型] 与'{word}'最相似的词:")
        similar_words = model_skipgram.wv.most_similar(word)
        for w, score in similar_words:
            print(f"{w}: {score:.4f}")


if __name__ == "__main__":
    main()