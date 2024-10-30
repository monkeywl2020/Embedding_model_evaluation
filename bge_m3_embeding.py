import requests
from numpy.linalg import norm
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import ast

# 确保输入是 NumPy 数组
def ensure_numpy_array(v):
    """确保输入是 NumPy 数组"""
    return np.array(v) if not isinstance(v, np.ndarray) else v

# 余弦相似度（Cosine Similarity）
def cosine_similarity(v1, v2):
    v1, v2 = ensure_numpy_array(v1), ensure_numpy_array(v2)
    return 1 - cosine(v1, v2)

# 欧几里得相似度（Euclidean Similarity）
def euclidean_similarity(v1, v2):
    v1, v2 = ensure_numpy_array(v1), ensure_numpy_array(v2)
    return 1 / (1 + euclidean(v1, v2))

# 曼哈顿相似度（Manhattan Similarity）
def manhattan_similarity(v1, v2):
    v1, v2 = ensure_numpy_array(v1), ensure_numpy_array(v2)
    return 1 / (1 + np.sum(np.abs(v1 - v2)))

# 皮尔逊相似度（Pearson Similarity）
def pearson_similarity(v1, v2):
    v1, v2 = ensure_numpy_array(v1), ensure_numpy_array(v2)
    corr, _ = pearsonr(v1, v2)
    return (corr + 1) / 2  # 将范围从[-1, 1]转换到[0, 1]

# 多指标相似度（Multi-metric Similarity）
def multi_metric_similarity(v1, v2):
    cos_sim = cosine_similarity(v1, v2)
    euc_sim = euclidean_similarity(v1, v2)
    man_sim = manhattan_similarity(v1, v2)
    return (cos_sim + euc_sim + man_sim) / 3

# 从文件中读取句子对
def read_sentence_pairs_from_file(file_path):
    sentence_pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 忽略以 # 开头的注释行和空行
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            # 处理有效的句子对，确保以 "|||" 分隔
            if "|||" in line:
                sentence1, sentence2 = line.strip().split("|||")
                sentence_pairs.append((sentence1.strip(), sentence2.strip()))
    return sentence_pairs


# 主函数：读取文件并计算相似度
def main():
    # 读取句子对文件
    sentence_pairs = read_sentence_pairs_from_file("sentence_pairs.txt")
    
    # 循环处理每一对句子
    for idx, (sentence1, sentence2) in enumerate(sentence_pairs, 1):
        # 获取两个句子的嵌入向量
        response1 = requests.post("http://172.21.30.221:8060/embed", json={
            "inputs": sentence1
        })
        #print("===response1:",response1.text)  # 确认返回的是一个嵌入向量列表
        embedding1 = ast.literal_eval(response1.text)
        #print("===embedding1:",embedding1) 

        response2 = requests.post("http://172.21.30.221:8060/embed", json={
            "inputs": sentence2
        })
        embedding2 = ast.literal_eval(response2.text)
        #embedding2 = response2.json()["data"][0]["embedding"]

        # 计算相似度
        cos_sim = cosine_similarity(embedding1[0], embedding2[0])
        euc_sim = euclidean_similarity(embedding1[0], embedding2[0])
        man_sim = manhattan_similarity(embedding1[0], embedding2[0])
        pear_sim = pearson_similarity(embedding1[0], embedding2[0])
        multi_sim = multi_metric_similarity(embedding1[0], embedding2[0])

        # 打印结果
        print(f"========= 对比 {idx} =========")
        print(f"句子 1: {sentence1}")
        print(f"句子 2: {sentence2}")
        print(f"余弦相似度 (Cosine Similarity): {cos_sim:.4f}")
        print(f"欧几里得相似度 (Euclidean Similarity): {euc_sim:.4f}")
        print(f"曼哈顿相似度 (Manhattan Similarity): {man_sim:.4f}")
        print(f"皮尔逊相似度 (Pearson Similarity): {pear_sim:.4f}")
        print(f"多指标相似度 (Multi-metric Similarity): {multi_sim:.4f}")
        print("==========================\n")

# 运行主程序
if __name__ == "__main__":
    main()
