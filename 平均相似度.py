import os
import numpy as np
import librosa
from scipy.spatial.distance import cosine

def load_audio(file_path):
    """加载音频文件并返回音频数据和采样率"""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_features(audio):
    """提取音频的 MFCC 特征"""
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def calculate_similarity(features1, features2):
    """计算两个音频特征之间的余弦相似度"""
    return 1 - cosine(features1, features2)

def average_similarity(group1, group2):
    """计算两组音频之间的平均相似度"""
    similarities = []
    for file1 in group1:
        audio1, sr1 = load_audio(file1)  # 加载音频文件
        features1 = extract_features(audio1)
        for file2 in group2:
            audio2, sr2 = load_audio(file2)  # 加载音频文件
            features2 = extract_features(audio2)
            sim = calculate_similarity(features1, features2)
            similarities.append(sim)
    return np.mean(similarities)

def main(groups):
    """主程序"""
    # 遍历每组，计算不同组之间的相似性
    num_groups = len(groups)
    results = np.zeros((num_groups, num_groups))
    
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            avg_sim = average_similarity(groups[i], groups[j])
            results[i, j] = avg_sim
            results[j, i] = avg_sim  # 对称填充

    # 输出结果
    for i in range(num_groups):
        for j in range(num_groups):
            if i < j:
                print(f"组 {i+1} 和 组 {j+1} 之间的平均相似度: {results[i, j]:.4f}")

# 将每组音频文件路径放在一个列表中
group1 = [
    r"D:\es\好的\gua.wav", r"D:\es\好的\gua2.wav", r"D:\es\好的\gua4.wav", r"D:\es\好的\gua5.wav",
    r"D:\es\好的\gua6.wav", r"D:\es\好的\gua7.wav", r"D:\es\好的\gua8.wav"
]
group2 = [
   r"D:\es\好的\qing.wav",
    r"D:\es\好的\qing2.wav", r"D:\es\好的\qing3.wav", r"D:\es\好的\qing4.wav", r"D:\es\好的\qing5.wav",
    r"D:\es\好的\qing6.wav", r"D:\es\好的\qing7.wav", r"D:\es\好的\qing8.wav",
]
group3 = [
r"D:\es\好的\yuan.wav",
    r"D:\es\好的\yuan2.wav", r"D:\es\好的\yuan3.wav", r"D:\es\好的\yuan4.wav", r"D:\es\好的\yuan5.wav",
    r"D:\es\好的\yuan6.wav", r"D:\es\好的\yuan7.wav", r"D:\es\好的\yuan8.wav"
]

# 将三组音频放入一个列表中
audio_groups = [group1, group2, group3]

# 运行主程序
main(audio_groups)
