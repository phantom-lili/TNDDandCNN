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

def find_threshold(similarities):
    """计算阈值：这里使用均值 + 一个标准差"""
    mean = np.mean(similarities)
    std_dev = np.std(similarities)
    threshold = mean - std_dev*2
    return threshold

def main(audio_files):
    """主程序"""
    if len(audio_files) < 2:
        print("请确保有至少两个音频文件。")
        return

    features = []
    for file in audio_files:
        audio, sr = load_audio(file)
        features.append(extract_features(audio))
    
    # 计算所有音频之间的相似度
    similarities = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sim = calculate_similarity(features[i], features[j])
            similarities.append(sim)
            print(f"{os.path.basename(audio_files[i])} 和 {os.path.basename(audio_files[j])} 之间的相似度: {sim:.4f}")

    # 计算并输出阈值
    threshold = find_threshold(similarities)
    print(f"建议的相似度阈值: {threshold:.4f}")

# 将所有音频文件路径放在一个列表中
audio_files = [
"D:\es\好的\yuan.wav",
"D:\es\好的\yuan2.wav",
"D:\es\好的\yuan3.wav",
"D:\es\好的\yuan4.wav",
"D:\es\好的\yuan5.wav",
"D:\es\好的\yuan6.wav",
"D:\es\好的\yuan7.wav",
"D:\es\好的\yuan8.wav"
]

main(audio_files)


