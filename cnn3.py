import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Audio file paths and corresponding labels
AUDIO_FILES = [r"D:\es\好的\gua.wav",r"D:\es\好的\gua2.wav",r"D:\es\好的\gua4.wav",r"D:\es\好的\gua5.wav",
r"D:\es\好的\gua6.wav",r"D:\es\好的\gua7.wav",r"D:\es\好的\gua8.wav",r"D:\es\好的\qing.wav",r"D:\es\好的\qing2.wav",
r"D:\es\好的\qing3.wav",r"D:\es\好的\qing4.wav",r"D:\es\好的\qing5.wav",r"D:\es\好的\qing6.wav",r"D:\es\好的\qing7.wav",r"D:\es\好的\qing8.wav",r"D:\es\好的\yuan.wav",r"D:\es\好的\yuan2.wav",
r"D:\es\好的\yuan3.wav",r"D:\es\好的\yuan4.wav",r"D:\es\好的\yuan5.wav",r"D:\es\好的\yuan6.wav",r"D:\es\好的\yuan7.wav",r"D:\es\好的\yuan8.wav",r"D:\es\碎的\sui.wav",r"D:\es\碎的\sui2.wav",
r"D:\es\碎的\sui3.wav",r"D:\es\碎的\sui4.wav",r"D:\es\碎的\sui5.wav",r"D:\es\碎的\sui7.wav",r"D:\es\碎的\sui8.wav",r"D:\es\碎的\sui9.wav",r"D:\es\碎的\sui10.wav",r"D:\es\碎的\sui11.wav",
r"D:\es\碎的\sui12.wav",r"D:\es\碎的\sui13.wav",r"D:\es\碎的\sui14.wav",r"D:\es\碎的\sui15.wav",r"D:\es\碎的\sui16.wav",r"D:\es\碎的\sui17.wav",r"D:\es\碎的\sui18.wav",r"D:\es\碎的\sui19.wav",
r"D:\es\碎的\sui20.wav",r"D:\es\碎的\sui21.wav",r"D:\es\碎的\sui22.wav",r"D:\es\碎的\sui23.wav",r"D:\es\碎的\sui24.wav",
r"D:\es\潮的\chao12.wav",r"D:\es\潮的\chao1.wav",r"D:\es\潮的\chao13.wav",r"D:\es\潮的\chao2.wav",
r"D:\es\潮的\chao14.wav",
r"D:\es\潮的\chao3.wav",
r"D:\es\潮的\chao15.wav",
r"D:\es\潮的\chao4.wav",
r"D:\es\潮的\chao16.wav",
r"D:\es\潮的\chao5.wav",
r"D:\es\潮的\chao17.wav",
r"D:\es\潮的\chao6.wav",
r"D:\es\潮的\chao18.wav",
r"D:\es\潮的\chao7.wav",
r"D:\es\潮的\chao19.wav",
r"D:\es\潮的\chao8.wav",
r"D:\es\潮的\chao20.wav",
r"D:\es\潮的\chao9.wav",
r"D:\es\潮的\chao21.wav",
r"D:\es\潮的\chao10.wav",
r"D:\es\潮的\chao22.wav",
r"D:\es\潮的\chao11.wav",
r"D:\es\潮的\chao23.wav",
r"D:\es\潮的\chao24.wav"]
LABELS = [0] * 23 + [1] * 23 + [2] * 24 # Example label list

# 检查音频文件和标签是否正常加载
if len(AUDIO_FILES) != len(LABELS):
    raise ValueError("音频文件数量与标签数量不匹配。")

# 提取 MFCC 特征
def extract_features(file_path, n_mfcc=13, max_time_steps=100):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs[:, :max_time_steps]  # 截断或填充
    mfccs = np.pad(mfccs, ((0, 0), (0, max_time_steps - mfccs.shape[1])), 'constant')
    return mfccs

# 准备数据
def prepare_data(audio_files, labels):
    features = np.array([extract_features(file) for file in audio_files])
    features = features[..., np.newaxis]  # 增加一个维度，适应 CNN 输入
    labels_one_hot = to_categorical(labels, num_classes=len(np.unique(labels)))
    return features, labels_one_hot

# 构建 CNN 模型
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 训练和评估模型
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = build_model(input_shape=(13, 100, 1), num_classes=len(np.unique(LABELS)))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return model

# 对新音频进行预测
def predict_new_audio(file_path, model):
    feature = extract_features(file_path)
    feature = feature[np.newaxis, ..., np.newaxis]  # 添加批次和通道维度
    prediction = model.predict(feature)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# 主执行流程
if __name__ == "__main__":
    # 准备数据
    features, labels_one_hot = prepare_data(AUDIO_FILES, LABELS)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, test_size=0.2, random_state=42)

    # 训练和评估模型
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # 保存模型
    model.save("voice_recognition_model.h5")

    # 加载模型
    from tensorflow.keras.models import load_model
    loaded_model = load_model("voice_recognition_model.h5")

    # 测试新的音频文件
    new_audio_file = r"D:\es\yu\15544-5-0-13.wav"  # 更改为你的新音频文件路径
    predicted_label = predict_new_audio(new_audio_file, loaded_model)
    print(f"Predicted speaker label: {predicted_label}")
