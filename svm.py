import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# 设置文件夹路径
train_path = r'C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\train'
test_path=r'C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\validate'

class_mapping = {
    'Jump': 0,
    'Run': 1,
    'Sit': 2,
    'Stand': 3,
    'Turn': 4,
    'Walk': 5
}

# 收集数据和标签
video_train = []
lable_train = []

# 遍历每个子文件夹
for class_folder in os.listdir(train_path):
    class_path = os.path.join(train_path, class_folder)
    
    # 遍历每个类别的特征文件
    for feature_file in os.listdir(class_path):
        if feature_file.endswith("_features.npy"):
            feature_path = os.path.join(class_path, feature_file)
            
            # 加载特征数据
            features = np.load(feature_path)
            
            # 将特征添加到X，类别标签添加到y
            video_train.append(features)
            lable_train.append(class_mapping[class_folder])

# 转换为NumPy数组
X_train =  np.concatenate(np.array(video_train))
y_train = np.array(lable_train)


# 设置测试数据集文件路径
test_mapping_path = r'C:\\Users\\songyu\\Desktop\\EE6222 train and validate 2023\\validate.txt' 

# 读取测试数据集映射
test_mapping = np.genfromtxt(test_mapping_path, dtype=str)


# 收集数据和标签
video_test = []
lable_teat = []

# 遍历测试映射
for entry in test_mapping:
    _,class_index,video_name = entry
    feature_file_name = f"{video_name.split('.')[0]}_features.npy"
    feature_file_path = os.path.join(test_path, feature_file_name)
    
    # 加载特征数据
    features = np.load(feature_file_path)
    
    # 将特征添加到X_test，类别标签添加到y_test
    video_test.append(features)
    lable_teat.append(int(class_index))

video_test=np.concatenate(np.array(video_test))
lable_teat=np.array(lable_teat)



# 构建SVM模型
svm_model = SVC(kernel='rbf',C=0.5)

# 训练模型
svm_model.fit(X_train, y_train)

# 预测测试集
y_pred = svm_model.predict(video_test)

print(y_pred)

cm = confusion_matrix(lable_teat, y_pred)

# 评估模型性能
accuracy = accuracy_score(lable_teat, y_pred)
print(f"Accuracy: {accuracy}")

# 画图
plt.figure(figsize=(16, 10))

class_labels = ["Jump", "Run", "Sit", "Stand", "Turn", "Walk"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

