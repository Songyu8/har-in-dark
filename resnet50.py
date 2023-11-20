
# %% 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from torchvision.models.video import r3d_18 
from torchvision.io import read_video 
from torchvision.datasets import ImageFolder 
from torchvision import transforms 
import torch.nn.functional as F 




# %% 
# 定义简单的 3D 卷积网络模型 
class Simple3DConvNet(nn.Module): 
def __init__(self, num_classes=10): 
super(Simple3DConvNet, self).__init__() 
self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1) 
self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1) 
self.pool = nn.MaxPool3d(2) 
self.fc1 = nn.Linear(128 * 4 * 4 * 4, 512) 
self.fc2 = nn.Linear(512, num_classes) 

def forward(self, x): 
x = self.pool(F.relu(self.conv1(x))) 
x = self.pool(F.relu(self.conv2(x))) 
x = x.view(-1, 128 * 4 * 4 * 4) 
x = F.relu(self.fc1(x)) 
x = self.fc2(x) 
return x 

# %% 
import os 
import cv2 
import numpy as np 
from tqdm import tqdm 

# %% 
# Assuming 'model' is your pretrained model and 'device' is your device (e.g., 'cuda' or 'cpu') 

class_mapping = { 
'Jump': 0, 
'Run': 1, 
'Sit': 2, 
'Stand': 3, 
'Turn': 4, 
'Walk': 5 
} 

# Transform to be applied to each frame 
transform = transforms.Compose([ 
transforms.ToPILImage(), 
transforms.Resize((224, 224)), 
transforms.ToTensor(), 
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2
), 
]) 

# Path to the folder containing images 
folder_path = "C:\\Users\\songyu\\Desktop\\output_images_enhance" 
# Lists to store features and labels 
fused_feature=[] 
fused_lables=[] 
# Read and process each image in the folder 
for video_folder in tqdm(os.listdir(folder_path)): 
category=video_folder.split('_')[0] 
all_features = [] 
all_labels = [] 
video_path=os.path.join(folder_path,video_folder) 
for image_name in os.listdir(video_path): 
image_path = os.path.join(video_path, image_name) 
frame = cv2.imread(image_path) 
frame = transform(frame) 
frame = torch.unsqueeze(frame, 0) 
all_features.append(frame) 
fused_feature.append(torch.mean(torch.squeeze(torch.stack(all_features)
), dim=0).cpu()) 
fused_lables.append(class_mapping[category]) 

# %% 
X_train = torch.stack(fused_feature) 
y_train = torch.tensor(fused_lables) 
from torch.utils.data import TensorDataset 
dataset = TensorDataset(X_train, y_train) 
print(X_train.shape) 
print(y_train) 

# %% 
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 

# %% 
# 3. 加载预训练的 ResNet-50 模型 
from torchvision import models 
model = models.resnet50(pretrained=True) 

# 4. 修改最后一层 
num_classes = 6 # 根据你的类别数进行设置 
model.fc = nn.Linear(model.fc.in_features, num_classes) 

# %% 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

# 6. 将模型设置为训练模式 
model.train() 

# %% 
# 7. 开始训练 
num_epochs = 50 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device) 

for epoch in range(num_epochs): 
for inputs, labels in train_dataloader: 
inputs, labels = inputs.to(device), labels.to(device) 

optimizer.zero_grad() 
outputs = model(inputs) 
loss = criterion(outputs, labels) 
loss.backward() 
optimizer.step() 

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}') 

# %% 
# 设置测试数据集文件路径 
test_mapping_path = r'C:\\Users\\songyu\\Desktop\\EE6222 train and validate
\validate.txt' 

# 读取测试数据集映射 
test_mapping = np.genfromtxt(test_mapping_path, dtype=str) 

# 收集数据和标签 
fused_feature_test=[] 
fused_lables_test=[] 

test_path='C:\\Users\\songyu\\Desktop\\output_images_enhance_test' 
# Lists to store features and labels 
# Read and process each image in the folder 
for entry in tqdm(test_mapping): 
video_test = [] 
lable_teat = [] 
_,class_index,video_name = entry 
video_extend=video_name.split('.')[0] 
video_path=os.path.join(test_path,video_extend) 
for image_name in os.listdir(video_path): 
image_path = os.path.join(video_path, image_name) 
frame = cv2.imread(image_path) 
frame = transform(frame) 
frame = torch.unsqueeze(frame, 0) 

video_test.append(frame) 
fused_lables_test.append(class_index) 
fused_feature_test.append(torch.mean(torch.squeeze(torch.stack(video_te
st)), dim=0).cpu()) 

# %% 
video_test=torch.stack(fused_feature_test,dim=0) 

lable_teat_true = [int(x) for x in fused_lables_test] 
lable_teat=torch.tensor(lable_teat_true) 
from torch.utils.data import TensorDataset 
dataset_test = TensorDataset(video_test, lable_teat) 
print(video_test.shape) 
print(lable_teat) 

# %% 
test_dataloader = DataLoader(dataset_test, batch_size=32, shuffle=True) 

# %% 


# %% 
model.eval() 

# %% 
pred_lab=[] 

# %% 
# 5. 进行推断 

for inputs in video_test: 
input_tensor = inputs.unsqueeze(0) 
print(input_tensor.shape) 
with torch.no_grad(): 
input_tensor= input_tensor.to(device) 
outputs = model(input_tensor) 
predicted_class = torch.argmax(outputs).item() 
pred_lab.append(predicted_class) 
print(f'Predicted Class: {predicted_class}') 

print( pred_lab) 

# %% 
from sklearn.metrics import accuracy_score 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 


cm = confusion_matrix(lable_teat_true, pred_lab) 

# 评估模型性能 
accuracy = accuracy_score(lable_teat_true, pred_lab) 
print(f"Accuracy: {accuracy}") 

# %% 
# 画图 
plt.figure(figsize=(16, 10)) 

class_labels = ["Jump", "Run", "Sit", "Stand", "Turn", "Walk"] 
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,xticklabels=c
lass_labels, yticklabels=class_labels) 
plt.title('Confusion Matrix') 
plt.xlabel('Predicted') 
plt.ylabel('True') 
plt.show()