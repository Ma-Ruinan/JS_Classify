# JS目标分类网络
## 0.到NAS928上获取权重和数据集
```bash
权重位置：/home/nas928/maruinan/Classify_ModelZoo/checkpoints
数据集位置：/home/nas928/maruinan/Classify_ModelZoo/WQ_dataset
```

## 1.训练好的网络包括

* *Alexnet* <br/>
```bash
accuracy: 94.67%
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 10)
```
```bash
# 模型加载
import torch
input = torch.randn(4, 3, 224, 224)
input = input.to("cuda")
model = torch.load("./checkpoints/AlexNet/6.pth")
model.to("cuda")
output = model(input)
print(output.size()) # torch.Size([4, 10])
```
* *Densenet121* <br/>
```bash
accuracy: 97.14%
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 10)
```
```bash
# 模型加载
import torch
input = torch.randn(4, 3, 224, 224)
input = input.to("cuda")
model = torch.load("./checkpoints/DenseNet121/1.pth")
model.to("cuda")
output = model(input)
print(output.size()) # torch.Size([4, 10])
```
* *ResNet18* <br/>
```bash
accuracy: 
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)
```
```bash
# 模型加载
import torch
input = torch.randn(4, 3, 224, 224)
input = input.to("cuda")
model = torch.load("./checkpoints/ResNet18/1.pth")
model.to("cuda")
output = model(input)
print(output.size()) # torch.Size([4, 10])
```
## 2.前处理后处理方式

<br/>***Pre-processing***<br/>
```bash
train_transforms = transforms.Compose([transforms.Resize(args.size), transforms.ToTensor()])
```
<br/>***Post-processing***<br/>
```bash
images = images.to(device)
labels = labels.to(device)
outputs = model(images)
_, predict = torch.max(outputs.data, dim=1)
total += labels.size(0)
accuracy = accuracy + (predict == labels).sum().item()
```

---

## 3.网络识别类别索引
```bash
0_bazooka （火箭炮）
1_destroyer（驱逐舰）
2_fighter（战斗机）
3_helicopter（直升机）
4_radar（雷达）
5_tank（坦克）
6_early_warning_aircraft（预警机）
7_military_transport_vehicle（军用运输车）
8_aircraft carrier（航空母舰）
9_hovercraft（气垫船）
```

---

## 4.数据集文件夹说明
```bash
├── WQ_dataset
|   ├── 0_bazooka
|   ├── 1_destroyer
|   ├── 2_fighter
|   ├── 3_helicopter
|   ├── 4_radar
|   └── 5_tank
|   └── 6_early_warning_aircraft
|   └── 7_military_transport_vehicle
|   └── 8_aircraft carrier
|   └── 9_hovercraft
（每类文件夹下是**.jpg**图片）
'''


