# JS目标分类网络

## 1.训练好的网络包括

* *Alexnet* <br/>
```bash
accuracy: 
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
```
* *Densenet121* <br/>
```bash
accuracy: 
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
```
<br/>***transforms process***<br/>
```bash
train_transforms = transforms.Compose([transforms.Resize(args.size), transforms.ToTensor()])
```

---

## 2.网络识别类别索引
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

## 3.数据集文件夹说明
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


