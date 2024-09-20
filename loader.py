import glob
import torch
# 语言： Python
# 作用：# 第一个py用来包装好测试集和训练集，完成所需工作
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms


# 2 继承Dataset实现Mydataset子类
class Mydataset(data.Dataset):
    # init() 初始化方法，传入数据文件夹路径
    def __init__(self, root):
        self.imgs_path = root

    # getitem() 切片方法，根据索引下标，获得相应的图片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]

    # len() 计算长度方法，返回整个数据文件夹下所有文件的个数
    def __len__(self):
        return len(self.imgs_path)


# 3、使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r"./data/*.png")  # 数据文件夹路径../jpeg

# 一 利用自定义的类来创建对象brake_dataset
brake_dataset = Mydataset(all_imgs_path)
brake_dataloader = torch.utils.data.DataLoader(brake_dataset, batch_size=4)  # 每次迭代时返回4个数据

# 为所有的图片制造相对应的标签
species = ['haemorrhage', 'hardexudate', 'normal', 'softexudate']

species_to_id = dict((c, i) for i, c in enumerate(species))

id_to_species = dict((v, k) for k, v in species_to_id.items())

# 二 为对应的图片，打上标签 PS：这一部分很重要DDDDDDDanger
all_labels = []
for img in all_imgs_path:
    # print(img)
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
# print(len(all_labels))

# 三 将图片转化为Tensor，展示图片与标签对应的关系
# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class Mydatasetpro(data.Dataset):
    # 初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 切片处理
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pill_img = Image.open(img)
        pill_img = pill_img.convert('RGB')
        data = self.transforms(pill_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 16
brake_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
brake_dataloader = data.DataLoader(
    brake_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True

)

imgs_batch, labels_batch = next(iter(brake_dataloader))
# print(imgs_batch.shape)


# 四 划分数据集和测试集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

s = int(len(all_imgs_path) * 0.9)
train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

print(len(train_imgs), len(train_labels), len(test_imgs), len(test_labels))

# 将对应的数据，转化为对应的Tensor Data
train_ds = Mydatasetpro(train_imgs, train_labels, transform)
test_ds = Mydatasetpro(test_imgs, test_labels, transform)
print("@#$%^&^%$^&*&^%$&(*&^%$^")
train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)



# import numpy as np
# from matplotlib import pyplot as plt
# # # 语言： Python
# # # 作用：#  将image_feature.npy文件+label.npy文件传到TSNE降维算法中，进行二维可视化展示
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
#
# train = np.load('Feature_ResNet_Test_FTLoss.npy')
# labels12 = np.load('Label_ResNet_Test_FTLoss.npy')
#
# # -===================================================================================
# # 降到二维：平面图展示
# asdasfa = 1
# if asdasfa == 1:
#     labels_huitu_string = ['haemorrhage', 'hardexudate', 'normal', 'softexudate']
#     tsne = TSNE(n_components=2, learning_rate=150).fit_transform(train)
#
#     plt.figure(figsize=(12, 6))
#     plt.scatter(tsne[:, 0], tsne[:, 1], c=labels12)
#     plt.show()
# # -===================================================================================
# if asdasfa == 2:
#     model_pca = PCA(n_components=2)
#     X_PCA = model_pca.fit_transform(train)
#     # # 绘图
#     labels_huitu = [0, 1, 2, 3]
#     Colors = ['red', 'orange', 'yellow', 'green']
#     labels_huitu_string = ['haemorrhage', 'hardexudate', 'normal', 'softexudate']
#
#     plt.figure(figsize=(8, 6), dpi=80)  # figsize定义画布大小，dpi定义画布分辨率
#     plt.title('Transformed samples via sklearn.decomposition.PCA')
#     # 分别确定x和y轴的含义及范围
#     plt.xlabel('x_values')
#     plt.ylabel('y_values')
#     for tlabel in labels_huitu:
#         # pca读取数据
#         x_pca_data = X_PCA[labels12 == tlabel, 0]
#         y_pca_data = X_PCA[labels12 == tlabel, 1]
#         plt.scatter(x=x_pca_data, y=y_pca_data, s=20, c=Colors[tlabel], label=labels_huitu_string[tlabel])
#     plt.legend(loc="upper right")  # 输出标签信息在右上角
#     plt.grid()
#     plt.show()