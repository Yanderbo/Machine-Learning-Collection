import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datasets(不均衡数据集):
# 1. Oversampling（常用）
# 2. Class weighting
#loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,50]))

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    #返回值按照字母顺序升序排序
    #定义权重
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        #返回值按照数据集大小排序
        subdir.sort()
        #为了统一排序，将文件按照数据集字母排序
        print(len(files))
        if len(files) > 0:
            class_weights.append(1/len(files))#让数据集长度和权重成反比，长度越长，权重越小

    print(class_weights)
    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]#类别权重
        sample_weights[idx] = class_weight#样品权重

    print(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=
                                    len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    loader = get_loader(root_dir="dataset", batch_size=8)

    num_retrievers = 0
    num_elkhounds = 0
    for epoch in range(1):
        for data, labels in loader:
            num_retrievers += torch.sum(labels==0)
            num_elkhounds += torch.sum(labels==1)

    print(num_retrievers)
    print(num_elkhounds)

if __name__ == "__main__":
    main()

