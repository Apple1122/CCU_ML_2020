from .transforms              import build_transform
from torchvision              import datasets
from torch.utils.data         import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from pandas                   import Series, DataFrame

import numpy  as np
import pandas as pd
import os

np.random.seed(11)

def make_train_loader(cfg):
    
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TRAIN_BATCH_SIZE
    valid_size  = cfg.DATA.VALIDATION_SIZE
    train_path  = cfg.PATH.TRAIN_SET
    
    train_df = pd.read_csv('C:\\Users\\Kyle\\Downloads\\CCU_ML_2020\\homework2\\ml-hw\\hw2\\C1-P1_Train Dev_fixed\\train.csv')

    labels = train_df['label']
    label_np = Series.to_numpy(labels)

    #看一下一共多少不同种类
    label_set = set(label_np)

    #构建一个编号与名称对应的字典，以后输出的数字要变成名字的时候用：
    label_list = list(label_set)
    dic = {}
    for i in range(3):
        dic[  label_list[i]   ] = i

    train_file =  Series.to_numpy(train_df["image_id"])
    file_train = [os.path.join("C:\\Users\\Kyle\\Downloads\\CCU_ML_2020\\homework2\\ml-hw\\hw2\\C1-P1_Train Dev_fixed\\C1-P1_Train", i) for i in train_file ]
    np.save( "file_train.npy" ,file_train )

    train_labels = Series.to_numpy(train_df["label"])

    number_train = []

    for i in range(train_labels.shape[0]):
        number_train.append(  dic[ train_labels[i] ]  )
              
    number_train = np.array(number_train)

    np.save( "number_train.npy" ,number_train )

    transforms = build_transform(cfg)

    def default_loader(path):
        img_pil =  Image.open(path)
        img_pil = img_pil.resize((224,224))
        img_tensor = transforms(img_pil)
        return img_tensor

    class train_set(Dataset):
        def __init__(self, loader=default_loader):
            #定义好 image 的路径
            self.images = file_train
            self.target = number_train
            self.loader = loader

        def __getitem__(self, index):
            fn = self.images[index]
            img = self.loader(fn)
            target = self.target[index]
            return img,target

        def __len__(self):
            return len(self.images)

    trainset  = train_set()

    
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
    
    return train_loader, valid_loader

def make_test_loader(cfg):

    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TEST_BATCH_SIZE
    test_path   = cfg.PATH.TEST_SET

    transforms = build_transform(cfg)

    testset = datasets.ImageFolder(test_path, transform=transforms)

    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)

    return test_loader


