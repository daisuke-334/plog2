import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transfroms.v2 as transforms

import models


#データセットの前処理を定義
ds_transfrom = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDype(torch.float32,scale=True)
])

#データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,  #訓練用を指定
    download=True,
    transform=ds_transform
)
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,  #テスト用を指定
    download=True,
    transform=ds_transform
)
#ミニバッチに分割する　DataLoaderを作る
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch=batch_size
)
#バッチを取り出す実験
#この後の処理では不要なので、確認したら削除してよい
for image_batch,label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break #1つ目で終了

#モデルをインスタンス化する
model = models.MyModel()

#損失関数（誤差関数・ロス関数）の選択
loss_fn = torch.nn.CrossEntropyLoss()

#最適化の方法の選択
learning_rate = 1e-3 #学習率
optimizer = torch.optin.SDG(model.parameters(),lr=learning_rata)

