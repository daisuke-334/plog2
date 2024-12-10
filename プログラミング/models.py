from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sepuential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
def test_accuracy(model,dataloader):
    #全てのミニバッチに対して推論をして、正解率を計算する
    n_corrects = 0 #正解した個数

    model.eval()
    for image_batch,label in dataloader:
        #モデルに入れて結果（logits）を出す
        with torch.no_grad():
            logits_batch = model(image_batch)

        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    #精度（正解率）を計算する
    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy

acc_test = models.test_accuracy(model,dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

def train(model,dataloader,loss_fn,optimizer):
    """1 epoch の学習を行う"""
    model.train()
    for image_batch,label_batch in dataloader:
        #モデルにパッチを入れて計算
        logits_batch = model(image_batch)

        #損失（誤差）を計算する
        loss = loss_fn(logits_batch,label_batch)

        #最適化
        optimizer.zero_gard()
        loss.backwaed()
        optimizer.step()

    #最後のバッチのロス
    return loss.item()

#精度を計算
acc_test = models.test_accuracy(model,dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

#学習
models.train(model,dataloader_test,loss_fn,optimizer)

#もう一度制度を計算
acc_test = models.test_accuracy(model,dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

def test(model,dataloader,loss_fn):
    loss_total - 0.0

    model.eval()
    for image_batch,label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)
        
        loss = loss_fn(logits_batch,label_batch)
        loss_total += loss.item()

    return loss_total / len(dataloader)

loss_train_history=[]
loss_test_history=[]
acc_train_history=[]
acc_test_history=[]

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}',end=': ',flush=True)

    loss_train=models.train(model,dataloader_train,loss_fn)
    loss_train_history.append(loss_train)
    print(f'train loss: {loss_train:.3f}',end=', ')

    loss_test=models.train(model,dataloader_test,loss_fn)
    loss_test_history.append(loss_test)
    print(f'test loss: {loss_test:.3f}',end=', ')

    acc_train=models.test_accuracy(model,dataloader_train)
    acc_train_history.append(acc_train)
    print(f'train accuracy: {acc_train*100:.3f}%',end=', ')

    acc_test=models.test_accuracy(model,dataloader_test)
    acc_test_history.append(acc_test)
    print(f'test accuracy: {acc_test*100:.3f}%',end=', ')


plt.plot(acc_train_history,label='train')
plt.plot(acc_test_history,label='test')
plt.xlabel('epochs')
plt.ylabe('accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(loss_train_history,label='train')
plt.plot(loss_test_history,label='test')
plt.xlabel('epochs')
plt.ylabe('loss')
plt.legend()
plt.grid()
plt.show()


        