import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from pokemon import Pokemon
from resnet import ResNet18
import warnings

warnings.filterwarnings("ignore")

batchsz = 32
lr = 1e-3
epochs = 10
device = torch.device('cpu')
torch.manual_seed(1234)  # 设置随机种子

# 加载数据#####################
train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()


# 验证，返回准确率
def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():
    model = ResNet18(5).to(device)  # 网络模型
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
    criteon = nn.CrossEntropyLoss()  # 损失熵

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line(Y=[0.], X=[-1.], win='loss', opts=dict(title='loss'))
    viz.line(Y=[0.], X=[-1.], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            # x:[b,3,224,224]   y:[b]
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 4 == 0:
            #     val_acc = evalute(model, val_loader) * 100
            #     print('epoch:', epoch, '\tbatch:', batch_idx, '\tval_acc:', '%.2f' % val_acc,
            #           '%', '\tloss:', loss.item())

            viz.line(Y=[loss.item()], X=[global_step], win='loss', update='append')
            global_step += 1

        # if epoch % 2 == 0:
        val_acc = evalute(model, val_loader) * 100
        viz.line(Y=[val_acc], X=[epoch], win='val_acc', update='append')
        print('epoch:', epoch, '\tval_acc:', '%.2f' % val_acc, '%')

        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.zzl')

    print('best_epoch:', best_epoch, '\tbest_val_acc:', '%.2f' % best_acc, '%')

    # 从最好的状态加载模型
    model.load_state_dict(torch.load('best.zzl'))
    print('loaded from check point!')

    # test测试
    test_acc = evalute(model, test_loader) * 100
    print('test_acc:', '%.2f' % test_acc, '%')


if __name__ == '__main__':
    main()
