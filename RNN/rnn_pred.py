import torch
from torch import nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from rnn import Net, num_time_steps, hidden_size, lr


def main():
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    hidden_prev = torch.zeros(1, 1, hidden_size)

    for iter in range(6000):
        start = np.random.randint(0, 3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_time_steps)
        data = np.sin(time_steps)
        data = data.reshape(num_time_steps, 1)
        x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
        y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        # for p in model.parameters():
        #     print(p.grad.norm())
        # torch.nn.utils.clip_grad_norm_(p, 10)
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))

    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

    predictions = []
    input = x[:, 0, :]
    for _ in range(x.shape[1]):
        input = input.view(1, 1, 1)
        (pred, hidden_prev) = model(input, hidden_prev)
        input = pred
        predictions.append(pred.detach().numpy().ravel()[0])

    x = x.data.numpy().ravel()
    y = y.data.numpy()
    plt.scatter(time_steps[:-1], x.ravel(), s=90)
    plt.plot(time_steps[:-1], x.ravel())

    plt.scatter(time_steps[1:], predictions)
    plt.show()


if __name__ == "__main__":
    main()

