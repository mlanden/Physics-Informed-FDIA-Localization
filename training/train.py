import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from models import PredictionModel


def rnn_train(conf, dataset):
    normal_data = DataLoader(dataset, batch_size = conf["train"]["batch_size"], shuffle = True)

    model = PredictionModel(conf)
    print(model)
    decay = conf["train"]["regularization"]
    if decay > 0:
        optimizer = optim.Adam(model.parameters(), lr = conf["train"]["lr"], weight_decay = decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr = conf["train"]["lr"])

    loss_fn = torch.nn.MSELoss()
    epochs = conf["train"]["epochs"]
    writer = SummaryWriter("tf_board/Swat_prediction")
    for i in range(epochs):
        epoch_loss = 0
        for seq, target in normal_data:
            predicted = model(seq)
            predicted = predicted[:, -1, :]

            loss = loss_fn(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            # for p in model.parameters():
            #     print(torch.max(p.grad))

            optimizer.step()

            epoch_loss += loss.item()
        writer.add_scalar("Loss/train", epoch_loss, i)

        if i % 10 == 0:
            print(f"Epoch {i :3d} / {epochs}: Loss: {epoch_loss}")