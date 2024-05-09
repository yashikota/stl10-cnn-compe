import datetime
import os
import random
import uuid

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data import TestDataset, TrainDataset
from plot import loss_acc_plot, plot_confusion_matrix


# ニューラルネットワークの定義
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.blocks = nn.ModuleList()
        channels = [3, 32, 64, 128, 256, 512]

        for i in range(1, len(channels)):
            self.blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels[i - 1]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )

        self.fc_block = nn.Sequential(
            nn.Linear(in_features=512 * 3 * 3, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=512, out_features=10),
        )

    # 順伝播の計算
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)

        return x


# データの読み込み用の関数
def load_data(batch):
    train_dataset = TrainDataset()
    train_loader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True
    )

    # CutMix or Mixup
    num_classes = 10
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes, alpha=0.8)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for images, labels in train_loader:
        images, labels = cutmix_or_mixup(images, labels)

    test_dataset = TestDataset()
    test_loader = DataLoader(
        test_dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True
    )

    return {"train": train_loader, "test": test_loader}


def objective(trial):
    trial.set_user_attr("uuid", uid)
    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("memo", memo)

    # 学習結果の保存用
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    learning_rate = 0.000358018666556301
    weight_decay = 0.00010000000000000002

    batch_size = 128
    optimizer_name = "Adam"

    # ハイパーパラメータの設定
    # learning_rate = trial.suggest_float(
    #     "learning_rate", 1e-5, 9e-4, log=True
    # )  # 0.00001 ~ 0.0009
    # weight_decay = trial.suggest_float(
    #     "weight_decay", 1e-6, 9e-4, log=True
    # )  # 0.000001 ~ 0.0009

    # batch_size = trial.suggest_categorical("batch_size", [128, 128, 256])
    # optimizer_name = trial.suggest_categorical(
    #     "optimizer", ["Adam", "RMSprop", "Adagrad"]
    # )

    # ネットワークの作成
    model = NeuralNetwork().to(device)

    # データの読み込み
    data_loader = load_data(batch_size)

    # 最適化手法の定義
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 誤差関数の定義
    criterion = nn.CrossEntropyLoss()

    # スケジューラーの定義
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    for i in tqdm(range(epochs)):
        # 学習
        model.train()
        accuracy = 0.0

        for data, target in data_loader["train"]:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct = torch.sum(pred == target).item()
            accuracy = correct / len(data_loader["train"].dataset)

        train_accuracy_history.append(accuracy)
        train_loss_history.append(loss.item())

        # テスト
        model.eval()
        test_loss = 0.0
        correct = 0.0
        confusion_matrix = torch.zeros(10, 10)

        with torch.no_grad():
            for data, target in data_loader["test"]:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += torch.sum(pred == target).item()

                for t, p in zip(target, pred):
                    confusion_matrix[t, p] += 1

        test_loss /= len(data_loader["test"].dataset)
        accuracy = correct / len(data_loader["test"].dataset)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(accuracy)

        scheduler.step(test_loss)

        plot_confusion_matrix(
            uid=uid, name=f"matrix", cm=confusion_matrix.cpu().numpy()
        )
        loss_acc_plot(
            uid=uid,
            name=f"loss_acc",
            train_loss_history=train_loss_history,
            train_accuracy_history=train_accuracy_history,
            test_loss_history=test_loss_history,
            test_accuracy_history=test_accuracy_history,
        )

        if accuracy > 0.89:
            torch.save(model.state_dict(), f"result/{uid}/{str(accuracy)}_{str(i)}.pth")
            torch.save(model, f"result/{uid}/{str(accuracy)}_{str(i)}.pth")

    accuracy = max(test_accuracy_history)

    with open(f"result/{uid}/{str(accuracy)}", "w") as f:
        f.write(str(test_accuracy_history.index(accuracy)))

    plot_confusion_matrix(
        uid=uid, name=f"matrix_{str(accuracy)}", cm=confusion_matrix.cpu().numpy()
    )
    loss_acc_plot(
        uid=uid,
        name=f"loss_acc_{str(accuracy)}",
        train_loss_history=train_loss_history,
        train_accuracy_history=train_accuracy_history,
        test_loss_history=test_loss_history,
        test_accuracy_history=test_accuracy_history,
    )

    return accuracy


def torch_fix_seed(seed=413):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def init():
    global uid, epochs, device, memo

    torch_fix_seed()
    uid = str(uuid.uuid4())
    device = torch.device("cuda")

    os.makedirs(f"result/{uid}", exist_ok=True)

    memo = "best"

    # memoを保存
    with open(f"result/{uid}/{memo}", "w") as f:
        f.write(memo)

    # 実行時時刻を保存
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"result/{uid}/{time}", "w") as f:
        f.write(time)

    # 自身のソースコードをコピー
    os.system(f"cp {__file__} result/{uid}/main.py")


if __name__ == "__main__":
    init()

    n_trials = 1
    epochs = 30

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
