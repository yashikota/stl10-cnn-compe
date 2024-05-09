import matplotlib.pyplot as plt
import numpy as np


def loss_acc_plot(
    uuid,
    train_loss_history,
    train_accuracy_history,
    test_loss_history,
    test_accuracy_history,
):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Train loss
    axes[0, 0].plot(train_loss_history)
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_title("Train Loss")

    # Train accuracy
    axes[0, 1].plot(train_accuracy_history)
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].set_title("Train Accuracy")

    # Test loss
    axes[1, 0].plot(test_loss_history)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("loss")
    axes[1, 0].set_title("Test Loss")

    # Test accuracy
    axes[1, 1].plot(test_accuracy_history)
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].set_title("Test Accuracy")

    fig.suptitle(f"{uuid}", fontsize=16)
    fig.tight_layout()
    plt.savefig(f"result/{uuid}/loss_acc.png")
    plt.close()


def plot_confusion_matrix(uuid, cm):
    labels = [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(uuid, fontsize=14, y=0.95)

    ax = fig.add_subplot(111)
    ax.matshow(cm, cmap="coolwarm")

    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Correct", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"result/{uuid}/matrix.png", bbox_inches="tight")
    plt.close()
