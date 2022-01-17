import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ClassificationDataset import PickOrPlaceClassificationDataset
from clfd import CLfD


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


def plot_loss(losses, filename, x=None, window=5):
    N = len(losses)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(losses[max(0, t - window) : (t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.clf()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(x, running_avg)
    plt.savefig(filename)


def inference(model, batch_size, type, device, view_point_type):
    image_dataset = PickOrPlaceClassificationDataset(
        type=type, normalize_input=True, view_point_type=view_point_type
    )
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )

    feature_vector = []
    labels_vector = []
    for x, y in image_loader:
        x = x.to(device)

        with torch.no_grad():
            h, _, z, _ = model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    return feature_vector, labels_vector


def create_data_loaders_from_features(
    model: CLfD, batch_size: int, device, view_point_type: str
):
    X_train, y_train = inference(
        model=model,
        batch_size=batch_size,
        type="train",
        device=device,
        view_point_type=view_point_type,
    )
    X_val, y_val = inference(
        model=model,
        batch_size=batch_size,
        type="val",
        device=device,
        view_point_type=view_point_type,
    )

    feature_train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        feature_train_dataset, batch_size=batch_size, shuffle=True
    )

    feature_val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    val_loader = torch.utils.data.DataLoader(
        feature_val_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader


def train(loader, model, criterion, optimizer, device):
    loss_epoch = 0
    accuracy_epoch = 0
    for x, y in loader:
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(loader, model, criterion, device):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for x, y in loader:
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def main():
    view_point_type = "both"
    batch_size = 50
    lr = 3e-4
    epochs = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CLfD(backbone="resnet18", out_features=32, projection_dim=64)
    encoder.load_state_dict(torch.load("pretrained.tar", map_location=device.type))
    encoder = encoder.to(device)
    encoder.eval()

    model = LogisticRegression(n_features=32, n_classes=2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = create_data_loaders_from_features(
        encoder, batch_size, device, view_point_type=view_point_type
    )

    for epoch in range(epochs):
        loss_epoch, accuracy_epoch = train(
            train_loader, model, criterion, optimizer, device
        )
        print(
            f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch = test(val_loader, model, criterion, device)
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(val_loader)}\t Accuracy: {accuracy_epoch / len(val_loader)}"
    )


if __name__ == "__main__":
    main()
