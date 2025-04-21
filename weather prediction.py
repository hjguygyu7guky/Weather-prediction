import torch
from torch import nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("weather(in).csv")
df = df.dropna()

X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = nn.Sequential(
    nn.Linear(in_features=18, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=5),
    nn.ReLU(),
    nn.Linear(in_features=5, out_features=1),
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch, (X_batch, y_batch) in enumerate(train_loader):
        y_logits = model(X_batch).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_batch)
        train_loss += loss.item()
        acc = accuracy_fn(y_true=y_batch, y_pred=y_pred)
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    avg_train_loss = train_loss / (batch + 1)
    avg_train_acc = train_acc / (batch + 1)


    model.eval()
    test_loss = 0.0
    total_test_acc = 0.0
    for batch, (X_batch, y_batch) in enumerate(test_loader):
      with torch.inference_mode():
        test_logits = model(X_batch).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        loss = loss_fn(test_logits, y_batch)
        test_loss += loss.item()
        test_acc = accuracy_fn(y_true=y_batch, y_pred=test_pred)
        total_test_acc += test_acc
    avg_test_loss = test_loss / (batch + 1)
    avg_test_acc = total_test_acc / (batch + 1)

    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {avg_train_loss:.5f}, Acc: {avg_train_acc:.2f}% | Test Loss: {avg_test_loss:.5f}, Test Acc: {avg_test_acc:.2f}%")

