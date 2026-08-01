import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv("student_performance.csv")
train_loss_values = []
test_loss_values = []
epoch_count = []
epoch = 0
x = df["study_hours"]
y = df["exam_score"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.8,random_state=42)
x_train = torch.tensor(x_train.values,dtype=torch.float32)
x_test = torch.tensor(x_test.values,dtype=torch.float32)
y_train = torch.tensor(y_train.values,dtype=torch.float32)
y_test = torch.tensor(y_test.values,dtype=torch.float32)

class simpleregression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float32),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float32),requires_grad=True)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)
model_0 = simpleregression()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)

epoch = 1000
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epoch):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred,y_test)
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
plt.plot(epoch_count,train_loss_values,c="b",label="Train")
plt.plot(epoch_count,test_loss_values, c="r",label="Test")
plt.ylabel("TIME")
plt.xlabel("LOSS")
plt.legend()
plt.show()

while True:
    personne = float(input("Enter your study hours: "))
    personne_t = torch.tensor(personne, dtype=torch.float32)

    with torch.inference_mode():
       conclusion = model_0(personne_t)
       if conclusion > 100:
           print("Solution:",100)
       if conclusion <= 0:
           print("Solution:",0)
       else:
           print(int(conclusion.item()))