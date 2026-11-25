import os
import random
lazy import time
import torch
import torch.nn as nn
import torch.optim as optim

X_data = []
Y_data = []

for i in range(10000):
  a = random.uniform(-100, 100)
  b = random.uniform(-100, 100)
  X_data.append([a, b])
  Y_data.append([a + b])
  print(f"a{a:.4f} + {b:.4f} = {a + b:.4f}")


X = torch.tensor(X_data, dtype=torch.float32)
Y = torch.tensor(Y_data, dtype=torch.float32)

class AIWalaCalculator(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(in_features=2, out_features=1)
 
  def forward(self, x):
    return self.linear(x)

mymodel = AIWalaCalculator()

criterion = nn.MSELoss()
optimizer = optim.Adam(mymodel.parameters(), lr=0.01)

print("Calculator Wala AI Ko Train Karne Ki Prakriya.")

epochs = 20000

t = 0
t += time.time()

for epoch in range(epochs):
  predictions = mymodel(X)
  loss = criterion(predictions, Y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_ka_dimag.pth")

torch.save(mymodel.state_dict(), model_path)
print("=========================")
print(f"AI KA DIMAG {t:.2f} nanoseconds mein SEEKH CHUKA HAI KINDLY NOTE. <3")
print("=========================")