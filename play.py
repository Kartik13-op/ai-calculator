import torch
import os
import torch.nn as nn

class AICalculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "ai_ka_dimag.pth")

model = AICalculator()
model.load_state_dict(torch.load(model_path))
model.eval()

print("Kripaya 2 sankhyaayein pesh kariye.")

while True:
    a = input("Pehli sankhya: ")
    if a.lower() == "exit":
        print("AI calculator ko band kiya ja raha hai. Alvida!")
        break
    b = input("Doosri sankhya: ")
    if b.lower() == "exit":
        print("AI calculator ko band kiya ja raha hai. Alvida!")
        break

    try:
        a = float(a)
        b = float(b)
        input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
        ai_prediction = model(input_tensor).item()
        actual_sum = a + b
        difference = abs(ai_prediction - actual_sum)

        print(f"    AI Ka Anuman          : {ai_prediction:.4f}")
        print(f"    Vastavik Yog          : {actual_sum:.4f}")
        print(f"    Antar                 : {difference:.6f}\n")

    except ValueError:
        print("Kripaya sahi sankhyaayein pravesh kariye.")




