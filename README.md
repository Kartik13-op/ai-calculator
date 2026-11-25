# Over-Engineered AI Calculator

A completely ridiculous, over-engineered linear regression model built with PyTorch that learns how to add two numbers through trial and error instead of using standard arithmetic.

Because why use `a + b` when you can burn compute cycles, optimize gradient descent, and watch an AI hallucinate basic math?

---

## Project Structure

```text
calcai/
│
├── train.py          # Generates training data and trains the PyTorch model
├── play.py           # Interactive inference loop to test the AI's math skills
└── ai_calculator.pth # Saved model weights (generated after running train.py)

```

---

## How It Works

1. **Data Generation (`train.py`)**: The script loops 10,000 times, picking random number pairs between `-100` and `100` and calculating their actual sums.
2. **Model Architecture**: Uses a simple PyTorch `nn.Linear(2, 1)` layer—literally implementing the linear regression equation:

$$\text{output} = (w_1 \cdot a) + (w_2 \cdot b) + \text{bias}$$


3. **Optimization**: Uses the **Adam optimizer** and **Mean Squared Error (MSE)** loss to iteratively tweak the weights until the AI figures out that both weights should be `1.0` and the bias should be `0.0`.
4. **Inference (`play.py`)**: Loads the saved weights from `ai_calculator.pth` and runs a continuous command-line loop comparing the AI's guess against real math.

---

## Getting Started

### Prerequisites

Make sure you have PyTorch installed in your Python environment:

```bash
pip install torch

```

### Step 1: Train the Model

Run the training script to generate data, train the linear regression model, and save the weights locally:

```bash
python train.py

```

### Step 2: Play with the AI Calculator

Fire up the interactive prompt to test out numbers and see the prediction error margin:

```bash
python play.py

```

Type two numbers separated by a space (e.g., `42 58`), or type `exit` to quit.

---
