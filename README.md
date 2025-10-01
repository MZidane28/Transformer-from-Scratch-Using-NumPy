# Transformer Implementation from Scratch with NumPy

This project contains an implementation of a **decoder-only Transformer** architecture from scratch using only **NumPy**. This code was created as part of an individual assignment to understand the mathematical and operational workflow behind the Transformer model without using deep learning libraries like PyTorch or TensorFlow.

This implementation covers all core components of the Transformer, including:
- Token & Positional Embedding
- Multi-Head Self-Attention
- Causal Masking
- Feed-Forward Network
- Layer Normalization & Residual Connections
- Bonus Feature: Attention Matrix Visualization

## Dependencies

This program has a few dependencies that need to be installed.

- **Required:**
  - `numpy`: For all mathematical and matrix operations.

- **Optional (for Visualization):**
  - `matplotlib`: For creating plots.
  - `seaborn`: For generating a better-looking attention visualization heatmap.

You can install all dependencies with a single command using `pip`:
```bash
pip install numpy matplotlib seaborn
```

## How to Run the Program

1.  **Clone the Repository (Optional)**
    If you are downloading the code from GitHub, clone this repository to your local machine.
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [REPOSITORY_FOLDER_NAME]
    ```

2.  **Run the Script**
    Execute the main Python file from your terminal. Assuming your file is named `transformer_attention_viz.py`.

    ```bash
    python transformer_attention_viz.py
    ```

3.  **Expected Output**
    After running the script, the program will:
    - Print the shape verification for the input tensor, output logits, and attention weights to the terminal.
    - Print the verification result for the `softmax` function.
    - Display a pop-up plot window containing the **attention visualization heatmap** from the model's final layer.

    Example terminal output:
    ```
    --- Tensor Shape Verification ---
    Input Tokens Shape: (1, 15)
    Output Logits Shape: (1, 15, 1000)
    Attention Weights Shape: (1, 8, 15, 15)

    ...

    --- Displaying Attention Visualization ---
    ```
    A plot window will then appear.
---
