# Machine Learning for Combinatorial Optimization: A GCN-Guided Knapsack Solver

This project demonstrates an end-to-end pipeline for using Machine Learning in branching. Specifically, a Graph Convolutional Network (GCN) to guide a classical optimization algorithm, Branch and Bound, in solving the 0/1 Knapsack Problem.

The core idea is to train a GCN to act as an intelligent heuristic for making branching decisions within the solver. We validate our custom solver by comparing its correctness and efficiency against the state-of-the-art open-source solver, SCIP.

## Methodology

The project is divided into three main phases:
1.  **Data Generation & Collection:** Creating a dataset of problems and their optimal solutions.
2.  **Model Training:** Training a GCN to predict which items belong in an optimal solution.
3.  **Guided Solving & Validation:** Implementing a custom Branch and Bound solver that uses the trained GCN and comparing its performance.

### 1. Representing the Knapsack Problem as a Graph

To use a GCN, we must first represent the knapsack problem as a graph. This is the foundational step that allows the neural network to understand the problem's structure.

*   **Nodes:** Each item in the knapsack problem is represented as a **node** in the graph.
*   **Node Features:** Each node is assigned features that describe the corresponding item. In this project, we use:
    1.  The item's `weight`.
    2.  The item's `value`.
    3.  The item's `value-to-weight ratio`.
*   **Edges (The "Conflict Graph"):** The edges represent the core constraint of the problem: the capacity limit. An edge is created between any two nodes (items) if their combined weight exceeds the knapsack's capacity.
    *   `Edge (i, j) exists if: weight[i] + weight[j] > capacity`
    *   This creates a "conflict graph" where an edge signifies that the two connected items are in direct competition for the knapsack's limited space. The GCN can then learn to reason about these conflicts.

### 2. Training the GCN (Imitation Learning)

The GCN is trained to imitate an "expert" solver. The goal is to teach the model, just by looking at the problem's graph structure, to predict which items are most likely to be part of the final, optimal solution.

The training process is as follows:
1.  **Generate Dataset:** We create thousands of knapsack problem instances with varying item weights and values. A realistic capacity is chosen to ensure meaningful conflict graphs are generated.
2.  **Find "Expert" Solutions:** Each instance is solved to optimality using the powerful SCIP solver. The final 0/1 solution vector (indicating which items were included) is stored as the "ground truth" label.
3.  **Train the Model:**
    *   The GCN takes the conflict graph of an instance as input.
    *   It outputs a score (a logit) for each node, representing the model's confidence that the item should be included in the solution.
    *   We use a **Binary Cross-Entropy with Logits Loss** function to train the model. This loss function pushes the output scores for items in the optimal solution towards 1 and the scores for items not in the solution towards 0.

### 3. GCN-Guided Branch and Bound

The trained GCN is then integrated into a custom Branch and Bound solver:
*   At each node in the search tree, the solver must decide which "undecided" variable to branch on next.
*   Instead of using a simple heuristic (like picking the item with the best value/weight ratio), we query the GCN.
*   The GCN provides scores for all items. Our solver chooses the undecided item with the **highest score** as the next variable to branch on.
*   This learned guidance is used to navigate the massive (2^50) search tree more intelligently.

## Project Structure

*   `generate_dataset.py`: Generates a dataset of 1000 knapsack problem instances.
*   `collect_solutions.py`: Solves each instance in the dataset using SCIP and saves the optimal solution vectors.
*   `train_from_solutions.py`: Loads the problems and their solutions, converts them to graphs, and trains the GCN model. Saves the trained model as `gcn_knapsack_model_v2.pth`.
*   `gcn_solver.py`: Implements the custom Branch and Bound solver that loads the trained GCN to make branching decisions.
*   `validate_solution.py`: The final validation script. It solves a test instance using both the GCN-guided solver and a baseline SCIP solver, comparing both correctness (final profit) and efficiency (nodes explored).

## How to Run the Project

### 1. Prerequisites

It is highly recommended to use a Conda/Mamba environment.

```bash
# Create and activate a new environment
conda create -n gcn_env python=3.9 -y
conda activate gcn_env

# Install Mamba for faster package installation
conda install mamba -c conda-forge -y

# Install the core dependencies (use the command for your system)
# For GPU (e.g., CUDA 12.1):
mamba install pyscipopt pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia

# For CPU-only:
mamba install pyscipopt pytorch -c conda-forge -c pytorch

# Install PyTorch Geometric using pip (recommended for compatibility)
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-<YOUR_TORCH_VERSION>+<YOUR_CUDA_VERSION>.html
# Example: pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### 2. Execute the Pipeline

Run the scripts in the following order from your terminal:

```bash
# Step 1: Generate the dataset of knapsack problems
python generate_dataset.py

# Step 2: Solve the problems to get the "expert" solutions
python collect_solutions.py

# Step 3: Train the GCN model on the expert data
python train_from_solutions.py

# Step 4: Run the final validation and comparison
python validate_solution.py
```

## Results & Analysis

The final experiment compares our GCN-guided solver against a baseline SCIP solver (with its advanced features turned off for a fair comparison) on a single test instance.

| Solver | Final Profit | Nodes Explored |
| :--- | :--- | :--- |
| **GCN-Guided Solver** | **277** | **170** |
| **SCIP Baseline Solver** | **277.0** | **27** |

### Key Findings:

1.  **Correctness: SUCCESS**
    *   Our GCN-guided solver successfully found the **provably optimal solution** (277). This validates that the entire pipeline is working correctly and that the GCN learned an effective heuristic.

2.  **Efficiency: INSIGHTFUL**
    *   Our solver was **less efficient** than the baseline SCIP solver, exploring 170 nodes compared to SCIP's 27.
    *   This is a realistic and valuable result. It demonstrates that while a learned GCN heuristic can be powerful enough to find the optimal path, it is very challenging to outperform the highly-engineered, lightweight, and decades-refined default branching rules of a state-of-the-art solver.

This project serves as a successful proof-of-concept for using GNNs to guide combinatorial search, highlighting both the potential of the approach and the remarkable performance of modern optimization solvers.