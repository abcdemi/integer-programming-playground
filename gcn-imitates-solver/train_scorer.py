import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# --- GCN Model Definition (The architecture is the same) ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.output_layer = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.output_layer(x).squeeze(-1)

# --- Graph Utility ---
def state_to_graph(instance, lp_solution_values, expert_scores):
    weights, values, capacity = instance['weights'], instance['values'], instance['capacity']
    num_items = len(weights)
    
    # Node features now include the dynamic LP value from the solver state
    node_features = torch.tensor([
        [weights[i], values[i], lp_solution_values[i]] for i in range(num_items)
    ], dtype=torch.float)

    edge_list = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if weights[i] + weights[j] > capacity:
                edge_list.extend([[i, j], [j, i]])
    
    if not edge_list:
        return None
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # The label is the array of expert scores
    labels = torch.tensor(expert_scores, dtype=torch.float)
    
    return Data(x=node_features, edge_index=edge_index, y=labels)

if __name__ == '__main__':
    # Load the expert score data you just collected
    with open('expert_scores_scip.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # We need a sample instance to get the static weights/values
    with open('knapsack_dataset/instance_0.pkl', 'rb') as f:
        sample_instance = pickle.load(f)

    pyg_dataset = []
    for lp_values, scores in raw_data:
        graph = state_to_graph(sample_instance, lp_values, scores)
        if graph is not None:
            pyg_dataset.append(graph)
    
    print(f"Successfully created {len(pyg_dataset)} graph instances for training.")
    
    if not pyg_dataset:
        print("Error: No valid graph instances were created. Exiting.")
        exit()

    train_loader = DataLoader(pyg_dataset, batch_size=16, shuffle=True)
    
    model = GCN(num_node_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # --- The Loss Function is now Mean Squared Error ---
    criterion = torch.nn.MSELoss()

    print("Starting GCN training to predict branching scores...")
    model.train()
    for epoch in range(20): # Train for a few more epochs for regression
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out_scores = model(batch)
            loss = criterion(out_scores, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss (MSE): {total_loss / len(train_loader):.6f}")

    # Save the new, more powerful model
    torch.save(model.state_dict(), 'gcn_knapsack_scorer.pth')
    print("Training finished. Scorer model saved to gcn_knapsack_scorer.pth")