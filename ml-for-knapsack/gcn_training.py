import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# --- GCN Model Definition ---
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

# --- Graph Conversion Utility ---
def instance_to_graph(instance, solution):
    weights, values, capacity = instance['weights'], instance['values'], instance['capacity']
    num_items = len(weights)
    
    node_features = torch.tensor([
        [weights[i], values[i], values[i] / (weights[i] + 1e-6)] for i in range(num_items)
    ], dtype=torch.float)

    edge_list = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if weights[i] + weights[j] > capacity:
                edge_list.extend([[i, j], [j, i]])
    
    # Robustness check: if a graph has no edges, we cannot process it.
    if not edge_list:
        return None
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    labels = torch.tensor(solution, dtype=torch.float)
    
    return Data(x=node_features, edge_index=edge_index, y=labels)

if __name__ == '__main__':
    SOLUTION_DIR = 'knapsack_solutions'
    solution_files = [os.path.join(SOLUTION_DIR, f) for f in os.listdir(SOLUTION_DIR)]

    pyg_dataset = []
    for sol_file in solution_files:
        with open(sol_file, 'rb') as f:
            data = pickle.load(f)
        graph = instance_to_graph(data['instance'], data['solution'])
        if graph is not None:
            pyg_dataset.append(graph)
    
    print(f"Successfully created {len(pyg_dataset)} graph instances for training.")
    
    if not pyg_dataset:
        print("Error: No valid graph instances were created. Exiting.")
        exit()

    train_loader = DataLoader(pyg_dataset, batch_size=16, shuffle=True)
    
    model = GCN(num_node_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Starting GCN training...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out_scores = model(batch)
            loss = criterion(out_scores, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'gcn_knapsack_model_v2.pth')
    print("Training finished. Model saved to gcn_knapsack_model_v2.pth")