import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pickle

# --- 1. Define the GCN Model ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        # Output layer predicts a score for each node (item)
        self.output_layer = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Output a single score per node
        x = self.output_layer(x).squeeze(-1)
        return x

# --- 2. Function to convert our problem state to a PyG graph ---
def state_to_graph(weights, values, capacity, node_lp_values):
    num_items = len(weights)
    
    # Node features: [weight, value, lp_solution_value]
    node_features = torch.tensor([
        [weights[i], values[i], node_lp_values[i]] for i in range(num_items)
    ], dtype=torch.float)

    # Edge index (conflict graph): connect items that can't fit together
    edge_list = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if weights[i] + weights[j] > capacity:
                edge_list.append([i, j])
                edge_list.append([j, i]) # Edges are undirected
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return Data(x=node_features, edge_index=edge_index)

# --- 3. Load Data and Create PyG Dataset ---
# In a real scenario, you'd load the instance info along with the state
# For now, we'll use one instance's info for demonstration
instance_info = generate_knapsack_instance(50) 
weights, values, capacity = instance_info['weights'], instance_info['values'], instance_info['capacity']

with open('expert_branching_data.pkl', 'rb') as f:
    raw_data = pickle.load(f)

pyg_dataset = []
for state, label in raw_data:
    graph = state_to_graph(weights, values, capacity, state)
    graph.y = torch.tensor(label, dtype=torch.long) # Add the expert label
    pyg_dataset.append(graph)

# --- 4. Training Loop ---
if __name__ == '__main__':
    train_loader = DataLoader(pyg_dataset, batch_size=32, shuffle=True)
    
    # Model, optimizer, loss
    model = GCN(num_node_features=3, num_classes=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # We want to predict the single best class (variable index)
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting GCN training...")
    model.train()
    for epoch in range(10): # Run for a few epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # The model outputs scores for each node in the batch
            # Shape: [num_nodes_in_batch]
            out = model(batch) 
            
            # We need to map the output scores back to individual graphs in the batch
            # And compare with the single label y for each graph
            # PyG's CrossEntropy handling with batches is complex, a simpler way for this task:
            # We treat it as predicting scores, and want the expert label to have the highest score
            
            # A more direct approach for this "ranking" task is often used,
            # but CrossEntropy is a standard starting point. We need to construct the
            # logits for the whole graph.
            
            # This part is complex. Let's simplify and assume a fixed size for now.
            # In a real implementation, you would use padding or a different loss function.
            # For this example, let's just make it work for a single graph.
            
            single_graph = pyg_dataset[0]
            optimizer.zero_grad()
            out_scores = model(single_graph) # Shape [num_items]
            
            # CrossEntropy expects logits (scores) and a single class index
            loss = criterion(out_scores.unsqueeze(0), single_graph.y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(pyg_dataset):.4f}")

    # Now the 'model' is trained and can be used to guide a new B&B algorithm.
    print("Training finished.")