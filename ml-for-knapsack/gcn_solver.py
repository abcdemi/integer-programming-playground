import heapq
import torch
import pickle

# --- We copy the GCN class definition here so this script is self-contained ---
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data

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

# --- We copy and FIX the graph utility function here ---
def instance_to_graph(instance, solution=None): # Default solution to None
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
    
    if not edge_list:
        return None
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # --- THIS IS THE CRITICAL FIX ---
    # Only create the labels if a solution is actually provided.
    if solution is not None:
        labels = torch.tensor(solution, dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index, y=labels)
    else:
        # For solving, we don't have labels, so we return a graph without them.
        return Data(x=node_features, edge_index=edge_index)
    # --------------------------------

def gcn_guided_solve(instance, gcn_model):
    """
    Solves a 0/1 knapsack problem using a Branch and Bound algorithm
    guided by a trained Graph Convolutional Network.
    """
    gcn_model.eval()

    pq = [(-calculate_bound({}, instance), {})] 
    
    best_profit = 0

    while pq:
        _, current_fixed_items = heapq.heappop(pq)
        
        # This call will now work correctly
        graph_state = instance_to_graph(instance) 
        if graph_state is None:
            print("Warning: Test instance has no graph edges. Cannot use GCN.")
            # In a real system, you'd fall back to a basic heuristic here.
            # For this example, we'll just stop.
            return 0 

        with torch.no_grad():
            scores = gcn_model(graph_state)
        
        branch_var = -1
        max_score = -float('inf')
        for i in range(instance['num_items']):
            if i not in current_fixed_items:
                if scores[i] > max_score:
                    max_score = scores[i]
                    branch_var = i

        if branch_var == -1:
            current_profit = calculate_profit(current_fixed_items, instance)
            if current_profit > best_profit:
                best_profit = current_profit
            continue

        for val in [1, 0]:
            new_fixed = current_fixed_items.copy()
            new_fixed[branch_var] = val
            
            if is_feasible(new_fixed, instance):
                bound = calculate_bound(new_fixed, instance)
                if bound > best_profit:
                    heapq.heappush(pq, (-bound, new_fixed))
                    
    print(f"GCN-guided solver found best profit: {best_profit}")
    return best_profit

# --- Helper functions for the B&B solver (unchanged) ---
def is_feasible(fixed_items, instance):
    current_weight = sum(instance['weights'][i] for i, v in fixed_items.items() if v == 1)
    return current_weight <= instance['capacity']

def calculate_profit(fixed_items, instance):
    return sum(instance['values'][i] for i, v in fixed_items.items() if v == 1)

def calculate_bound(fixed_items, instance):
    profit = calculate_profit(fixed_items, instance)
    weight = sum(instance['weights'][i] for i, v in fixed_items.items() if v == 1)
    remaining_capacity = instance['capacity'] - weight
    bound = profit
    undecided_items = []
    for i in range(instance['num_items']):
        if i not in fixed_items:
            ratio = instance['values'][i] / (instance['weights'][i] + 1e-6)
            undecided_items.append((ratio, instance['weights'][i]))
    undecided_items.sort(key=lambda x: x[0], reverse=True)
    for ratio, item_weight in undecided_items:
        if remaining_capacity <= 0:
            break
        fill_weight = min(item_weight, remaining_capacity)
        bound += fill_weight * ratio
        remaining_capacity -= fill_weight
    return bound

if __name__ == '__main__':
    # Load the trained model
    model = GCN(num_node_features=3)
    # Addressing the warning by setting weights_only=True
    model.load_state_dict(torch.load('gcn_knapsack_model_v2.pth', weights_only=True))
    
    # Load a test instance to solve
    with open('knapsack_dataset/instance_0.pkl', 'rb') as f:
        test_instance = pickle.load(f)

    print("\n--- Solving a new instance with the GCN-Guided Solver ---")
    gcn_guided_solve(test_instance, model)