import heapq
import torch
import pickle

# We need the GCN class definition and the graph creation utility
# so we can load the model and process new instances.
from gcn_training import GCN, instance_to_graph 

def gcn_guided_solve(instance, gcn_model):
    """
    Solves a 0/1 knapsack problem using a Branch and Bound algorithm
    guided by a trained Graph Convolutional Network.
    """
    gcn_model.eval() # Set the model to evaluation mode (no gradients needed)

    # The priority queue will store tuples of (-bound, fixed_items_dict)
    # We use a negative bound because heapq is a min-heap, and we want to
    # explore the node with the highest potential profit first.
    pq = [(-calculate_bound({}, instance), {})] 
    
    best_profit = 0
    best_solution = {}

    while pq:
        _, current_fixed_items = heapq.heappop(pq)
        
        # --- GCN-GUIDED VARIABLE SELECTION (The "Smart" Part) ---
        
        # We don't have a solution yet, so we pass None
        graph_state = instance_to_graph(instance, None)
        if graph_state is None:
            # Fallback for the rare case a test instance has no edges
            print("Warning: Test instance has no graph edges. Cannot use GCN.")
            return 0 # Or implement a simple heuristic here

        with torch.no_grad():
            # Get the GCN's prediction for how likely each item is to be in the final solution
            scores = gcn_model(graph_state)
        
        # Find the best *undecided* item to branch on, based on the GCN's scores
        branch_var = -1
        max_score = -float('inf')
        for i in range(instance['num_items']):
            if i not in current_fixed_items: # If the item's fate is not yet decided
                if scores[i] > max_score:
                    max_score = scores[i]
                    branch_var = i

        # If branch_var is -1, it means all variables have been fixed. We have a leaf node.
        if branch_var == -1:
            current_profit = calculate_profit(current_fixed_items, instance)
            if current_profit > best_profit:
                best_profit = current_profit
                best_solution = current_fixed_items
            continue

        # --- BRANCHING ---
        # Explore two branches for the chosen variable: set it to 1 (include) and 0 (exclude)
        for val in [1, 0]:
            new_fixed = current_fixed_items.copy()
            new_fixed[branch_var] = val
            
            # Pruning: Only explore this branch if it's feasible and promising
            if is_feasible(new_fixed, instance):
                bound = calculate_bound(new_fixed, instance)
                if bound > best_profit:
                    heapq.heappush(pq, (-bound, new_fixed))
                    
    print(f"GCN-guided solver found best profit: {best_profit}")
    return best_profit

# --- Helper functions for the B&B solver ---

def is_feasible(fixed_items, instance):
    """Check if a partial solution is within the knapsack's capacity."""
    current_weight = sum(instance['weights'][i] for i, v in fixed_items.items() if v == 1)
    return current_weight <= instance['capacity']

def calculate_profit(fixed_items, instance):
    """Calculate the profit of a (partial or complete) solution."""
    return sum(instance['values'][i] for i, v in fixed_items.items() if v == 1)

def calculate_bound(fixed_items, instance):
    """
    Calculates an optimistic upper bound on the best possible solution from this node.
    This is a simple greedy relaxation used for pruning.
    """
    profit = calculate_profit(fixed_items, instance)
    weight = sum(instance['weights'][i] for i, v in fixed_items.items() if v == 1)
    remaining_capacity = instance['capacity'] - weight
    
    bound = profit
    
    # Create a list of (ratio, weight) for undecided items
    undecided_items = []
    for i in range(instance['num_items']):
        if i not in fixed_items:
            ratio = instance['values'][i] / (instance['weights'][i] + 1e-6)
            undecided_items.append((ratio, instance['weights'][i]))
    
    # Sort by value/weight ratio in descending order
    undecided_items.sort(key=lambda x: x[0], reverse=True)
    
    # Fill remaining capacity greedily with fractions of items
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
    model.load_state_dict(torch.load('gcn_knapsack_model_v2.pth'))
    
    # Load a test instance to solve (let's use the first one from our dataset)
    with open('knapsack_dataset/instance_0.pkl', 'rb') as f:
        test_instance = pickle.load(f)

    print("\n--- Solving a new instance with the GCN-Guided Solver ---")
    gcn_guided_solve(test_instance, model)