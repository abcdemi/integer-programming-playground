import pickle
from pyscipopt import Model, SCIP_PARAMSETTING
import torch

from gcn_solver import GCN, gcn_guided_solve

def get_scip_baseline_solution(instance_file):
    """
    Solves a knapsack instance with a basic SCIP B&B search to get a
    baseline node count for comparison.
    """
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    model = Model("KnapsackValidator")
    model.hideOutput()
    
    # --- ADDED: Make the comparison fair by disabling SCIP's advanced features ---
    # This forces SCIP to perform a more traditional Branch and Bound search,
    # similar to our custom solver.
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    # --------------------------------------------------------------------------
    
    num_items = instance['num_items']
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(num_items)}
    
    model.setObjective(sum(instance['values'][i] * x[i] for i in range(num_items)), "maximize")
    model.addCons(sum(instance['weights'][i] * x[i] for i in range(num_items)) <= instance['capacity'])
    
    model.optimize()
    
    optimal_profit = 0
    node_count = 0
    if model.getStatus() == "optimal":
        optimal_profit = model.getObjVal()
        # --- ADDED: Get the number of nodes SCIP explored ---
        node_count = model.getNNodes()
        print(f"SCIP found a provably optimal solution with profit: {optimal_profit}")
        print(f"Nodes explored by basic SCIP solver: {node_count}")
    else:
        print(f"SCIP could not find an optimal solution. Status: {model.getStatus()}")
        
    return optimal_profit, node_count

if __name__ == '__main__':
    INSTANCE_TO_TEST = 'knapsack_dataset/instance_0.pkl'
    
    # --- Run both solvers and compare ---
    print("--- Solving with GCN-Guided Solver ---")
    model = GCN(num_node_features=3)
    model.load_state_dict(torch.load('gcn_knapsack_model_v2.pth', weights_only=True))
    with open(INSTANCE_TO_TEST, 'rb') as f:
        test_instance = pickle.load(f)
    gcn_profit, gcn_nodes = gcn_guided_solve(test_instance, model)

    print("\n--- Solving with SCIP Baseline Solver ---")
    scip_profit, scip_nodes = get_scip_baseline_solution(INSTANCE_TO_TEST)
    
    print("\n" + "="*25)
    print("      FINAL COMPARISON")
    print("="*25)
    print(f"GCN Solver Result:  Profit={gcn_profit}, Nodes={gcn_nodes}")
    print(f"SCIP Solver Result: Profit={scip_profit}, Nodes={scip_nodes}")
    print("="*25)

    if gcn_profit == scip_profit:
        if gcn_nodes < scip_nodes:
            print("\nSUCCESS! The GCN solver found the optimal solution and was MORE EFFICIENT.")
        elif gcn_nodes == scip_nodes:
            print("\nSUCCESS! The GCN solver found the optimal solution with similar efficiency.")
        else:
            print("\nSUCCESS! The GCN solver found the optimal solution but was LESS EFFICIENT.")
    else:
        print("\nNOTE: The GCN solver did not find the optimal solution.")