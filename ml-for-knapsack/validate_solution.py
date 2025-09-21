import pickle
from pyscipopt import Model

def get_scip_optimal_solution(instance_file):
    """
    Solves a single knapsack instance with SCIP to find the ground truth
    optimal value.
    """
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    # Set up the SCIP model. We are NOT disabling any features this time.
    # We want SCIP to use its full power to solve this as fast as possible.
    model = Model("KnapsackValidator")
    model.hideOutput() # We only care about the final result
    
    num_items = instance['num_items']
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(num_items)}
    
    model.setObjective(sum(instance['values'][i] * x[i] for i in range(num_items)), "maximize")
    model.addCons(sum(instance['weights'][i] * x[i] for i in range(num_items)) <= instance['capacity'])
    
    # Solve the problem
    model.optimize()
    
    optimal_profit = 0
    if model.getStatus() == "optimal":
        optimal_profit = model.getObjVal()
        print(f"SCIP found a provably optimal solution with profit: {optimal_profit}")
    else:
        print(f"SCIP could not find an optimal solution. Status: {model.getStatus()}")
        
    return optimal_profit

if __name__ == '__main__':
    INSTANCE_TO_TEST = 'knapsack_dataset/instance_0.pkl'
    
    print(f"--- Validating solution for {INSTANCE_TO_TEST} ---")
    
    # Get the ground truth from SCIP
    scip_profit = get_scip_optimal_solution(INSTANCE_TO_TEST)
    
    # The result from your custom solver
    gcn_profit = 277
    
    print("\n--- Comparison ---")
    print(f"GCN-Guided Solver Profit: {gcn_profit}")
    print(f"SCIP Optimal Profit:      {scip_profit}")
    
    if gcn_profit == scip_profit:
        print("\nSUCCESS! The GCN-guided solver found the provably optimal solution.")
    else:
        print(f"\nNOTE: The GCN-guided solver found a good, but suboptimal, solution.")
        print(f"Optimality Gap: {((scip_profit - gcn_profit) / scip_profit * 100):.2f}%")