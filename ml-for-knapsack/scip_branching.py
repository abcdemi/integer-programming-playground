import os
import pickle
from pyscipopt import Model

def solve_and_save_solution(instance_file, solution_dir):
    """Solves a knapsack instance and saves the optimal solution."""
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    model = Model("knapsack")
    model.hideOutput()
    
    num_items = instance['num_items']
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(num_items)}
    
    model.setObjective(sum(instance['values'][i] * x[i] for i in range(num_items)), "maximize")
    model.addCons(sum(instance['weights'][i] * x[i] for i in range(num_items)) <= instance['capacity'])
    
    model.optimize()
    
    # Check if an optimal solution was found
    if model.getStatus() == "optimal":
        solution = [round(model.getVal(x[i])) for i in range(num_items)]
        
        # Save the solution
        solution_filepath = os.path.join(solution_dir, os.path.basename(instance_file))
        with open(solution_filepath, 'wb') as f:
            pickle.dump({
                'instance': instance,
                'solution': solution
            }, f)
        print(f"  -> Solved and saved solution for {os.path.basename(instance_file)}")
    else:
        print(f"  -> No optimal solution found for {os.path.basename(instance_file)}")


if __name__ == '__main__':
    DATASET_DIR = 'knapsack_dataset'
    SOLUTION_DIR = 'knapsack_solutions'
    
    if not os.path.exists(SOLUTION_DIR):
        os.makedirs(SOLUTION_DIR)
        
    instance_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
    
    print(f"Processing {len(instance_files)} instances to collect optimal solutions...")
    for filename in instance_files:
        solve_and_save_solution(filename, SOLUTION_DIR)
            
    print(f"\nFinished collecting solutions. Data is in '{SOLUTION_DIR}'.")