import os
import pickle
from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING

# This list will store our training data: (graph_state, expert_choice)
training_data = []

class ExpertBranchingRecorder(Branchrule):
    """
    A custom SCIP branching rule to record expert decisions.
    This rule observes strong branching but doesn't enforce the branch itself,
    letting SCIP's default behavior continue.
    """
    def __init__(self, model_vars):
        self.model_vars = model_vars
        self.data_limit = 50 # Stop collecting after this many samples per instance

    def branchexeclp(self):
        if len(training_data) >= self.data_limit:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        # --- 1. Get branching candidates ---
        # These are the variables with fractional values in the LP relaxation
        candidates, _, _, _, _ = self.model.getLPBranchCands()
        
        if not candidates:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        # --- 2. Perform Strong Branching via SCIP's built-in function ---
        # This is much easier than the manual Gurobi implementation
        down_bounds, up_bounds, _, best_cand_idx = self.model.getStrongbranchs(candidates)
        
        # The best candidate is the one SCIP's strong branching chose
        expert_choice_var = candidates[best_cand_idx]
        
        # Find the index of this variable in our original list
        expert_label = -1
        for i, var in enumerate(self.model_vars):
            if var.name == expert_choice_var.name:
                expert_label = i
                break
        
        if expert_label != -1:
            # --- 3. Get the current state (LP solution) ---
            lp_solution_values = [self.model.getVal(var) for var in self.model_vars]
            
            # --- 4. Save the data point ---
            state_representation = lp_solution_values
            
            training_data.append((state_representation, expert_label))
            print(f"Collected data point #{len(training_data)}. Expert choice: variable {expert_label}")
        
        # IMPORTANT: We tell SCIP we haven't actually performed a branching
        # We are just observing. SCIP will then proceed with its own branching logic.
        return {'result': SCIP_RESULT.DIDNOTRUN}

def solve_with_scip_recorder(instance_file):
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)

    weights = instance['weights']
    values = instance['values']
    capacity = instance['capacity']
    n = instance['num_items']

    # --- Model Setup ---
    model = Model("knapsack")
    model.hideOutput() # Suppress solver output

    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(n)}
    
    model.setObjective(sum(values[i] * x[i] for i in range(n)), "maximize")
    model.addCons(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    # --- Include the Custom Branching Rule ---
    model_vars_list = [x[i] for i in range(n)]
    recorder = ExpertBranchingRecorder(model_vars_list)
    model.includeBranchrule(
        recorder,
        "ExpertRecorder",
        "Python branching rule to record strong branching decisions",
        priority=999999, # High priority to be called first
        maxdepth=-1,
        maxbounddist=1
    )

    # --- Optimize ---
    model.optimize()

# --- Main script to process the dataset ---
if __name__ == '__main__':
    DATASET_DIR = 'knapsack_dataset'
    
    # Generate data if the directory doesn't exist
    if not os.path.exists(DATASET_DIR):
        print("Dataset directory not found. Please run the data generation script first.")
    else:
        instance_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
        
        for filename in instance_files[:5]: # Process first 5 for demonstration
            print(f"Processing {filename} with SCIP...")
            # Reset the data limit for each instance if desired, or use a global limit
            solve_with_scip_recorder(filename)
            
        # Save the collected training data
        with open('expert_branching_data_scip.pkl', 'wb') as f:
            pickle.dump(training_data, f)
            
        print(f"\nCollected a total of {len(training_data)} training samples using SCIP.")