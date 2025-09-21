import os
import pickle
# ADD THIS IMPORT
from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING

training_data = []

class ExpertBranchingRecorder(Branchrule):
    def __init__(self, model_vars):
        self.model_vars = model_vars

    def branchexeclp(self):
        # ... (The rest of this class stays exactly the same)
        candidates, _, _, _, _ = self.model.getLPBranchCands()
        if not candidates:
            return {'result': SCIP_RESULT.DIDNOTRUN}
        _, _, _, best_cand_idx = self.model.getStrongbranchs(candidates)
        expert_choice_var = candidates[best_cand_idx]
        expert_label = -1
        for i, var in enumerate(self.model_vars):
            if var.name == expert_choice_var.name:
                expert_label = i
                break
        if expert_label != -1:
            lp_solution_values = [self.model.getVal(var) for var in self.model_vars]
            training_data.append((lp_solution_values, expert_label))
        return {'result': SCIP_RESULT.DIDNOTRUN}


def process_instance(instance_file):
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    model = Model("knapsack")
    model.hideOutput()

    # --- ADD THESE CRITICAL LINES TO DISABLE ADVANCED FEATURES ---
    print(f"Processing {instance_file} with SCIP (advanced features OFF)...")
    # Turn off all presolving routines
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    # Turn off all heuristics
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    # Turn off all cutting plane separators
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    # -----------------------------------------------------------
    
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(instance['num_items'])}
    model.setObjective(sum(instance['values'][i] * x[i] for i in range(instance['num_items'])), "maximize")
    model.addCons(sum(instance['weights'][i] * x[i] for i in range(instance['num_items'])) <= instance['capacity'])
    
    model_vars_list = [x[i] for i in range(instance['num_items'])]
    recorder = ExpertBranchingRecorder(model_vars_list)
    model.includeBranchrule(recorder, "ExpertRecorder", "Records strong branching", 999999, -1, 1)
    
    model.optimize()

# --- Main execution block (no changes here) ---
if __name__ == '__main__':
    DATASET_DIR = 'knapsack_dataset'
    instance_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
    
    for filename in instance_files:
        process_instance(filename)
            
    with open('expert_branching_data_scip.pkl', 'wb') as f:
        pickle.dump(training_data, f)
            
    print(f"\nCollected a total of {len(training_data)} training samples using SCIP.")