import os
import pickle
import numpy as np
from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING

training_data = []

class ExpertScoreRecorder(Branchrule):
    def __init__(self, model_vars):
        self.model_vars = model_vars

    def branchexeclp(self, allowaddcons):
        # Get the list of variables with fractional values (our candidates)
        candidates, *_ = self.model.getLPBranchCands()
        
        if not candidates:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        # The "state" is the solution to the LP relaxation at this node
        lp_solution_values = [self.model.getVal(var) for var in self.model_vars]
        
        # Initialize an array to store the score for every variable
        expert_scores = np.zeros(len(self.model_vars))
        
        current_lp_obj = self.model.getLPObjVal()

        for cand_var in candidates:
            # Get the objective values for branching down (to 0) and up (to 1)
            down_obj, up_obj, *_ = self.model.getVarStrongbranch(cand_var, -1)

            # Calculate the score. The product of degradations is a strong heuristic.
            # Add a small epsilon for numerical stability.
            score = (current_lp_obj - down_obj + 1e-6) * (current_lp_obj - up_obj + 1e-6)
            
            # Find the original index of this candidate variable
            for i, var in enumerate(self.model_vars):
                if var.name == cand_var.name:
                    expert_scores[i] = score
                    break
        
        # Save the state and the corresponding array of expert scores
        training_data.append((lp_solution_values, expert_scores))
        print(f"  -> Collected score set #{len(training_data)}")

        return {'result': SCIP_RESULT.DIDNOTRUN}

def process_instance(instance_file):
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    model = Model("knapsack")
    model.hideOutput()

    # We disable SCIP's advanced features to force it to branch,
    # which is necessary to trigger our callback and collect data.
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(instance['num_items'])}
    model.setObjective(sum(instance['values'][i] * x[i] for i in range(instance['num_items'])), "maximize")
    model.addCons(sum(instance['weights'][i] * x[i] for i in range(instance['num_items'])) <= instance['capacity'])
    
    model_vars_list = [x[i] for i in range(instance['num_items'])]
    recorder = ExpertScoreRecorder(model_vars_list)
    model.includeBranchrule(recorder, "ExpertScoreRecorder", "Records strong branching scores", 999999, -1, 1)
    
    try:
        model.optimize()
    except Exception:
        print(f"  -> SCIP process finished for this instance.")

if __name__ == '__main__':
    DATASET_DIR = 'knapsack_dataset'
    instance_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
    
    print(f"Processing instances to collect strong branching scores...")
    # It's better to use fewer instances to start, as this is slow.
    for filename in instance_files[:100]:
        print(f"Processing {filename}...")
        process_instance(filename)
            
    with open('expert_scores_scip.pkl', 'wb') as f:
        pickle.dump(training_data, f)
            
    print(f"\nCollected a total of {len(training_data)} training samples (state-score pairs).")