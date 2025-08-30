import numpy as np
from scipy.optimize import linprog
import time
import pandas as pd
import random

# =============================================================================
# PART 1: SIMULATED ML MODEL AND BRANCHING STRATEGIES
# =============================================================================

class SimulatedBranchingModel:
    """
    A simulated ML model that mimics a learned policy.
    POLICY: Give a higher score to variables with a larger objective coefficient.
    This simulates the model learning that "high-profit" items are more
    impactful to make decisions on first.
    """
    def predict(self, features):
        # features is a list of [fractional_value, objective_coefficient]
        # We return the objective_coefficient as the "score".
        scores = [f[1] for f in features]
        return np.array(scores)

def select_most_fractional_variable(candidates, solution, **kwargs):
    """CONTROL GROUP: Standard 'most fractional' branching heuristic."""
    # Find the variable closest to 0.5
    fractional_parts = [abs(solution[idx] - 0.5) for idx in candidates]
    return candidates[np.argmin(fractional_parts)]

def select_ml_predicted_variable(candidates, solution, objective_coeffs, model):
    """EXPERIMENTAL GROUP: Branching heuristic guided by the simulated ML model."""
    # 1. Extract features for each candidate variable
    features = []
    for idx in candidates:
        frac_value = solution[idx]
        obj_coeff = objective_coeffs[idx]
        features.append([frac_value, obj_coeff])

    # 2. Get "predictions" (scores) from the model
    scores = model.predict(features)

    # 3. Choose the variable with the highest score
    best_candidate_index = np.argmax(scores)
    return candidates[best_candidate_index]


# =============================================================================
# PART 2: THE CORE BRANCH AND BOUND SOLVER
# =============================================================================

def branch_and_bound_solver(c, A_ub, b_ub, branching_strategy, ml_model=None):
    """
    A flexible B&B solver that accepts a branching strategy as an argument.
    """
    # We are maximizing, so we convert the objective for scipy's minimizer
    c_min = -np.array(c)
    num_vars = len(c)
    
    best_integer_solution = None
    best_obj_value = -np.inf
    nodes_explored = 0
    
    # Stack for nodes to explore. A node is defined by its variable bounds.
    # Bounds are tuples of (min, max) for each variable.
    initial_bounds = tuple([(0, 1) for _ in range(num_vars)])
    stack = [(initial_bounds)]

    while stack:
        current_bounds = stack.pop()
        nodes_explored += 1
        
        # 1. Solve the LP relaxation
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=list(current_bounds), method='highs')

        # 2. Fathom (Prune) the node if it's not promising
        if not res.success or -res.fun <= best_obj_value:
            continue

        solution = res.x
        
        # 3. Check for an integer solution
        is_integer = np.all(np.isclose(solution, np.round(solution)))
        
        if is_integer:
            current_obj = np.dot(c, solution)
            if current_obj > best_obj_value:
                best_obj_value = current_obj
                best_integer_solution = solution
            continue

        # 4. Branching Step: Select a fractional variable to branch on
        fractional_indices = [i for i, val in enumerate(solution) if not np.isclose(val, np.round(val))]

        if not fractional_indices:
            continue
            
        # --- HERE THE BRANCHING STRATEGY IS CALLED ---
        strategy_kwargs = {
            'candidates': fractional_indices,
            'solution': solution,
            'objective_coeffs': c,
            'model': ml_model
        }
        branch_var_index = branching_strategy(**strategy_kwargs)
        # ---

        # 5. Create two new subproblems (nodes)
        val = solution[branch_var_index]
        
        # New problem 1: var <= 0
        new_bounds1 = list(current_bounds)
        new_bounds1[branch_var_index] = (0, 0)
        stack.append(tuple(new_bounds1))
        
        # New problem 2: var >= 1
        new_bounds2 = list(current_bounds)
        new_bounds2[branch_var_index] = (1, 1)
        stack.append(tuple(new_bounds2))

    return best_obj_value, nodes_explored


# =============================================================================
# PART 3: PROBLEM GENERATOR
# =============================================================================

def generate_knapsack_problem(num_items):
    """Generates a random 0-1 Knapsack Problem."""
    values = np.random.randint(10, 100, size=num_items)  # Objective coefficients
    weights = np.random.randint(5, 50, size=num_items)   # Constraint coefficients
    capacity = int(np.sum(weights) * 0.6)              # Knapsack capacity
    
    # Standard form: max c*x s.t. A*x <= b
    c = values
    A_ub = [weights]
    b_ub = [capacity]
    
    return c, A_ub, b_ub


# =============================================================================
# PART 4: EXPERIMENT RUNNER AND ANALYSIS
# =============================================================================

if __name__ == '__main__':
    print("Designing and running a self-contained experiment...")
    print("Objective: Compare a simulated ML branching model vs. a standard heuristic.\n")

    # --- Experiment Parameters ---
    NUM_PROBLEMS_TO_RUN = 20
    NUM_ITEMS_PER_PROBLEM = 30
    
    # Instantiate our simulated ML model
    simulated_ml_model = SimulatedBranchingModel()
    
    experiment_results = []

    for i in range(NUM_PROBLEMS_TO_RUN):
        print(f"--- Running on randomly generated problem #{i+1} ---")
        
        # Generate a new, unique problem for a fair comparison
        c, A_ub, b_ub = generate_knapsack_problem(NUM_ITEMS_PER_PROBLEM)

        # Run Control Group
        start_time_control = time.time()
        obj_control, nodes_control = branch_and_bound_solver(c, A_ub, b_ub, select_most_fractional_variable)
        time_control = time.time() - start_time_control
        
        # Run Experimental Group
        start_time_ml = time.time()
        obj_ml, nodes_ml = branch_and_bound_solver(c, A_ub, b_ub, select_ml_predicted_variable, ml_model=simulated_ml_model)
        time_ml = time.time() - start_time_ml
        
        # Store results
        experiment_results.append({
            'ProblemID': i + 1,
            'Control_Nodes': nodes_control,
            'ML_Nodes': nodes_ml,
            'Control_Time_s': time_control,
            'ML_Time_s': time_ml,
            'OptimalValue': obj_control
        })

    # --- Analyze and Display Results ---
    df = pd.DataFrame(experiment_results)
    
    # Calculate winner for each run based on node count
    df['Winner'] = np.where(df['ML_Nodes'] < df['Control_Nodes'], 'ML Model',
                           np.where(df['ML_Nodes'] > df['Control_Nodes'], 'Control', 'Tie'))

    print("\n\n" + "="*50)
    print("               EXPERIMENT RESULTS")
    print("="*50)
    print(df[['ProblemID', 'Control_Nodes', 'ML_Nodes', 'Winner']].to_string(index=False))
    print("-"*50)

    # Summary Statistics
    avg_nodes_control = df['Control_Nodes'].mean()
    avg_nodes_ml = df['ML_Nodes'].mean()
    
    ml_wins = (df['Winner'] == 'ML Model').sum()
    control_wins = (df['Winner'] == 'Control').sum()
    ties = (df['Winner'] == 'Tie').sum()

    print("\n--- SUMMARY STATISTICS ---")
    print(f"Average Nodes (Control Heuristic): {avg_nodes_control:.2f}")
    print(f"Average Nodes (Simulated ML Model): {avg_nodes_ml:.2f}\n")
    
    print(f"Total Wins for ML Model: {ml_wins} / {NUM_PROBLEMS_TO_RUN}")
    print(f"Total Wins for Control: {control_wins} / {NUM_PROBLEMS_TO_RUN}")
    print(f"Ties: {ties} / {NUM_PROBLEMS_TO_RUN}\n")

    # Conclusion
    node_reduction = (avg_nodes_control - avg_nodes_ml) / avg_nodes_control * 100
    print("--- CONCLUSION ---")
    if avg_nodes_ml < avg_nodes_control:
        print(f"The experiment supports the hypothesis (H1).")
        print(f"The simulated ML model reduced the average number of explored nodes by {node_reduction:.2f}%,")
        print("demonstrating that a learned, context-aware policy can be more efficient than a generic one.")
    else:
        print("The experiment does not support the hypothesis (H0).")
        print("There was no significant performance improvement from the simulated ML model.")