import numpy as np
from scipy.optimize import linprog
import time
import pandas as pd
import warnings

# Suppress warnings from SciPy for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# PART 1: THE BRANCHING STRATEGIES
# =============================================================================

def select_most_fractional_variable(candidates, solution, **kwargs):
    """CONTROL GROUP: Standard 'most fractional' branching heuristic."""
    fractional_parts = [abs(solution[idx] - 0.5) for idx in candidates]
    return candidates[np.argmin(fractional_parts)]


def select_strong_branching_variable(candidates, solution, c_min, A_ub, b_ub, current_bounds, **kwargs):
    """
    EXPERIMENTAL GROUP: Strong Branching.
    This simulates a PERFECT pre-trained ML model that has learned to predict
    the best variable to branch on by performing a look-ahead.
    """
    best_candidate = -1
    best_score = -np.inf

    # For each candidate, solve two LPs to see which is most promising
    for idx in candidates:
        
        # --- Look-ahead on the 'down' branch (var <= 0) ---
        bounds_down = list(current_bounds)
        bounds_down[idx] = (0, 0)
        res_down = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds_down, method='highs')
        obj_down = -res_down.fun if res_down.success else -np.inf

        # --- Look-ahead on the 'up' branch (var >= 1) ---
        bounds_up = list(current_bounds)
        bounds_up[idx] = (1, 1)
        res_up = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds_up, method='highs')
        obj_up = -res_up.fun if res_up.success else -np.inf

        # A common strong branching score: maximize the minimum improvement.
        score = min(obj_down, obj_up)
        
        if score > best_score:
            best_score = score
            best_candidate = idx
    
    # If all look-aheads fail, fall back to a simple choice
    if best_candidate == -1:
        return candidates[0]
        
    return best_candidate


# =============================================================================
# PART 2: THE CORE BRANCH AND BOUND SOLVER
# =============================================================================

def branch_and_bound_solver(c, A_ub, b_ub, branching_strategy):
    """
    A flexible B&B solver that now passes necessary problem data to the strategy.
    """
    c_min = -np.array(c)
    num_vars = len(c)
    
    best_obj_value = -np.inf
    nodes_explored = 0
    
    initial_bounds = tuple([(0, 1) for _ in range(num_vars)])
    stack = [(initial_bounds)]

    while stack:
        current_bounds = stack.pop()
        nodes_explored += 1
        
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=list(current_bounds), method='highs')

        if not res.success or -res.fun <= best_obj_value:
            continue

        solution = res.x
        is_integer = np.all(np.isclose(solution, np.round(solution)))
        
        if is_integer:
            current_obj = np.dot(c, solution)
            if current_obj > best_obj_value:
                best_obj_value = current_obj
            continue

        fractional_indices = [i for i, val in enumerate(solution) if not np.isclose(val, np.round(val))]
        if not fractional_indices:
            continue
            
        # --- HERE THE BRANCHING STRATEGY IS CALLED ---
        branch_var_index = branching_strategy(
            candidates=fractional_indices,
            solution=solution,
            c_min=c_min,
            A_ub=A_ub,
            b_ub=b_ub,
            current_bounds=current_bounds
        )
        # ---

        # Create two new subproblems (nodes)
        new_bounds1 = list(current_bounds)
        new_bounds1[branch_var_index] = (0, 0)
        stack.append(tuple(new_bounds1))
        
        new_bounds2 = list(current_bounds)
        new_bounds2[branch_var_index] = (1, 1)
        stack.append(tuple(new_bounds2))

    return best_obj_value, nodes_explored


# =============================================================================
# PART 3: PROBLEM GENERATOR
# =============================================================================

def generate_knapsack_problem(num_items, seed):
    """Generates a random 0-1 Knapsack Problem."""
    np.random.seed(seed)
    values = np.random.randint(10, 100, size=num_items)
    weights = np.random.randint(5, 50, size=num_items)
    capacity = int(np.sum(weights) * 0.6)
    return values, [weights], [capacity]


# =============================================================================
# PART 4: EXPERIMENT RUNNER AND ANALYSIS
# =============================================================================

if __name__ == '__main__':
    print("Running experiment: Strong Branching (simulated ML) vs. Standard Heuristic.\n")

    # --- Experiment Parameters ---
    NUM_PROBLEMS_TO_RUN = 10  # Reduced because strong branching is slow
    NUM_ITEMS_PER_PROBLEM = 35 # Slightly larger problems to show a clearer difference
    
    experiment_results = []

    for i in range(NUM_PROBLEMS_TO_RUN):
        print(f"--- Running on problem instance #{i+1} ---")
        
        # Generate a new problem with a fixed seed for reproducibility
        c, A_ub, b_ub = generate_knapsack_problem(NUM_ITEMS_PER_PROBLEM, seed=i)

        # Run Control Group (Most Fractional)
        print("  Running Control (Most Fractional)...")
        start_time_control = time.time()
        obj_control, nodes_control = branch_and_bound_solver(c, A_ub, b_ub, select_most_fractional_variable)
        time_control = time.time() - start_time_control
        
        # Run Experimental Group (Strong Branching)
        print("  Running 'Pre-trained Model' (Strong Branching)...")
        start_time_ml = time.time()
        obj_ml, nodes_ml = branch_and_bound_solver(c, A_ub, b_ub, select_strong_branching_variable)
        time_ml = time.time() - start_time_ml
        
        experiment_results.append({
            'ProblemID': i + 1,
            'Control_Nodes': nodes_control,
            'Model_Nodes': nodes_ml,
            'Control_Time_s': time_control,
            'Model_Time_s': time_ml
        })

    # --- Analyze and Display Results ---
    df = pd.DataFrame(experiment_results)
    df['Winner'] = np.where(df['Model_Nodes'] < df['Control_Nodes'], 'Pre-trained Model', 'Control')

    print("\n\n" + "="*65)
    print("                     EXPERIMENT RESULTS")
    print("="*65)
    print(df[['ProblemID', 'Control_Nodes', 'Model_Nodes', 'Control_Time_s', 'Model_Time_s', 'Winner']].round(2).to_string(index=False))
    print("-"*65)

    # Summary Statistics
    avg_nodes_control = df['Control_Nodes'].mean()
    avg_nodes_model = df['Model_Nodes'].mean()
    ml_wins = (df['Winner'] == 'Pre-trained Model').sum()

    print("\n--- SUMMARY STATISTICS (based on Node Count) ---")
    print(f"Average Nodes (Control Heuristic):   {avg_nodes_control:.2f}")
    print(f"Average Nodes ('Pre-trained' Model): {avg_nodes_model:.2f}\n")
    print(f"Total Wins for 'Pre-trained' Model: {ml_wins} / {NUM_PROBLEMS_TO_RUN}")
    
    # Note on time
    print("\n--- NOTE ON TIMING ---")
    print("The 'Pre-trained Model' is slower in this simulation because we are")
    print("actually performing the expensive look-ahead computations. A real,")
    print("deployed ML model would approximate this decision almost instantly.")

    # Conclusion
    node_reduction = (avg_nodes_control - avg_nodes_model) / avg_nodes_control * 100
    print("\n--- CONCLUSION ---")
    if avg_nodes_model < avg_nodes_control:
        print(f"The experiment strongly supports the hypothesis.")
        print(f"The high-quality decisions of the simulated model reduced the")
        print(f"average number of explored nodes by an impressive {node_reduction:.2f}%.")
        print("This demonstrates the significant potential of accelerating solvers by")
        print("using ML to predict the results of powerful but slow heuristics.")
    else:
        print("The experiment failed to show a benefit, indicating a possible issue.")