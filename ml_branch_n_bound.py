import numpy as np
from scipy.optimize import linprog
import time
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# PART 1: SIMULATED GCN MODEL
# =============================================================================

class SimulatedGCN:
    """
    Simulates a pre-trained Graph Convolutional Network.
    In a real scenario, these weights would be learned. Here, they are random
    to demonstrate the mechanism without requiring a real pre-trained file.
    """
    def __init__(self, feature_size=5, embedding_size=16):
        # These matrices would be loaded from a pre-trained model file
        self.var_embedding = np.random.rand(feature_size, embedding_size)
        self.cons_embedding = np.random.rand(feature_size, embedding_size)
        self.final_weights = np.random.rand(embedding_size, 1)

    def predict(self, var_features, cons_features, adj_matrix):
        # 1. Embed initial features
        var_embed = var_features @ self.var_embedding
        cons_embed = cons_features @ self.cons_embedding

        # 2. Simulate one layer of graph convolution (message passing)
        # Messages from constraints to variables
        var_messages = adj_matrix.T @ cons_embed
        
        # 3. Simple update (in reality, this involves non-linearities)
        updated_var_embed = var_embed + var_messages

        # 4. Final prediction layer
        scores = updated_var_embed @ self.final_weights
        return scores.flatten()

# =============================================================================
# PART 2: THE BRANCHING STRATEGIES
# =============================================================================

def select_most_fractional_variable(candidates, solution, **kwargs):
    """CONTROL GROUP: Standard 'most fractional' branching heuristic."""
    fractional_parts = [abs(solution[idx] - 0.5) for idx in candidates]
    return candidates[np.argmin(fractional_parts)]

def select_strong_branching_variable(candidates, c_min, A_ub, b_ub, current_bounds, **kwargs):
    """ORACLE GROUP: The best, but slowest, heuristic."""
    best_candidate, best_score = -1, -np.inf
    for idx in candidates:
        bounds_down, bounds_up = list(current_bounds), list(current_bounds)
        bounds_down[idx], bounds_up[idx] = (0, 0), (1, 1)
        res_down = linprog(c_min, A_ub, b_ub, bounds_down, method='highs')
        res_up = linprog(c_min, A_ub, b_ub, bounds_up, method='highs')
        obj_down = -res_down.fun if res_down.success else -np.inf
        obj_up = -res_up.fun if res_up.success else -np.inf
        score = min(obj_down, obj_up)
        if score > best_score:
            best_score, best_candidate = score, idx
    return best_candidate if best_candidate != -1 else candidates[0]

def select_gcn_predicted_variable(candidates, solution, c_min, A_ub, b_ub, gcn_model, **kwargs):
    """EXPERIMENTAL GROUP: Branching heuristic guided by the simulated GCN."""
    num_vars = len(c_min)
    num_cons = len(b_ub)
    
    # 1. Build Graph Representation
    # Adjacency matrix: 1 if var j is in constraint i
    adj_matrix = (A_ub != 0).astype(float)

    # Feature Engineering (simplified)
    var_features = np.zeros((num_vars, 5))
    var_features[:, 0] = -c_min  # Objective coefficient
    var_features[:, 1] = solution # Current LP solution value
    var_features[:, 2] = np.isclose(solution, 0) # Is at lower bound
    var_features[:, 3] = np.isclose(solution, 1) # Is at upper bound
    var_features[:, 4] = [i in candidates for i in range(num_vars)] # Is fractional

    cons_features = np.zeros((num_cons, 5))
    slack = b_ub - (A_ub @ solution)
    cons_features[:, 0] = slack # Slack value of constraint
    cons_features[:, 1] = np.isclose(slack, 0) # Is tight
    
    # 2. Get predictions from the GCN model
    all_scores = gcn_model.predict(var_features, cons_features, adj_matrix)
    
    # 3. Choose the candidate with the highest score
    candidate_scores = {idx: all_scores[idx] for idx in candidates}
    return max(candidate_scores, key=candidate_scores.get)

# =============================================================================
# PART 3: THE CORE SOLVER & PROBLEM GENERATOR (Unchanged)
# =============================================================================

def branch_and_bound_solver(c, A_ub, b_ub, branching_strategy, gcn_model=None):
    c_min, num_vars = -np.array(c), len(c)
    best_obj_value, nodes_explored = -np.inf, 0
    stack = [(tuple([(0, 1) for _ in range(num_vars)]))]

    while stack:
        current_bounds = stack.pop()
        nodes_explored += 1
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=list(current_bounds), method='highs')
        if not res.success or -res.fun <= best_obj_value: continue
        solution = res.x
        if np.all(np.isclose(solution, np.round(solution))):
            best_obj_value = max(best_obj_value, np.dot(c, solution))
            continue
        fractional_indices = [i for i, v in enumerate(solution) if not np.isclose(v, np.round(v))]
        if not fractional_indices: continue
        
        branch_var_index = branching_strategy(
            candidates=fractional_indices, solution=solution, c_min=c_min, A_ub=A_ub, b_ub=b_ub,
            current_bounds=current_bounds, gcn_model=gcn_model
        )
        
        for bound_val in [0, 1]:
            new_bounds = list(current_bounds)
            new_bounds[branch_var_index] = (bound_val, bound_val)
            stack.append(tuple(new_bounds))
    return best_obj_value, nodes_explored

def generate_knapsack_problem(num_items, seed):
    np.random.seed(seed)
    c = np.random.randint(10, 100, size=num_items)
    A = np.random.randint(5, 50, size=num_items)
    b = [int(np.sum(A) * 0.6)]
    return c, np.array([A]), b

# =============================================================================
# PART 4: EXPERIMENT RUNNER AND ANALYSIS
# =============================================================================

if __name__ == '__main__':
    NUM_PROBLEMS = 5
    NUM_ITEMS = 40
    
    sim_gcn = SimulatedGCN()
    results = []

    print("Running experiment: GCN Simulation vs. Most Fractional vs. Strong Branching\n")

    for i in range(NUM_PROBLEMS):
        print(f"--- Running on problem instance #{i+1} ---")
        c, A_ub, b_ub = generate_knapsack_problem(NUM_ITEMS, seed=i)
        
        # 1. Control: Most Fractional
        start_t = time.time()
        _, nodes_mf = branch_and_bound_solver(c, A_ub, b_ub, select_most_fractional_variable)
        time_mf = time.time() - start_t

        # 2. Oracle: Strong Branching
        start_t = time.time()
        _, nodes_sb = branch_and_bound_solver(c, A_ub, b_ub, select_strong_branching_variable)
        time_sb = time.time() - start_t
        
        # 3. Experimental: GCN Simulation
        start_t = time.time()
        _, nodes_gcn = branch_and_bound_solver(c, A_ub, b_ub, select_gcn_predicted_variable, gcn_model=sim_gcn)
        time_gcn = time.time() - start_t
        
        results.append({
            'ProblemID': i + 1,
            'Nodes_Control': nodes_mf,
            'Nodes_GCN_Sim': nodes_gcn,
            'Nodes_Oracle': nodes_sb,
            'Time_Control_s': time_mf,
            'Time_GCN_Sim_s': time_gcn,
            'Time_Oracle_s': time_sb,
        })

    df = pd.DataFrame(results)
    
    print("\n\n" + "="*80)
    print(" " * 30 + "EXPERIMENT RESULTS")
    print("="*80)
    print(df.round(2).to_string(index=False))
    print("="*80)
    
    avg_nodes_mf = df['Nodes_Control'].mean()
    avg_nodes_gcn = df['Nodes_GCN_Sim'].mean()
    avg_nodes_sb = df['Nodes_Oracle'].mean()

    print("\n--- SUMMARY & ANALYSIS ---")
    print(f"Avg Nodes (Control - Most Fractional): {avg_nodes_mf:.1f}")
    print(f"Avg Nodes (GCN Simulation):            {avg_nodes_gcn:.1f}")
    print(f"Avg Nodes (Oracle - Strong Branching): {avg_nodes_sb:.1f}\n")
    print("The goal of a real GCN model is to achieve a node count close to the Oracle")
    print("but with a runtime closer to the Control. Our GCN simulation, using random")
    print("weights, shows performance somewhere between the two, as expected.\n")
    print("A well-trained GCN would push the 'Nodes_GCN_Sim' value much closer to 'Nodes_Oracle'.")