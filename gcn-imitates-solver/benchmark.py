import os
import pickle
import torch
import torch.nn.functional as F
from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --- GCN Model and Graph Utility ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.output_layer = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.output_layer(x).squeeze(-1)

def state_to_graph(instance, lp_solution_values):
    weights, values, capacity = instance['weights'], instance['values'], instance['capacity']
    num_items = len(weights)
    node_features = torch.tensor(
        [[weights[i], values[i], lp_solution_values[i]] for i in range(num_items)],
        dtype=torch.float
    )
    edge_list = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if weights[i] + weights[j] > capacity:
                edge_list.extend([[i, j], [j, i]])
    if not edge_list: return None
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

# --- Custom GCN Branching Rule ---
class GCNBranchingRule(Branchrule):
    def __init__(self, gcn_model, variables, instance):
        self.gcn_model = gcn_model
        self.variables = variables
        self.instance = instance
        self.num_items = instance['num_items']
        self.gcn_model.eval()

    def branchexeclp(self, allowaddcons):
        candidates, *_ = self.model.getLPBranchCands()
        if not candidates:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        lp_solution_values = [self.model.getVal(var) for var in self.variables]
        graph = state_to_graph(self.instance, lp_solution_values)
        if graph is None: return {'result': SCIP_RESULT.DIDNOTRUN}

        with torch.no_grad():
            scores = self.gcn_model(graph)

        best_candidate = None
        max_score = -float('inf')
        for cand_var in candidates:
            var_idx = cand_var.getIndex()
            if var_idx < self.num_items:
                if scores[var_idx] > max_score:
                    max_score = scores[var_idx]
                    best_candidate = cand_var

        if best_candidate is not None:
            self.model.branchVar(best_candidate)
            return {'result': SCIP_RESULT.BRANCHED}
        return {'result': SCIP_RESULT.DIDNOTRUN}

# --- Solver Functions ---
def run_gcn_scip(instance, gcn_model):
    scip = Model("GCNKnapsack")
    scip.hideOutput()
    x = {i: scip.addVar(vtype="B", name=f"x_{i}") for i in range(instance['num_items'])}
    scip.setObjective(sum(instance['values'][i] * x[i] for i in range(instance['num_items'])), "maximize")
    scip.addCons(sum(instance['weights'][i] * x[i] for i in range(instance['num_items'])) <= instance['capacity'])
    
    variables_list = [x[i] for i in range(instance['num_items'])]
    branching_rule = GCNBranchingRule(gcn_model=gcn_model, variables=variables_list, instance=instance)
    scip.includeBranchrule(branching_rule, "GCNBranching", "GCN branching rule", 999999, -1, 1)
    
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    
    scip.optimize()
    return scip.getObjVal(), scip.getNNodes()

def run_baseline_scip(instance):
    scip = Model("BaselineKnapsack")
    scip.hideOutput()
    x = {i: scip.addVar(vtype="B", name=f"x_{i}") for i in range(instance['num_items'])}
    scip.setObjective(sum(instance['values'][i] * x[i] for i in range(instance['num_items'])), "maximize")
    scip.addCons(sum(instance['weights'][i] * x[i] for i in range(instance['num_items'])) <= instance['capacity'])
    
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    
    scip.optimize()
    return scip.getObjVal(), scip.getNNodes()

# --- Main Benchmarking Logic ---
if __name__ == '__main__':
    # Load the trained scorer model once
    gcn_model = GCN(num_node_features=3)
    gcn_model.load_state_dict(torch.load('gcn_knapsack_scorer.pth', weights_only=True))
    
    # Define the set of instances to test on
    NUM_TEST_INSTANCES = 30
    instance_files = [f'knapsack_dataset/instance_{i}.pkl' for i in range(NUM_TEST_INSTANCES)]
    
    results = []
    print(f"--- Running benchmark on {NUM_TEST_INSTANCES} instances ---")

    for instance_file in instance_files:
        print(f"Processing {instance_file}...")
        with open(instance_file, 'rb') as f:
            instance = pickle.load(f)
        
        gcn_profit, gcn_nodes = run_gcn_scip(instance, gcn_model)
        scip_profit, scip_nodes = run_baseline_scip(instance)
        
        results.append({
            'instance': os.path.basename(instance_file),
            'gcn_profit': gcn_profit, 'gcn_nodes': gcn_nodes,
            'scip_profit': scip_profit, 'scip_nodes': scip_nodes
        })

    # --- Print the final summary table ---
    print("\n" + "="*70)
    print("                      BENCHMARKING RESULTS")
    print("="*70)
    print(f"{'Instance':<20} | {'GCN Profit':<12} | {'SCIP Profit':<12} | {'GCN Nodes':<10} | {'SCIP Nodes':<10}")
    print("-"*70)

    total_gcn_nodes = 0
    total_scip_nodes = 0
    gcn_wins = 0
    scip_wins = 0
    ties = 0

    for res in results:
        print(f"{res['instance']:<20} | {res['gcn_profit']:<12.1f} | {res['scip_profit']:<12.1f} | {res['gcn_nodes']:<10} | {res['scip_nodes']:<10}")
        total_gcn_nodes += res['gcn_nodes']
        total_scip_nodes += res['scip_nodes']
        if res['gcn_nodes'] < res['scip_nodes']:
            gcn_wins += 1
        elif res['scip_nodes'] < res['gcn_nodes']:
            scip_wins += 1
        else:
            ties += 1
            
    print("="*70)
    print("\n--- AGGREGATE STATISTICS ---")
    print(f"Average GCN Nodes:  {(total_gcn_nodes / len(results)):.2f}")
    print(f"Average SCIP Nodes: {(total_scip_nodes / len(results)):.2f}")
    print(f"\nWin/Loss Record (based on fewer nodes):")
    print(f"  GCN Heuristic Wins: {gcn_wins}")
    print(f"  SCIP Default Wins:  {scip_wins}")
    print(f"  Ties:               {ties}")
    print("="*70)