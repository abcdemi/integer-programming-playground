import os
import pickle
import torch
import torch.nn.functional as F
from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --- GCN class and graph utility (unchanged) ---
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

# --- The Custom SCIP Branching Rule with the FINAL FIX ---
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
        if graph is None:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        with torch.no_grad():
            scores = self.gcn_model(graph)

        best_candidate = None
        max_score = -float('inf')
        
        for cand_var in candidates:
            var_idx = cand_var.getIndex()
            
            # --- THIS IS THE FINAL, ROBUST FIX ---
            # Only consider variables that are part of our original problem,
            # ignoring any auxiliary/slack variables SCIP might have created.
            if var_idx < self.num_items:
                if scores[var_idx] > max_score:
                    max_score = scores[var_idx]
                    best_candidate = cand_var
            # ------------------------------------

        if best_candidate is not None:
            self.model.branchVar(best_candidate)
            return {'result': SCIP_RESULT.BRANCHED}
        
        return {'result': SCIP_RESULT.DIDNOTRUN}

if __name__ == '__main__':
    gcn_model = GCN(num_node_features=3)
    gcn_model.load_state_dict(torch.load('gcn_knapsack_scorer.pth', weights_only=True))
    with open('knapsack_dataset/instance_0.pkl', 'rb') as f:
        test_instance = pickle.load(f)

    print("\n--- Solving with SCIP guided by the GCN Scorer Model ---")
    
    scip = Model("GCNKnapsack")
    scip.hideOutput()
    
    num_items = test_instance['num_items']
    x = {i: scip.addVar(vtype="B", name=f"x_{i}") for i in range(num_items)}
    
    scip.setObjective(sum(test_instance['values'][i] * x[i] for i in range(num_items)), "maximize")
    scip.addCons(sum(test_instance['weights'][i] * x[i] for i in range(num_items)) <= test_instance['capacity'])
    
    variables_list = [x[i] for i in range(num_items)]
    
    branching_rule = GCNBranchingRule(gcn_model=gcn_model, variables=variables_list, instance=test_instance)
    
    scip.includeBranchrule(
        branching_rule,
        "GCNBranching",
        "Uses a GCN to make branching decisions",
        priority=999999,
        maxdepth=-1,
        maxbounddist=1
    )
    
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)

    scip.optimize()
    
    print("\n--- GCN-Powered SCIP Results ---")
    print(f"Optimal Profit: {scip.getObjVal()}")
    print(f"Nodes Explored: {scip.getNNodes()}")
    print("\nCompare this to the baseline SCIP result (27 nodes).")