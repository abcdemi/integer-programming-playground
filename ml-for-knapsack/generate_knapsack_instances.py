import numpy as np
import pickle
import os

def generate_knapsack_instance(num_items, correlation_type='uncorrelated'):
    """Generates a single 0/1 knapsack problem instance."""
    weights = np.random.randint(20, 100, size=num_items) # Let's make items a bit heavier
    
    if correlation_type == 'uncorrelated':
        values = np.random.randint(20, 100, size=num_items)
    elif correlation_type == 'weakly_correlated':
        noise = np.random.randint(-10, 10, size=num_items)
        values = np.maximum(1, weights + noise)
    elif correlation_type == 'strongly_correlated':
        noise = np.random.randint(-5, 5, size=num_items)
        values = np.maximum(1, weights + noise)
        
    # --- THIS IS THE FINAL, CORRECTED LOGIC ---
    # A capacity of 100 ensures that pairs of items with weights > 50 will
    # conflict, while smaller items won't, creating meaningful graphs.
    capacity = 100
    # -------------------------------------------
    
    return {
        'weights': weights.tolist(),
        'values': values.tolist(),
        'capacity': capacity,
        'num_items': num_items
    }

if __name__ == '__main__':
    NUM_INSTANCES = 1000
    # Using a smaller number of items is fine with this new capacity logic
    NUM_ITEMS = 50 
    DATASET_DIR = 'knapsack_dataset'
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    print(f"Generating {NUM_INSTANCES} knapsack instances...")
    
    for i in range(NUM_INSTANCES):
        corr_type = ['uncorrelated', 'weakly_correlated', 'strongly_correlated'][i % 3]
        instance = generate_knapsack_instance(NUM_ITEMS, corr_type)
        filepath = os.path.join(DATASET_DIR, f'instance_{i}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(instance, f)
            
    print(f"Dataset generated in '{DATASET_DIR}' directory.")