import numpy as np
import pickle
import os

def generate_knapsack_instance(num_items, correlation_type='uncorrelated'):
    """
    Generates a single 0/1 knapsack problem instance.
    
    Args:
        num_items (int): The number of items in the problem.
        correlation_type (str): The relationship between weights and values.
            - 'uncorrelated': Weights and values are independent.
            - 'weakly_correlated': Value is based on weight with large noise.
            - 'strongly_correlated': Value is based on weight with small noise.

    Returns:
        dict: A dictionary containing item weights, values, and knapsack capacity.
    """
    
    # Generate weights randomly
    weights = np.random.randint(1, 100, size=num_items)
    
    if correlation_type == 'uncorrelated':
        values = np.random.randint(1, 100, size=num_items)
    elif correlation_type == 'weakly_correlated':
        # Values are based on weights with a large random component
        noise = np.random.randint(-25, 25, size=num_items)
        values = np.maximum(1, weights + noise)
    elif correlation_type == 'strongly_correlated':
        # Values are tightly linked to weights plus a small random component
        noise = np.random.randint(-5, 5, size=num_items)
        values = np.maximum(1, weights + noise)
    else:
        raise ValueError("Invalid correlation type")

    # Set capacity to 40% of the total weight of all items
    # This usually creates non-trivial problems
    capacity = int(np.sum(weights) * 0.4)
    
    return {
        'weights': weights.tolist(),
        'values': values.tolist(),
        'capacity': capacity,
        'num_items': num_items
    }

# --- Main script to generate the dataset ---
if __name__ == '__main__':
    NUM_INSTANCES = 1000  # Generate 1000 problems
    NUM_ITEMS = 500       # Each with 500 items
    DATASET_DIR = 'knapsack_dataset'
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    print(f"Generating {NUM_INSTANCES} knapsack instances...")
    
    for i in range(NUM_INSTANCES):
        # Cycle through correlation types to get a diverse dataset
        corr_type = ['uncorrelated', 'weakly_correlated', 'strongly_correlated'][i % 3]
        
        instance = generate_knapsack_instance(NUM_ITEMS, corr_type)
        
        # Save each instance as a separate file
        filepath = os.path.join(DATASET_DIR, f'instance_{i}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(instance, f)
            
    print(f"Dataset generated in '{DATASET_DIR}' directory.")