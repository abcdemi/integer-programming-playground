import heapq

class Item:
    """Represents an item with a weight and a value."""
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.ratio = value / weight if weight > 0 else 0

class Node:
    """Represents a node in the search tree."""
    def __init__(self, level, profit, weight, bound):
        self.level = level  # Level in the decision tree
        self.profit = profit  # Profit of nodes on the path from the root
        self.weight = weight  # Weight of nodes on the path from the root
        self.bound = bound  # Upper bound of the maximum profit in the subtree

    def __lt__(self, other):
        # heapq is a min-heap, so we invert the comparison
        return self.bound > other.bound

def bound(node, n, capacity, items):
    """
    Calculates the upper bound of the profit in the subtree rooted with node.
    This is our heuristic function.
    """
    if node.weight >= capacity:
        return 0

    profit_bound = node.profit
    j = node.level + 1
    total_weight = node.weight

    while j < n and total_weight + items[j].weight <= capacity:
        total_weight += items[j].weight
        profit_bound += items[j].value
        j += 1

    if j < n:
        profit_bound += (capacity - total_weight) * items[j].ratio

    return profit_bound

def knapsack_branch_and_bound(items, capacity):
    """
    Solves the 0/1 Knapsack problem using Branch and Bound.
    """
    # Sort items by value-to-weight ratio in descending order
    items.sort(key=lambda x: x.ratio, reverse=True)
    n = len(items)

    # Use a priority queue (max-heap based on bound) to store live nodes
    pq = []
    
    # Initial dummy node
    root = Node(-1, 0, 0, 0.0)
    root.bound = bound(root, n, capacity, items)
    heapq.heappush(pq, root)

    max_profit = 0

    while pq:
        # Get the node with the highest bound
        u = heapq.heappop(pq)

        if u.bound > max_profit:
            # Explore the next level
            v_level = u.level + 1

            # --- Branch 1: Include the next item ---
            if v_level < n:
                v_include = Node(v_level, u.profit + items[v_level].value, u.weight + items[v_level].weight, 0.0)

                if v_include.weight <= capacity:
                    if v_include.profit > max_profit:
                        max_profit = v_include.profit
                    
                    v_include.bound = bound(v_include, n, capacity, items)
                    
                    if v_include.bound > max_profit:
                        heapq.heappush(pq, v_include)

                # --- Branch 2: Exclude the next item ---
                v_exclude = Node(v_level, u.profit, u.weight, 0.0)
                v_exclude.bound = bound(v_exclude, n, capacity, items)

                if v_exclude.bound > max_profit:
                    heapq.heappush(pq, v_exclude)

    return max_profit

if __name__ == '__main__':
    # Example Usage
    capacity = 10
    items = [Item(2, 40), Item(3.14, 50), Item(1.98, 100), Item(5, 95), Item(3, 30)]

    max_profit = knapsack_branch_and_bound(items, capacity)
    print(f"Maximum profit: {max_profit}")