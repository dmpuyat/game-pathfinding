
from pathfinding import Grid
import matplotlib.pyplot as plt

from pathfinding import run_comparison

def plot_results(results, title):
    algorithms = list(results.keys())
    times = [results[alg]['time'] for alg in algorithms]
    costs = [results[alg]['cost'] if isinstance(results[alg]['cost'], (int, float)) else float('inf') for alg in algorithms]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Time
    ax[0].bar(algorithms, times, color=['blue', 'green', 'red'])
    ax[0].set_title(f'{title} - Time (s)')
    ax[0].set_ylabel('Time (seconds)')
    
    # Plot Cost
    ax[1].bar(algorithms, costs, color=['blue', 'green', 'red'])
    ax[1].set_title(f'{title} - Path Cost')
    ax[1].set_ylabel('Cost')
    
    plt.tight_layout()
    plt.show()

# Test for Small, Medium, Large, Dense, Sparse, Weighted, and Unweighted Graphs
def run_tests():
    # Small Graphs (10x10)
    grid_small_sparse_unweighted = Grid(10, obstacle_density=0.1, weighted=False, seed=42)
    grid_small_dense_unweighted = Grid(10, obstacle_density=0.4, weighted=False, seed=42)
    grid_small_sparse_weighted = Grid(10, obstacle_density=0.1, weighted=True, seed=42)
    grid_small_dense_weighted = Grid(10, obstacle_density=0.4, weighted=True, seed=42)
    
    # Medium Graphs (50x50)
    grid_medium_sparse_unweighted = Grid(50, obstacle_density=0.1, weighted=False, seed=42)
    grid_medium_dense_unweighted = Grid(50, obstacle_density=0.4, weighted=False, seed=42)
    grid_medium_sparse_weighted = Grid(50, obstacle_density=0.1, weighted=True, seed=42)
    grid_medium_dense_weighted = Grid(50, obstacle_density=0.4, weighted=True, seed=42)
    
    # Large Graphs (100x100)
    grid_large_sparse_unweighted = Grid(100, obstacle_density=0.1, weighted=False, seed=42)
    grid_large_dense_unweighted = Grid(100, obstacle_density=0.4, weighted=False, seed=42)
    grid_large_sparse_weighted = Grid(100, obstacle_density=0.1, weighted=True, seed=42)
    grid_large_dense_weighted = Grid(100, obstacle_density=0.4, weighted=True, seed=42)
    
    # Run comparisons
    results_small_sparse_unweighted = run_comparison(grid_small_sparse_unweighted)
    results_small_dense_unweighted = run_comparison(grid_small_dense_unweighted)
    results_small_sparse_weighted = run_comparison(grid_small_sparse_weighted)
    results_small_dense_weighted = run_comparison(grid_small_dense_weighted)
    
    results_medium_sparse_unweighted = run_comparison(grid_medium_sparse_unweighted)
    results_medium_dense_unweighted = run_comparison(grid_medium_dense_unweighted)
    results_medium_sparse_weighted = run_comparison(grid_medium_sparse_weighted)
    results_medium_dense_weighted = run_comparison(grid_medium_dense_weighted)
    
    results_large_sparse_unweighted = run_comparison(grid_large_sparse_unweighted)
    results_large_dense_unweighted = run_comparison(grid_large_dense_unweighted)
    results_large_sparse_weighted = run_comparison(grid_large_sparse_weighted)
    results_large_dense_weighted = run_comparison(grid_large_dense_weighted)
    
    # Plot results
    plot_results(results_small_sparse_unweighted, "Small Sparse Unweighted")
    plot_results(results_small_dense_unweighted, "Small Dense Unweighted")
    plot_results(results_small_sparse_weighted, "Small Sparse Weighted")
    plot_results(results_small_dense_weighted, "Small Dense Weighted")
    
    plot_results(results_medium_sparse_unweighted, "Medium Sparse Unweighted")
    plot_results(results_medium_dense_unweighted, "Medium Dense Unweighted")
    plot_results(results_medium_sparse_weighted, "Medium Sparse Weighted")
    plot_results(results_medium_dense_weighted, "Medium Dense Weighted")
    
    plot_results(results_large_sparse_unweighted, "Large Sparse Unweighted")
    plot_results(results_large_dense_unweighted, "Large Dense Unweighted")
    plot_results(results_large_sparse_weighted, "Large Sparse Weighted")
    plot_results(results_large_dense_weighted, "Large Dense Weighted")

# Run all tests
run_tests()
