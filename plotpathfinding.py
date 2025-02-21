import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from pathfinding import Grid, dijkstra, a_star, GeneticPathfinder

def plot_grid_with_path(grid, path, title):
    """
    Plots the grid with obstacles and (if weighted) terrain,
    and overlays the path taken.
    
    grid: instance of Grid.
    path: list of (row, col) coordinates for the path.
    title: Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create background: if weighted, use terrain values; else, plain white.
    if grid.weighted:
        terrain = grid.terrain.astype(float)
        norm = plt.Normalize(vmin=terrain.min(), vmax=terrain.max())
        background = plt.cm.Greys(norm(terrain))
    else:
        background = np.ones((grid.size, grid.size, 4))
    
    # Overlay obstacles as black cells.
    obstacle_mask = grid.obstacles == 1
    background[obstacle_mask] = [0, 0, 0, 1]
    
    ax.imshow(background, origin='upper')
    
    # Debug: print the path length
    if path:
        print(f"{title}: Path found with {len(path)} points.")
    else:
        print(f"{title}: No valid path found.")
    
    # Plot the path if it exists.
    if path and len(path) > 0:
        ys = [coord[0] for coord in path]
        xs = [coord[1] for coord in path]
        ax.plot(xs, ys, marker='o', markersize=5, color='red', linewidth=3, label='Path', zorder=5)
    
    # Mark the start and end points.
    ax.scatter([grid.start[1]], [grid.start[0]], marker='s', color='green', s=150, label='Start', zorder=6)
    ax.scatter([grid.end[1]], [grid.end[0]], marker='X', color='blue', s=150, label='End', zorder=6)
    
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.legend()
    plt.show()

def run_visualizations():
    # ------------------------- SMALL GRID (10x10) -------------------------
    grid_small = Grid(10, obstacle_density=0.1, weighted=False, seed=42)
    
    # Dijkstra on Small Grid
    tracemalloc.start()
    start_time = time.time()
    d_path, d_cost = dijkstra(grid_small)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d_peak_MB = d_peak / (1024 * 1024)
    print(f"\n[Dijkstra on 10x10] Time: {d_time:.4f}s, Memory: {d_peak_MB:.2f}MB, Cost: {d_cost}")
    plot_grid_with_path(grid_small, d_path, "Dijkstra on 10x10 Sparse Unweighted")
    
    # A* on Small Grid
    tracemalloc.start()
    start_time = time.time()
    a_path, a_cost = a_star(grid_small)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    a_peak_MB = a_peak / (1024 * 1024)
    print(f"[A* on 10x10] Time: {a_time:.4f}s, Memory: {a_peak_MB:.2f}MB, Cost: {a_cost}")
    plot_grid_with_path(grid_small, a_path, "A* on 10x10 Sparse Unweighted")
    
    # Genetic Algorithm on Small Grid
    gp_small = GeneticPathfinder(grid_small, population_size=100, mutation_rate=0.15, max_generations=300, max_steps=100)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp_small.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_peak_MB = g_peak / (1024 * 1024)
    print(f"[Genetic on 10x10] Time: {g_time:.4f}s, Memory: {g_peak_MB:.2f}MB, Cost: {g_cost}")
    if g_path and g_path[-1] == grid_small.end:
        plot_grid_with_path(grid_small, g_path, "Genetic Algorithm on 10x10 Sparse Unweighted")
    else:
        print("Genetic algorithm did not find a valid path on 10x10 Sparse Unweighted")
    
    # ------------------------- MEDIUM GRID (50x50) -------------------------
    grid_medium = Grid(50, obstacle_density=0.3, weighted=True, seed=7)
    
    # Dijkstra on Medium Grid
    tracemalloc.start()
    start_time = time.time()
    d_path, d_cost = dijkstra(grid_medium)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d_peak_MB = d_peak / (1024 * 1024)
    print(f"\n[Dijkstra on 50x50] Time: {d_time:.4f}s, Memory: {d_peak_MB:.2f}MB, Cost: {d_cost}")
    plot_grid_with_path(grid_medium, d_path, "Dijkstra on 50x50 Dense Weighted")
    
    # A* on Medium Grid
    tracemalloc.start()
    start_time = time.time()
    a_path, a_cost = a_star(grid_medium)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    a_peak_MB = a_peak / (1024 * 1024)
    print(f"[A* on 50x50] Time: {a_time:.4f}s, Memory: {a_peak_MB:.2f}MB, Cost: {a_cost}")
    plot_grid_with_path(grid_medium, a_path, "A* on 50x50 Dense Weighted")
    
    # Genetic Algorithm on Medium Grid
    gp_medium = GeneticPathfinder(grid_medium, population_size=200, mutation_rate=0.15, max_generations=500, max_steps=150)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp_medium.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_peak_MB = g_peak / (1024 * 1024)
    print(f"[Genetic on 50x50] Time: {g_time:.4f}s, Memory: {g_peak_MB:.2f}MB, Cost: {g_cost}")
    if g_path and g_path[-1] == grid_medium.end:
        plot_grid_with_path(grid_medium, g_path, "Genetic Algorithm on 50x50 Dense Weighted")
    else:
        print("Genetic algorithm did not find a valid path on 50x50 Dense Weighted")
    
    # ------------------------- LARGE GRID (100x100) -------------------------
    grid_large = Grid(100, obstacle_density=0.35, weighted=True, seed=10)
    
    # Dijkstra on Large Grid
    tracemalloc.start()
    start_time = time.time()
    d_path, d_cost = dijkstra(grid_large)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d_peak_MB = d_peak / (1024 * 1024)
    print(f"\n[Dijkstra on 100x100] Time: {d_time:.4f}s, Memory: {d_peak_MB:.2f}MB, Cost: {d_cost}")
    plot_grid_with_path(grid_large, d_path, "Dijkstra on 100x100 Dense Weighted")
    
    # A* on Large Grid
    tracemalloc.start()
    start_time = time.time()
    a_path, a_cost = a_star(grid_large)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    a_peak_MB = a_peak / (1024 * 1024)
    print(f"[A* on 100x100] Time: {a_time:.4f}s, Memory: {a_peak_MB:.2f}MB, Cost: {a_cost}")
    plot_grid_with_path(grid_large, a_path, "A* on 100x100 Dense Weighted")
    
    # Genetic Algorithm on Large Grid
    gp_large = GeneticPathfinder(grid_large, population_size=300, mutation_rate=0.15, max_generations=1000, max_steps=300)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp_large.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_peak_MB = g_peak / (1024 * 1024)
    print(f"[Genetic on 100x100] Time: {g_time:.4f}s, Memory: {g_peak_MB:.2f}MB, Cost: {g_cost}")
    if g_path and g_path[-1] == grid_large.end:
        plot_grid_with_path(grid_large, g_path, "Genetic Algorithm on 100x100 Dense Weighted")
    else:
        print("Genetic algorithm did not find a valid path on 100x100 Dense Weighted")

if __name__ == "__main__":
    run_visualizations()
