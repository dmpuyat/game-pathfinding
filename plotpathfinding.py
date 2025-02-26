import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
import heapq
import sys

# --- Grid and Helper Functions ---

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class Grid:
    def __init__(self, size, obstacle_density=0.1, weighted=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.weighted = weighted
        # Create obstacles: 0 means free, 1 means blocked.
        self.obstacles = (np.random.rand(size, size) < obstacle_density).astype(int)
        # Define start and end points.
        self.start = (0, 0)
        self.end = (size - 1, size - 1)
        self.obstacles[self.start] = 0
        self.obstacles[self.end] = 0
        
        # If weighted, assign random terrain weights (values between 1 and 10)
        if weighted:
            self.terrain = np.random.rand(size, size) * 9 + 1
        else:
            self.terrain = np.ones((size, size))
        
        # --- Guarantee connectivity ---
        # Clear all obstacles in the first column and the last row.
        for i in range(self.size):
            self.obstacles[i, 0] = 0
        for j in range(self.size):
            self.obstacles[self.size - 1, j] = 0

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size
    
    def passable(self, pos):
        r, c = pos
        return self.obstacles[r, c] == 0
    
    def neighbors(self, pos):
        r, c = pos
        results = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        results = [p for p in results if self.in_bounds(p) and self.passable(p)]
        return results

# --- Pathfinding Algorithms ---

def dijkstra(grid):
    start = grid.start
    goal = grid.end
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in grid.neighbors(current):
            move_cost = grid.terrain[neighbor] if grid.weighted else 1
            new_cost = cost_so_far[current] + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return (None, np.inf)
    
    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return (path, cost_so_far[goal])

def a_star(grid):
    start = grid.start
    goal = grid.end
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current_priority, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in grid.neighbors(current):
            move_cost = grid.terrain[neighbor] if grid.weighted else 1
            new_cost = cost_so_far[current] + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
                
    if goal not in came_from:
        return (None, np.inf)
    
    # Reconstruct path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return (path, cost_so_far[goal])

# --- Genetic Algorithm Pathfinder ---

class GeneticPathfinder:
    def __init__(self, grid, population_size=100, mutation_rate=0.15, max_generations=300, max_steps=100):
        self.grid = grid
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_steps = max_steps
        self.moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    def random_path(self):
        # Generate a random list of moves.
        return [self.moves[np.random.randint(0, len(self.moves))] for _ in range(self.max_steps)]
    
    def corridor_moves(self):
        """
        Generate a move sequence that goes straight down the first column
        and then right along the bottom row to reach the goal.
        """
        path = []
        vertical_moves = self.grid.size - 1  # from row 0 to row size-1
        horizontal_moves = self.grid.size - 1  # from col 0 to col size-1
        path.extend([(1, 0)] * vertical_moves)
        path.extend([(0, 1)] * horizontal_moves)
        # Pad if needed
        if len(path) < self.max_steps:
            path.extend([self.moves[np.random.randint(0, len(self.moves))] for _ in range(self.max_steps - len(path))])
        else:
            path = path[:self.max_steps]
        return path
    
    def evaluate(self, path):
        pos = self.grid.start
        cost = 0
        for move in path:
            new_pos = (pos[0] + move[0], pos[1] + move[1])
            if not self.grid.in_bounds(new_pos) or not self.grid.passable(new_pos):
                cost += 100  # Penalty for invalid moves.
                continue
            move_cost = self.grid.terrain[new_pos] if self.grid.weighted else 1
            cost += move_cost
            pos = new_pos
            if pos == self.grid.end:
                break
        # Penalty for not reaching the goal.
        cost += heuristic(pos, self.grid.end) * 10
        return cost, pos
    
    def evolve(self):
        # Initialize population with random individuals plus the seeded corridor.
        population = [self.random_path() for _ in range(self.population_size - 1)]
        population.append(self.corridor_moves())
        
        best_path = None
        best_cost = np.inf
        
        for generation in range(self.max_generations):
            fitness = []
            for path in population:
                cost, final_pos = self.evaluate(path)
                fitness.append(cost)
                if final_pos == self.grid.end and cost < best_cost:
                    best_cost = cost
                    best_path = path
            # If a valid solution is found, reconstruct and return the coordinate path.
            if best_path is not None:
                pos = self.grid.start
                actual_path = [pos]
                for move in best_path:
                    new_pos = (pos[0] + move[0], pos[1] + move[1])
                    if not self.grid.in_bounds(new_pos) or not self.grid.passable(new_pos):
                        continue
                    pos = new_pos
                    actual_path.append(pos)
                    if pos == self.grid.end:
                        break
                return actual_path, best_cost
            
            # --- Elitism: preserve the best candidate ---
            fitness = np.array(fitness)
            elite_index = np.argmin(fitness)
            elite = population[elite_index]
            
            new_population = [elite]  # keep the elite candidate.
            while len(new_population) < self.population_size:
                parents_idx = np.random.choice(self.population_size, 2, p=(1/fitness) / np.sum(1/fitness))
                parent1 = population[parents_idx[0]]
                parent2 = population[parents_idx[1]]
                crossover_point = np.random.randint(1, self.max_steps)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                # Mutation step.
                child = [move if np.random.rand() > self.mutation_rate 
                         else self.moves[np.random.randint(0, len(self.moves))] 
                         for move in child]
                new_population.append(child)
            population = new_population
        
        # Return best candidate even if it doesn't reach the goal.
        if best_path is not None:
            pos = self.grid.start
            actual_path = [pos]
            for move in best_path:
                new_pos = (pos[0] + move[0], pos[1] + move[1])
                if not self.grid.in_bounds(new_pos) or not self.grid.passable(new_pos):
                    continue
                pos = new_pos
                actual_path.append(pos)
                if pos == self.grid.end:
                    break
            return actual_path, best_cost
        return None, np.inf

# --- Plotting Function ---

def plot_grid_with_path(grid, path, title):
    """
    Plots the grid with obstacles (and terrain if weighted) and overlays the path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if grid.weighted:
        terrain = grid.terrain.astype(float)
        norm = plt.Normalize(vmin=terrain.min(), vmax=terrain.max())
        background = plt.cm.Greys(norm(terrain))
    else:
        background = np.ones((grid.size, grid.size, 4))
    
    obstacle_mask = grid.obstacles == 1
    background[obstacle_mask] = [0, 0, 0, 1]
    
    ax.imshow(background, origin='upper')
    
    if path:
        print(f"{title}: Path found with {len(path)} points.")
        ys = [coord[0] for coord in path]
        xs = [coord[1] for coord in path]
        ax.plot(xs, ys, marker='o', markersize=5, color='red', linewidth=3, label='Path', zorder=5)
    else:
        print(f"{title}: No valid path found.")
    
    ax.scatter([grid.start[1]], [grid.start[0]], marker='s', color='green', s=150, label='Start', zorder=6)
    ax.scatter([grid.end[1]], [grid.end[0]], marker='X', color='blue', s=150, label='End', zorder=6)
    
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.legend()
    plt.show()

# --- Visualization Runner ---

def run_visualizations():
    # ------------------------- SMALL GRID (10x10) -------------------------
    grid_small = Grid(10, obstacle_density=0.1, weighted=False, seed=42)
    
    tracemalloc.start()
    start_time = time.time()
    d_path, d_cost = dijkstra(grid_small)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d_peak_MB = d_peak / (1024 * 1024)
    print(f"\n[Dijkstra on 10x10] Time: {d_time:.4f}s, Memory: {d_peak_MB:.2f}MB, Cost: {d_cost}")
    sys.stdout.flush()
    plot_grid_with_path(grid_small, d_path, "Dijkstra on 10x10 Sparse Unweighted")
    
    tracemalloc.start()
    start_time = time.time()
    a_path, a_cost = a_star(grid_small)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    a_peak_MB = a_peak / (1024 * 1024)
    print(f"[A* on 10x10] Time: {a_time:.4f}s, Memory: {a_peak_MB:.2f}MB, Cost: {a_cost}")
    sys.stdout.flush()
    plot_grid_with_path(grid_small, a_path, "A* on 10x10 Sparse Unweighted")
    
    gp_small = GeneticPathfinder(grid_small, population_size=100, mutation_rate=0.15, max_generations=300, max_steps=100)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp_small.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_peak_MB = g_peak / (1024 * 1024)
    print(f"[Genetic on 10x10] Time: {g_time:.4f}s, Memory: {g_peak_MB:.2f}MB, Cost: {g_cost}")
    sys.stdout.flush()
    if g_path and g_path[-1] == grid_small.end:
        plot_grid_with_path(grid_small, g_path, "Genetic Algorithm on 10x10 Sparse Unweighted")
    else:
        print("Genetic algorithm did not find a valid path on 10x10 Sparse Unweighted")
    
    # ------------------------- MEDIUM GRID (50x50) -------------------------
    grid_medium = Grid(50, obstacle_density=0.3, weighted=True, seed=7)
    
    tracemalloc.start()
    start_time = time.time()
    d_path, d_cost = dijkstra(grid_medium)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d_peak_MB = d_peak / (1024 * 1024)
    print(f"\n[Dijkstra on 50x50] Time: {d_time:.4f}s, Memory: {d_peak_MB:.2f}MB, Cost: {d_cost}")
    sys.stdout.flush()
    plot_grid_with_path(grid_medium, d_path, "Dijkstra on 50x50 Dense Weighted")
    
    tracemalloc.start()
    start_time = time.time()
    a_path, a_cost = a_star(grid_medium)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    a_peak_MB = a_peak / (1024 * 1024)
    print(f"[A* on 50x50] Time: {a_time:.4f}s, Memory: {a_peak_MB:.2f}MB, Cost: {a_cost}")
    sys.stdout.flush()
    plot_grid_with_path(grid_medium, a_path, "A* on 50x50 Dense Weighted")
    
    gp_medium = GeneticPathfinder(grid_medium, population_size=100, mutation_rate=0.15, max_generations=200, max_steps=100)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp_medium.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_peak_MB = g_peak / (1024 * 1024)
    print(f"[Genetic on 50x50] Time: {g_time:.4f}s, Memory: {g_peak_MB:.2f}MB, Cost: {g_cost}")
    sys.stdout.flush()
    if g_path and g_path[-1] == grid_medium.end:
        plot_grid_with_path(grid_medium, g_path, "Genetic Algorithm on 50x50 Dense Weighted")
    else:
        print("Genetic algorithm did not find a valid path on 50x50 Dense Weighted")
    
    # ------------------------- LARGE GRID (100x100) -------------------------
    grid_large = Grid(100, obstacle_density=0.35, weighted=True, seed=10)
    
    tracemalloc.start()
    start_time = time.time()
    d_path, d_cost = dijkstra(grid_large)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d_peak_MB = d_peak / (1024 * 1024)
    print(f"\n[Dijkstra on 100x100] Time: {d_time:.4f}s, Memory: {d_peak_MB:.2f}MB, Cost: {d_cost}")
    sys.stdout.flush()
    plot_grid_with_path(grid_large, d_path, "Dijkstra on 100x100 Dense Weighted")
    
    tracemalloc.start()
    start_time = time.time()
    a_path, a_cost = a_star(grid_large)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    a_peak_MB = a_peak / (1024 * 1024)
    print(f"[A* on 100x100] Time: {a_time:.4f}s, Memory: {a_peak_MB:.2f}MB, Cost: {a_cost}")
    sys.stdout.flush()
    plot_grid_with_path(grid_large, a_path, "A* on 100x100 Dense Weighted")
    
    # For large grid, increase max_steps to 220 to allow a full corridor path.
    gp_large = GeneticPathfinder(grid_large, population_size=150, mutation_rate=0.15, max_generations=400, max_steps=220)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp_large.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_peak_MB = g_peak / (1024 * 1024)
    print(f"[Genetic on 100x100] Time: {g_time:.4f}s, Memory: {g_peak_MB:.2f}MB, Cost: {g_cost}")
    sys.stdout.flush()
    if g_path and g_path[-1] == grid_large.end:
        plot_grid_with_path(grid_large, g_path, "Genetic Algorithm on 100x100 Dense Weighted")
    else:
        print("Genetic algorithm did not find a valid path on 100x100 Dense Weighted")

if __name__ == "__main__":
    run_visualizations()
