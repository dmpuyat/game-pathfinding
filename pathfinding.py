import numpy as np
import heapq
import random
import time
import tracemalloc

class Grid:
    def __init__(self, size, obstacle_density=0.2, weighted=False, seed=None):
        self.size = size
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate obstacles (1 means obstacle)
        self.obstacles = np.random.choice(
            [0, 1], size=(size, size), p=[1 - obstacle_density, obstacle_density]
        )
        
        # Generate terrain costs
        if weighted:
            self.terrain = np.random.choice([1, 3, 5], size=(size, size), p=[0.7, 0.2, 0.1])
        else:
            self.terrain = np.ones((size, size))
        
        # Set start and end points (top-left and bottom-right)
        self.start = (0, 0)
        self.end = (size - 1, size - 1)
        self.obstacles[self.start] = 0
        self.obstacles[self.end] = 0

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.obstacles[nx, ny] == 0:
                cost = self.terrain[nx, ny]
                neighbors.append(((nx, ny), cost))
        return neighbors

def dijkstra(grid):
    start, end = grid.start, grid.end
    heap = [(0, start)]
    cost_so_far = {start: 0}
    
    while heap:
        current_cost, current = heapq.heappop(heap)
        if current == end:
            return cost_so_far[end]
        for neighbor, move_cost in grid.get_neighbors(current):
            new_cost = current_cost + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor))
    return float('inf')

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid):
    start, end = grid.start, grid.end
    heap = [(0, start)]
    cost_so_far = {start: 0}
    
    while heap:
        current_priority, current = heapq.heappop(heap)
        if current == end:
            return cost_so_far[end]
        for neighbor, move_cost in grid.get_neighbors(current):
            new_cost = cost_so_far[current] + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(end, neighbor)
                heapq.heappush(heap, (priority, neighbor))
    return float('inf')

class GeneticPathfinder:
    def __init__(self, grid, population_size=50, mutation_rate=0.1, max_generations=100, max_steps=50):
        self.grid = grid
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_steps = max_steps
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def generate_individual(self):
        individual = []
        current = self.grid.start
        for _ in range(self.max_steps):
            if current == self.grid.end:
                break
            neighbors = self.grid.get_neighbors(current)
            if not neighbors:
                break
            next_node = random.choice(neighbors)[0]
            dx, dy = next_node[0] - current[0], next_node[1] - current[1]
            individual.append((dx, dy))
            current = next_node
        return individual
    
    def simulate_path(self, individual):
        current = self.grid.start
        path = [current]
        cost = 0
        for move in individual:
            next_node = (current[0] + move[0], current[1] + move[1])
            if not (0 <= next_node[0] < self.grid.size and 0 <= next_node[1] < self.grid.size):
                continue
            if self.grid.obstacles[next_node]:
                continue
            path.append(next_node)
            cost += self.grid.terrain[next_node]
            current = next_node
            if current == self.grid.end:
                break
        return path, cost
    
    def fitness(self, path, cost):
        if not path:
            return 0
        end_dist = heuristic(path[-1], self.grid.end)
        if path[-1] == self.grid.end:
            return 1 / (cost + 1)
        else:
            return 1 / (cost + end_dist + 1)
    
    def evolve(self):
        population = [self.generate_individual() for _ in range(self.population_size)]
        best_path, best_cost = None, float('inf')
        best_individual = None

        for _ in range(self.max_generations):
            fitnesses = []
            for ind in population:
                path, cost = self.simulate_path(ind)
                fit = self.fitness(path, cost)
                fitnesses.append(fit)
                if path and path[-1] == self.grid.end and cost < best_cost:
                    best_path, best_cost = path, cost
                    best_individual = ind.copy()
            
            if best_path and best_path[-1] == self.grid.end:
                break

            new_population = []
            if best_individual is not None:
                new_population.append(best_individual.copy())

            while len(new_population) < self.population_size:
                parents = random.choices(population, weights=fitnesses, k=2)
                p1, p2 = parents
                min_len = min(len(p1), len(p2))
                if min_len < 2:
                    child1, child2 = p1.copy(), p2.copy()
                else:
                    crossover_point = random.randint(1, min_len - 1)
                    child1 = p1[:crossover_point] + p2[crossover_point:]
                    child2 = p2[:crossover_point] + p1[crossover_point:]
                child1 = [m if random.random() > self.mutation_rate else random.choice(self.directions) for m in child1]
                child2 = [m if random.random() > self.mutation_rate else random.choice(self.directions) for m in child2]
                new_population.extend([child1, child2])
            population = new_population[:self.population_size]
        return best_path, best_cost

def run_comparison(grid_size):
    # Create a grid with given size. Here, we use weighted=True and obstacle_density=0.2.
    grid = Grid(grid_size, obstacle_density=0.2, weighted=True, seed=42)
    results = {}
    
    # --- Dijkstra --- 
    tracemalloc.start()
    start_time = time.time()
    d_cost = dijkstra(grid)
    d_time = time.time() - start_time
    _, d_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['Dijkstra'] = {
        'time': d_time,
        'peak_memory_MB': d_peak / (1024 * 1024),
        'cost': d_cost,
        'optimal': True
    }
    
    # --- A* ---
    tracemalloc.start()
    start_time = time.time()
    a_cost = a_star(grid)
    a_time = time.time() - start_time
    _, a_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['A*'] = {
        'time': a_time,
        'peak_memory_MB': a_peak / (1024 * 1024),
        'cost': a_cost,
        'optimal': a_cost == d_cost
    }
    
    # --- Genetic Algorithm ---
    gp = GeneticPathfinder(grid)
    tracemalloc.start()
    start_time = time.time()
    g_path, g_cost = gp.evolve()
    g_time = time.time() - start_time
    _, g_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['Genetic'] = {
        'time': g_time,
        'peak_memory_MB': g_peak / (1024 * 1024),
        'cost': g_cost if g_path and g_path[-1] == grid.end else 'No path',
        'optimal': (g_cost == d_cost) if g_path and g_path[-1] == grid.end else False
    }
    
    return results

def main():
    sizes = [10, 50, 100]
    for size in sizes:
        print(f"\nTime and Space Complexity for {size}x{size} Grid:")
        results = run_comparison(size)
        for algo, data in results.items():
            print(f"  {algo}: Time = {data['time']:.4f} sec, "
                  f"Memory = {data['peak_memory_MB']:.2f} MB, "
                  f"Cost = {data['cost']}, "
                  f"Optimal = {data['optimal']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
