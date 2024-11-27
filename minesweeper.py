from typing import List, Dict, Tuple, Any, Optional
from enum import Enum
import random
import numpy as np
from base import Variable, CSP, Constraint
from analysis import CSPAnalyzer
from collections import defaultdict
from copy import deepcopy
from solvers import BacktrackingSolver, AC3Solver,ForwardCheckingSolver, BackjumpingSolver,ACLookAheadSolver, PC2Solver
import time
class CellState(Enum):
    UNKNOWN = 'U'
    MINE = 'M'
    SAFE = 'S'

class MinesweeperConstraint(Constraint):
    def __init__(self, center_var: Variable, neighbor_vars: List[Variable], mine_count: int):
        """
        Constraint for a numbered cell and its neighbors
        center_var: The numbered cell
        neighbor_vars: List of neighboring cells
        mine_count: Number of mines that should be around the center cell
        """
        super().__init__([center_var] + neighbor_vars if center_var else neighbor_vars)
        self.center_var = center_var
        self.neighbor_vars = neighbor_vars
        self.mine_count = mine_count
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        # Check if all relevant variables are assigned
        if not all(var.name in assignment for var in self.neighbor_vars):
            return True  # Not all variables assigned yet
        
        # Count mines in neighbor cells
        mine_count = sum(1 for var in self.neighbor_vars 
                        if assignment.get(var.name) == CellState.MINE)
        
        return mine_count == self.mine_count
    
    def get_variables(self) -> List[Variable]:
        return self.variables

class TotalMinesConstraint(Constraint):
    def __init__(self, variables: List[Variable], mine_count: int):
        super().__init__(variables)
        self.mine_count = mine_count
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        if not all(var.name in assignment for var in self.variables):
            return True
        return sum(1 for val in assignment.values() 
                if val == CellState.MINE) == self.mine_count
    
    def get_variables(self) -> List[Variable]:
        return self.variables


class MinesweeperBoard:
    def __init__(self, width: int, height: int, num_mines: int):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = np.full((height, width), -1)  # -1 for unknown
        self.mine_locations = set()
        self.revealed = np.zeros((height, width), dtype=bool)
        self.initialize_mines()
    
    def initialize_mines(self):
        positions = [(x, y) for x in range(self.height) for y in range(self.width)]
        mine_positions = random.sample(positions, self.num_mines)
        for x, y in mine_positions:
            self.mine_locations.add((x, y))
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.height and 
                    0 <= new_y < self.width):
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def get_cell_value(self, x: int, y: int) -> int:
        if (x, y) in self.mine_locations:
            return -1
        
        count = 0
        for nx, ny in self.get_neighbors(x, y):
            if (nx, ny) in self.mine_locations:
                count += 1
        return count
    
    def reveal(self, x: int, y: int) -> bool:
        if (x, y) in self.mine_locations:
            return False
        
        if self.revealed[x, y]:
            return True
        
        self.revealed[x, y] = True
        value = self.get_cell_value(x, y)
        self.board[x, y] = value
        
        if value == 0:
            for nx, ny in self.get_neighbors(x, y):
                if not self.revealed[nx, ny]:
                    self.reveal(nx, ny)
        
        return True
    
    def display_board(self, solution: Optional[Dict[Tuple[int, int], CellState]] = None):
        print("\n  " + " ".join(str(i) for i in range(self.width)))
        print("  " + "-" * (self.width * 2 - 1))
        
        for i in range(self.height):
            row = [f"{i}|"]
            for j in range(self.width):
                if self.revealed[i, j]:
                    if self.board[i, j] == 0:
                        row.append(".")
                    else:
                        row.append(str(self.board[i, j]))
                elif solution and (i, j) in solution:
                    row.append("M" if solution[(i, j)] == CellState.MINE else "S")
                else:
                    row.append("?")
            print(" ".join(row))
        print()

    
    def __str__(self) -> str:
        result = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if not self.revealed[i, j]:
                    row.append('?')
                elif self.board[i, j] == -1:
                    row.append('M')
                elif self.board[i, j] == 0:
                    row.append('.')
                else:
                    row.append(str(self.board[i, j]))
            result.append(' '.join(row))
        return '\n'.join(result)

class MinesweeperCSP:
    def __init__(self, board):
        self.board = board
        self.csp = CSP()
        self.analyzer = None
        self.mine_locations = board.mine_locations
        self.num_mines = board.num_mines
        self.metrics = defaultdict(list)
        self.initialize_csp()
        self.solvers = [
            BacktrackingSolver(timeout=1800),
            AC3Solver(timeout=1800),
            PC2Solver(timeout=1800),
            ForwardCheckingSolver(timeout=1800),
            BackjumpingSolver(timeout=1800),
            ACLookAheadSolver(timeout=1800)
        ]
        self.analyzer = CSPAnalyzer(self.csp)

    
    def initialize_csp(self):
        # Create variables for unrevealed cells
        for x in range(self.board.height):
            for y in range(self.board.width):
                if not self.board.revealed[x, y]:
                    var_name = f"cell_{x}_{y}"
                    var = Variable(var_name, {CellState.MINE, CellState.SAFE})
                    self.csp.add_variable(var)
        
        # Add number constraints
        for x in range(self.board.height):
            for y in range(self.board.width):
                if self.board.revealed[x, y] and self.board.board[x, y] > 0:
                    neighbor_vars = []
                    for nx, ny in self.board.get_neighbors(x, y):
                        if not self.board.revealed[nx, ny]:
                            var = self.csp.variables[f"cell_{nx}_{ny}"]
                            neighbor_vars.append(var)
                    
                    if neighbor_vars:
                        constraint = MinesweeperConstraint(
                            None, neighbor_vars, self.board.board[x, y]
                        )
                        self.csp.add_constraint(constraint)
        
        # Add total mines constraint
        remaining = self.board.num_mines - sum(
            1 for x, y in self.board.mine_locations 
            if self.board.revealed[x, y]
        )
        
        total_constraint = TotalMinesConstraint(
            list(self.csp.variables.values()),
            remaining
        )
        self.csp.add_constraint(total_constraint)


    def valid(self, solution: Dict[str, Any]) -> bool:
        # Convert solution format to coordinate mapping
        coord_solution = {}
        for var_name, state in solution.items():
            x, y = map(int, var_name.split('_')[1:])
            coord_solution[(x, y)] = state
            
        # Validate total number of mines
        mine_count = sum(1 for state in coord_solution.values() if state == CellState.MINE)
        remaining_mines = self.board.num_mines - sum(
            1 for x, y in self.board.mine_locations 
            if self.board.revealed[x, y]
        )
        if mine_count != remaining_mines:
            return False
            
        # Validate all number constraints
        for x in range(self.board.height):
            for y in range(self.board.width):
                if self.board.revealed[x, y] and self.board.board[x, y] > 0:
                    adjacent_mines = 0
                    for nx, ny in self.board.get_neighbors(x, y):
                        if not self.board.revealed[nx, ny]:
                            if coord_solution.get((nx, ny)) == CellState.MINE:
                                adjacent_mines += 1
                        elif (nx, ny) in self.board.mine_locations:
                            adjacent_mines += 1
                    
                    if adjacent_mines != self.board.board[x, y]:
                        return False
        
        # Validate against known mine locations
        for x, y in self.mine_locations:
            if not self.board.revealed[x, y] and coord_solution.get((x, y)) != CellState.MINE:
                return False
        
        return True

    def print_constraint_graph_info(self):
        """Print detailed information about the constraint graph structure"""
        print("\nConstraint Graph Analysis:")
        print("-" * 50)
        
        # Get graph analysis metrics
        metrics = self.analyzer.analyze_graph()
        
        # Print basic graph metrics
        print(f"Number of Variables: {metrics['num_variables']}")
        print(f"Number of Constraints: {metrics['num_constraints']}")
        print(f"Graph Density: {metrics['density']:.3f}")
        print(f"Average Degree: {metrics['average_degree']:.2f}")
        print(f"Average Clustering Coefficient: {metrics['average_clustering']:.3f}")
        print(f"Is Connected: {metrics['is_connected']}")
        if metrics['is_connected']:
            print(f"Diameter: {metrics['diameter']}")
        
        
    def print_metrics_summary(self):
        """Print summary of solver metrics"""
        print("\nSolver Performance Summary:")
        print("-" * 100)
        headers = ['Solver', 'Time (s)', 'Checks', 'Revisions', 'Nodes', 'Backtracks', 'Solutions Found']
        print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} "
            f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<15}")
        print("-" * 100)
        
        for solver_name, runs in self.metrics.items():
            if runs:  # Check if there are any runs
                latest_metrics = runs[-1]  # Get the last run
                solutions_found = sum(1 for run in runs if run['solution_found'])
                
                print(f"{solver_name:<15} "
                    f"{latest_metrics['execution_time']:<10.3f} "
                    f"{latest_metrics['constraint_checks']:<10} "
                    f"{latest_metrics['revisions']:<10} "
                    f"{latest_metrics['nodes_explored']:<10} "
                    f"{latest_metrics['backtracks']:<10} "
                    f"{solutions_found:<15}")

    
    def solve(self) -> Dict[Tuple[int, int], CellState]:
        """Solve using multiple solvers and collect metrics"""
        found_solution = None
        invalid_assignments = defaultdict(set)
        max_attempts_per_solver = 1000  # Limit attempts
        solver_timeout = 1800  # Timeout in seconds for each solver's attempts
        
        def assignment_to_tuple(assignment: Dict[str, Any]) -> tuple:
            return tuple(sorted((k, v.value) for k, v in assignment.items()))
        
        # Print constraint graph information
        self.print_constraint_graph_info()
        
        for solver in self.solvers:
            print(f"\nTrying {solver.name}...")
            invalid_assignments[solver.name] = set()
            valid_solution_found = False
            start_time = time.time()
            attempts = 0
            
            while not valid_solution_found and attempts < max_attempts_per_solver:
                # Check total time for this solver
                if time.time() - start_time > solver_timeout:
                    print(f"{solver.name} timed out after {solver_timeout} seconds")
                    break
                    
                attempts += 1
                csp_copy = deepcopy(self.csp)
                
                try:
                    solution = solver.solve(csp_copy)
                    metrics = solver.get_metrics()
                    
                    # Record metrics for this attempt
                    self.metrics[solver.name].append({
                        'execution_time': time.time() - start_time,
                        'constraint_checks': metrics.constraint_checks,
                        'revisions': metrics.revisions,
                        'nodes_explored': metrics.nodes_explored,
                        'backtracks': metrics.backtracks,
                        'solution_found': False,
                        'timed_out': metrics.timed_out
                    })
                    
                    if metrics.timed_out:
                        print(f"{solver.name} attempt {attempts} timed out")
                        continue
                    
                    if solution:
                        solution_tuple = assignment_to_tuple(solution)
                        if solution_tuple in invalid_assignments[solver.name]:
                            continue
                        
                        if self.valid(solution):
                            valid_solution_found = True
                            self.metrics[solver.name][-1]['solution_found'] = True
                            if not found_solution:
                                found_solution = solution
                                print(f"{solver.name} found valid solution on attempt {attempts}")
                            break
                        else:
                            invalid_assignments[solver.name].add(solution_tuple)
                    else:
                        # print(f"{solver.name} found no solution on attempt {attempts}")
                        break
                
                except Exception as e:
                    print(f"Error in {solver.name} attempt {attempts}: {e}")
                    continue
                
                if len(invalid_assignments[solver.name]) % 100 == 0 and len(invalid_assignments[solver.name]) > 0:
                    print(f"{solver.name} - Invalid solutions: {len(invalid_assignments[solver.name])}")
        
        # Print final performance summary
        self.print_metrics_summary()
        
        if found_solution:
            result = {}
            for var_name, state in found_solution.items():
                x, y = map(int, var_name.split('_')[1:])
                result[(x, y)] = state
            return result
        
        return None



def solve_minesweeper(width: int, height: int, num_mines: int, 
                     initial_moves: List[Tuple[int, int]] = None) -> bool:
    board = MinesweeperBoard(width, height, num_mines)
    
    if initial_moves:
        for x, y in initial_moves:
            if not board.reveal(x, y):
                print("Hit a mine in initial move!")
                return False
    
    print("Initial board state:")
    board.display_board()
    
    solver = MinesweeperCSP(board)
    solution = solver.solve()
    
    if not solution:
        print("\nNo solution found!")
        return False
    
    # Verify solution
    correct = True
    for (x, y), state in solution.items():
        if state == CellState.MINE and (x, y) not in board.mine_locations:
            correct = False
        elif state == CellState.SAFE and (x, y) in board.mine_locations:
            correct = False
    
    print("\nFinal board state with solution:")
    board.display_board(solution)
    
    
    print(f"\nSolution is {'correct' if correct else 'incorrect'}!")
    return correct


if __name__ == "__main__":
    success = solve_minesweeper(
        width=8,
        height=8,
        num_mines=6,
        initial_moves=[(0, 0), (3, 3)]
    )
