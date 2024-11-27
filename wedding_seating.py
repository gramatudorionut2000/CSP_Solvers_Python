from typing import List, Set, Dict, Tuple, Any, Optional
from base import Variable, CSP, Constraint
import random
from solvers import BacktrackingSolver, AC3Solver, ForwardCheckingSolver, BackjumpingSolver, ACLookAheadSolver, PC2Solver, CSPMetrics
from collections import defaultdict
from copy import deepcopy
from analysis import CSPAnalyzer

class CapacityConstraint(Constraint):
    def __init__(self, variables: List[Variable], num_tables: int, seats_per_table: int):
        super().__init__(variables)
        self.num_tables = num_tables
        self.seats_per_table = seats_per_table
        
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        table_counts = defaultdict(int)
        for _, table in assignment.items():
            table_counts[table] += 1
            if table_counts[table] > self.seats_per_table:
                return False
        return True

    def get_variables(self) -> List[Variable]:
        return self.variables

    
class RelativeConstraint(Constraint):
    """relatives seated at the same table"""
    def __init__(self, variables: List[Variable]):
        super().__init__(variables)
        self.relative_pairs: Set[Tuple[str, str]] = set()
        
    def add_relative_pair(self, guest1: str, guest2: str):
        self.relative_pairs.add(tuple(sorted([guest1, guest2])))
        
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        for guest1, guest2 in self.relative_pairs:
            if guest1 in assignment and guest2 in assignment:
                if assignment[guest1] != assignment[guest2]:
                    return False
        return True

    def get_variables(self) -> List[Variable]:
        return self.variables


    
    def get_variables(self) -> List[Variable]:
        return self.variables

class EnemyConstraint(Constraint):
    """enemies seated at different tables"""
    def __init__(self, variables: List[Variable]):
        super().__init__(variables)
        self.enemy_pairs: Set[Tuple[str, str]] = set()
        
    def add_enemy_pair(self, guest1: str, guest2: str):
        self.enemy_pairs.add(tuple(sorted([guest1, guest2])))
        
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        for guest1, guest2 in self.enemy_pairs:
            if guest1 in assignment and guest2 in assignment:
                if assignment[guest1] == assignment[guest2]:
                    return False
        return True
    
    def get_variables(self) -> List[Variable]:
        return self.variables

class WeddingSeatingCSP:
    def __init__(self, num_guests: int, num_tables: int, seats_per_table: int):
        self.num_guests = num_guests
        self.num_tables = num_tables
        self.seats_per_table = seats_per_table
        self.csp = CSP()
        self.relative_constraint = None
        self.enemy_constraint = None
        self.relative_pairs = set()
        self.analyzer = None
        self.enemy_pairs = set()
        self.metrics: Dict[str, CSPMetrics] = {}
        self.initialize_csp()


    
    def initialize_csp(self):
        variables = []
        for i in range(self.num_guests):
            var_name = f"Guest{i}"
            domain = set(range(self.num_tables))
            var = Variable(var_name, domain)
            self.csp.add_variable(var)
            variables.append(var)
        
        table_capacity = CapacityConstraint(variables, self.num_tables, self.seats_per_table)
        self.relative_constraint = RelativeConstraint(variables)
        self.enemy_constraint = EnemyConstraint(variables)
        
        self.csp.add_constraint(table_capacity)
        self.csp.add_constraint(self.relative_constraint)
        self.csp.add_constraint(self.enemy_constraint)

        self.analyzer = CSPAnalyzer(self.csp)



    
    def add_relatives(self, guest1: int, guest2: int):
        guest1_name = f"Guest{guest1}"
        guest2_name = f"Guest{guest2}"
        self.relative_constraint.add_relative_pair(guest1_name, guest2_name)
        self.relative_pairs.add(tuple(sorted([guest1, guest2])))

    
    def add_enemies(self, guest1: int, guest2: int):
        guest1_name = f"Guest{guest1}"
        guest2_name = f"Guest{guest2}"
        self.enemy_constraint.add_enemy_pair(guest1_name, guest2_name)
        self.enemy_pairs.add(tuple(sorted([guest1, guest2])))

    def generate_random_pairs(self, num_relative_pairs: int, num_enemy_pairs: int):
        """Generate random relative and enemy pairs"""
        all_pairs = {
            tuple(sorted([i, j]))
            for i in range(self.num_guests)
            for j in range(i + 1, self.num_guests)
        }
        
        # Select random pairs for relatives
        available = list(all_pairs)
        for _ in range(min(num_relative_pairs, len(available))):
            if not available:
                break
            pair = random.choice(available)
            self.add_relatives(*pair)
            available.remove(pair)
            all_pairs.remove(pair)
        

        available = list(all_pairs)
        for _ in range(min(num_enemy_pairs, len(available))):
            if not available:
                break
            pair = random.choice(available)
            self.add_enemies(*pair)
            available.remove(pair)


    
    def is_valid_solution(self, solution: Dict[str, Any]) -> bool:
        if not solution or len(solution) != self.num_guests:
            return False
        
        table_counts = defaultdict(int)
        for _, table in solution.items():
            table_counts[table] += 1
            if table_counts[table] > self.seats_per_table:
                return False
            
        # Check relative constraints
        for pair in self.relative_pairs:
            guest1_name = f"Guest{pair[0]}"
            guest2_name = f"Guest{pair[1]}"
            if solution[guest1_name] != solution[guest2_name]:
                return False
                
        # Check enemy constraints
        for pair in self.enemy_pairs:
            guest1_name = f"Guest{pair[0]}"
            guest2_name = f"Guest{pair[1]}"
            if solution[guest1_name] == solution[guest2_name]:
                return False
                
        return True


    def print_constraint_graph_info(self):
        """Print information about the constraint graph structure"""
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
        
        # Print constraint relationships
        print("\nConstraint Relationships:")
        print("-" * 25)
        print(f"Number of Relative Pairs: {len(self.relative_pairs)}")
        print(f"Number of Enemy Pairs: {len(self.enemy_pairs)}")
        
        # Print variable connectivity
        print("\nVariable Connectivity:")
        print("-" * 25)
        G = self.analyzer.create_constraint_graph()
        for node in sorted(G.nodes()):
            neighbors = sorted(G.neighbors(node))
            print(f"{node}: Connected to {len(neighbors)} variables")
            print(f"   Neighbors: {', '.join(neighbors)}")



    def solve(self) -> Optional[Dict[str, Tuple[int, int]]]:
        found_solution = None
        
        #constraint graph information
        self.print_constraint_graph_info()
        
        solvers = [
            BacktrackingSolver(),
            AC3Solver(),
            PC2Solver(),
            ACLookAheadSolver(),
            ForwardCheckingSolver(),
            BackjumpingSolver()
        ]

        print("\nSolver Performance and Solutions:")
        print("=" * 100)

        for solver in solvers:
            print(f"\n{solver.name} Solver:")
            print("-" * 50)
            
            csp_copy = deepcopy(self.csp)
            
            # Solve and get metrics
            solution = solver.solve(csp_copy)
            metrics = solver.get_metrics()
            self.metrics[solver.name] = metrics
            
            # Print metrics
            print(f"Time: {metrics.execution_time:.3f}s")
            print(f"Constraint checks: {metrics.constraint_checks}")
            print(f"Domain revisions: {metrics.revisions}")
            print(f"Nodes explored: {metrics.nodes_explored}")
            print(f"Backtracks: {'-' if metrics.backtracks == 0 else metrics.backtracks}")
            
            # If solution is valid, keep track of it
            if solution and self.is_valid_solution(solution):
                print("\nValid solution found:")
                self._print_solution(solution)
                if found_solution is None:
                    found_solution = solution
            else:
                print("\nNo valid solution found")
            
            print("-" * 50)

        self.print_metrics_summary()
        return found_solution



    def _print_solution(self, solution: Dict[str, int]):
        tables = defaultdict(list)
        for guest, table in solution.items():
            tables[table].append(guest)
        
        print("\nSeating Arrangement:")
        for table in range(self.num_tables):
            print(f"\nTable {table}:")
            if table in tables:
                for guest in sorted(tables[table]):
                    print(f"  {guest}")
            else:
                print("  (Empty)")


    def print_metrics_summary(self):
        print("\nSolver Performance Summary:")
        print("-" * 100)
        
        headers = ['Solver', 'Time (s)', 'Checks', 'Revisions', 'Nodes', 'Backtracks']
        print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} "
              f"{headers[4]:<10} {headers[5]:<10}")
        print("-" * 100)
        
        for solver_name, metrics in self.metrics.items():
            print(f"{solver_name:<15} {metrics.execution_time:<10.3f} "
                  f"{metrics.constraint_checks:<10} {metrics.revisions:<10} "
                  f"{metrics.nodes_explored:<10} {'-' if metrics.backtracks == 0 else metrics.backtracks:<10}")


    def print_relationship_pairs(self):
        print("\nRelationship Constraints:")
        print("-" * 50)
        
        print("\nRelative Pairs (must sit at same table):")
        if self.relative_pairs:
            for guest1, guest2 in sorted(self.relative_pairs):
                print(f"Guests {guest1} and {guest2}")
        else:
            print("No relative pairs defined")
            
        print("\nEnemy Pairs (must sit at different tables):")
        if self.enemy_pairs:
            for guest1, guest2 in sorted(self.enemy_pairs):
                print(f"Guests {guest1} and {guest2}")
        else:
            print("No enemy pairs defined")


    
    def print_seating_arrangement(self, solution: Dict[str, Tuple[int, int]]):
        if not solution:
            print("No solution found!")
            return
            
        tables = {}
        for guest, (table, seat) in solution.items():
            if table not in tables:
                tables[table] = {}
            tables[table][seat] = guest
            
        print("\nSeating Arrangement:")
        print("=" * 50)
        
        for table in range(self.num_tables):
            print(f"\nTable {table}:")
            print("-" * 20)
            if table in tables:
                for seat in range(self.seats_per_table):
                    guest = tables[table].get(seat, "Empty")
                    print(f"Seat {seat}: {guest}")
            else:
                print("(Empty table)")


def analyze_constraint_graph(csp: CSP) -> Dict[str, float]:
    """Analyze the constraint graph structure"""
    variables = list(csp.variables.keys())
    n = len(variables)
    graph = {var: set() for var in variables}
    
    # Build graph from constraints - a pair of variables is connected if they appear 
    # together in any constraint
    for constraint in csp.constraints:
        vars_in_constraint = [var.name for var in constraint.get_variables()]
        
        # For each constraint, all variables involved are connected to each other
        for i, v1 in enumerate(vars_in_constraint):
            for v2 in vars_in_constraint[i+1:]:
                graph[v1].add(v2)
                graph[v2].add(v1)
    
    num_variables = len(variables)
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    max_possible_edges = (num_variables * (num_variables - 1)) // 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    total = sum(len(neighbors) for neighbors in graph.values())
    avg_degree = total / num_variables if num_variables > 0 else 0
    
    clustering_coeff = 0
    for var in variables:
        neighbors = graph[var]
        if len(neighbors) > 1:
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = sum(1 for n1 in neighbors for n2 in neighbors 
                             if n1 < n2 and n2 in graph[n1])
            if possible_edges > 0:
                clustering_coeff += actual_edges / possible_edges
    avg_clustering = clustering_coeff / num_variables if num_variables > 0 else 0
    
    constraint_types = {}
    for constraint in csp.constraints:
        constraint_type = constraint.__class__.__name__
        constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1
    
    return {
        'Number of Variables': num_variables,
        'Number of Edges': num_edges,
        'Graph Density': density,
        'Average Degree': avg_degree,
        'Average Clustering': avg_clustering,
        'Constraint Types': constraint_types
    }


def print_graph_analysis(metrics: Dict[str, float]):
    print("\nConstraint Graph Analysis:")
    print("-" * 50)
    
    # Print numerical metrics
    numerical_metrics = ['Number of Variables', 'Number of Constraints', 'Graph Density', 
                        'Average Degree', 'Average Clustering']
    for metric in numerical_metrics:
        value = metrics[metric]
        if isinstance(value, float):
            print(f"{metric:<25}: {value:.3f}")
        else:
            print(f"{metric:<25}: {value}")



def compare_solvers(problem: 'WeddingSeatingCSP'):
    graph_metrics = analyze_constraint_graph(problem.csp)
    print_graph_analysis(graph_metrics)
    


    print("\nSolver Performance Comparison:")
    print("-" * 100)
    headers = ['Solver', 'Time (s)', 'Checks', 'Revisions', 'Nodes', 'Backtracks', 'Solution Found']
    print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} "
          f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<15}")
    print("-" * 100)

    for solver in problem.solvers:
        csp_copy = deepcopy(problem.csp)
        
        solution = solver.solve(csp_copy)
        metrics = solver.get_metrics()
        
        print(f"{solver.name:<20} {metrics.execution_time:<10.3f} {metrics.constraint_checks:<10} "
              f"{metrics.revisions:<10} {metrics.nodes_explored:<10} {metrics.backtracks:<10} "
              f"{solution is not None:<15}")
        
        if solution:
            print(f"\nSolution found by {solver.name}:")
            print_seating_arrangement(solution, problem.num_tables, 
                                   problem.seats_per_table)
            print()



def solve_wedding_seating(num_guests: int, num_tables: int, seats_per_table: int,
                         num_relative_pairs: int = 0, num_enemy_pairs: int = 0) -> bool:
    seating = WeddingSeatingCSP(num_guests, num_tables, seats_per_table)
    seating.generate_random_pairs(num_relative_pairs, num_enemy_pairs)
    
    print("\nProblem Configuration:")
    print(f"Guests: {num_guests}")
    print(f"Tables: {num_tables}")
    print(f"Seats per table: {seats_per_table}")
    seating.print_relationship_pairs()
    
    solution = seating.solve()
    if solution:
        print("\nSolution found!")

        return True
    
    print("\nNo solution found!")

    return False



if __name__ == "__main__":
    solve_wedding_seating(
    num_guests=20,
    num_tables=4,
    seats_per_table=5,
    num_relative_pairs=0,
    num_enemy_pairs=0
)
