from typing import Dict, Any
from collections import defaultdict
import networkx as nx
from base import CSP
from solvers import (
    BacktrackingSolver,
    ForwardCheckingSolver,
    AC3Solver,
    PC2Solver,
    ACLookAheadSolver,
    BackjumpingSolver
)


class CSPAnalyzer:
    def __init__(self, csp: CSP):
        self.original_csp = csp
        self.results = defaultdict(list)
        self.attempt_counts = defaultdict(int)
        self.solvers = [
            BacktrackingSolver(),
            ForwardCheckingSolver(),
            AC3Solver(),
            ACLookAheadSolver(),
            BackjumpingSolver(),
            PC2Solver
        ]
        
    def create_constraint_graph(self) -> nx.Graph:
        """Creates the constraint Graph"""
        G = nx.Graph()
        
        # Add nodes (variables)
        for var_name in self.original_csp.variables:
            G.add_node(var_name)
        
        # Add edges (constraints)
        for constraint in self.original_csp.constraints:
            variables = constraint.get_variables()
            for i, var1 in enumerate(variables):
                for var2 in variables[i+1:]:
                    G.add_edge(var1.name, var2.name)
        
        return G

    def analyze_graph(self) -> Dict[str, Any]:
        """Analyzes the constraint graph"""
        G = self.create_constraint_graph()
        
        analysis = {
            'num_variables': len(G.nodes),
            'num_constraints': len(G.edges),
            'density': nx.density(G),
            'average_degree': sum(dict(G.degree()).values()) / len(G),
            'is_connected': nx.is_connected(G),
            'average_clustering': nx.average_clustering(G)
        }
        
        # Add diameter if graph is connected
        if analysis['is_connected']:
            analysis['diameter'] = nx.diameter(G)
        
        return analysis
    
    def get_best_solver(self):
        """Designed with flexibility in mind. If we allow multiple solutions, we grab the solver with the highest success rate.
        In our case, however, as we stop after the first solution, this instead returns the inverse of the number of tries it took to 
        successfully find the first solution, or 0, if none are found
        """
        best_solver = None
        best_ratio = -1
        
        for solver in self.solvers:
            runs = self.results[solver.name]
            solutions = sum(1 for run in runs if run['solution_found'])
            attempts = self.attempt_counts[solver.name]
            ratio = solutions / attempts if attempts > 0 else 0
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_solver = solver
                
        return best_solver, best_ratio
