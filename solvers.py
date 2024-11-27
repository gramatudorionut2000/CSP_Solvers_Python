from typing import Dict, Set, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from collections import defaultdict
import time
from base import CSP, Variable
class CSPMetrics:
    def __init__(self):
        self.constraint_checks = 0
        self.revisions = 0
        self.nodes_explored = 0
        self.backtracks = 0
        self.start_time = None
        self.end_time = None
        self.solution_found = False
        self.timed_out = False
        
    def start_timer(self):
        self.start_time = time.time()
    
    def stop_timer(self):
        self.end_time = time.time()
    
    @property
    def execution_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class BaseSolver(ABC):
    def __init__(self, name: str, timeout: float = 1800):  # 30 minutes timeout
        self.name = name
        self.metrics = CSPMetrics()
        self.timeout = timeout
    
    def check_timeout(self) -> bool:
        """Check if solver has exceeded timeout limit"""
        if self.metrics.start_time and time.time() - self.metrics.start_time > self.timeout:
            self.metrics.timed_out = True
            return True
        return False
    
    @abstractmethod
    def solve(self, csp: 'CSP') -> Optional[Dict[str, Any]]:
        pass
    
    def get_metrics(self) -> CSPMetrics:
        return self.metrics


class BacktrackingSolver(BaseSolver):
    def __init__(self, timeout: float = 1800):
        super().__init__("Backtracking", timeout)

    
    def solve(self, csp: 'CSP') -> Optional[Dict[str, Any]]:
        self.metrics = CSPMetrics()
        self.metrics.start_timer()
        result = self._backtrack(csp, {})
        self.metrics.stop_timer()
        return result

    
    def _backtrack(self, csp: 'CSP', assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.check_timeout():
            return None
            
        self.metrics.nodes_explored += 1
        
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = self._select_unassigned_variable(csp, assignment)
        for value in self._order_domain_values(csp, var):
            self.metrics.constraint_checks += 1
            if self._is_consistent(csp, var.name, value, assignment):
                assignment[var.name] = value
                result = self._backtrack(csp, assignment)
                if result is not None:
                    return result
                assignment.pop(var.name)
                self.metrics.backtracks += 1
        
        return None

    
    def _select_unassigned_variable(self, csp: 'CSP', assignment: Dict[str, Any]):
        for var in csp.variables.values():
            if var.name not in assignment:
                return var
        return None
    
    def _order_domain_values(self, csp: 'CSP', var):
        return list(var.domain)
    
    def _is_consistent(self, csp: 'CSP', var_name: str, value: Any, assignment: Dict[str, Any]) -> bool:
        assignment = assignment.copy()
        assignment[var_name] = value
        
        for constraint in csp.constraints:
            vars_in_constraint = [var.name for var in constraint.variables]
            if var_name in vars_in_constraint:
                if all(v in assignment for v in vars_in_constraint):
                    if not constraint.is_satisfied(assignment):
                        return False
        return True

class AC3Solver(BaseSolver):
    def __init__(self, timeout: float = 1800):
        super().__init__("AC3", timeout)

    
    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        self.metrics = CSPMetrics()
        self.metrics.start_timer()
        
        if not self._ac3(csp):
            self.metrics.stop_timer()
            return None
            
        result = self._backtrack(csp, {})
        
        self.metrics.stop_timer()
        return result

    
    def _ac3(self, csp: CSP) -> bool:
        Q = []
        for constraint in csp.constraints:
            vars_in_constraint = constraint.get_variables()
            if len(vars_in_constraint) == 2:
                x, y = vars_in_constraint
                Q.append((x.name, y.name))
                Q.append((y.name, x.name))
        
        while Q:
            if self.check_timeout():
                return False
                
            x_name, y_name = Q.pop(0)
            self.metrics.revisions += 1
            
            if self._revise(csp, csp.variables[x_name], csp.variables[y_name]):
                if self.check_timeout():
                    return False
        
                if not csp.variables[x_name].domain:
                    return False
                
                neighbors = set()
                for constraint in csp.constraints:
                    vars_in_constraint = constraint.get_variables()
                    if len(vars_in_constraint) == 2:
                        if x_name in [var.name for var in vars_in_constraint]:
                            for var in vars_in_constraint:
                                if var.name != x_name and var.name != y_name:
                                    neighbors.add(var.name)
                
                for z_name in neighbors:
                    Q.append((z_name, x_name))
        
        return True

    
    def _revise(self, csp: CSP, xi: Variable, xj: Variable) -> bool:
        """
        Revises the domain of xi with respect to xj, amd returns True if domain was changed
        """
        revised = False
        to_remove = set()
        
        for x in xi.domain:
            # Check whether there exists a value in xj's domain that satisfies constraints, for each value in xi's domain
            satisfiable = False
            for y in xj.domain:
                self.metrics.constraint_checks += 1
                
                # Check that all relevant constraints are satisfied
                assignment = {xi.name: x, xj.name: y}
                if all(constraint.is_satisfied(assignment)
                       for constraint in csp.constraints
                       if set(v.name for v in constraint.variables) == {xi.name, xj.name}):
                    satisfiable = True
                    break
            
            if not satisfiable:
                to_remove.add(x)
                revised = True
        
        # Remove unsupported values
        xi.domain -= to_remove
        return revised
    
    def _backtrack(self, csp: CSP, assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.check_timeout():
            return False

        self.metrics.nodes_explored += 1
        
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = self._select_unassigned_variable(csp, assignment)
        
        # Try each value in its domain
        for value in var.domain:
            self.metrics.constraint_checks += 1
            
            if self._is_consistent(csp, var.name, value, assignment):
                assignment[var.name] = value
                result = self._backtrack(csp, assignment)
                if result is not None:
                    return result
                self.metrics.backtracks += 1
                assignment.pop(var.name)
        
        return None
    
    def _select_unassigned_variable(self, csp: CSP, assignment: Dict[str, Any]) -> Variable:
        """Selects the first unassigned variable"""
        for var_name, var in csp.variables.items():
            if var_name not in assignment:
                return var
        return None
    
    def _is_consistent(self, csp: CSP, var_name: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Checks if the current assignment is consistent with constraints"""
        test_assignment = assignment.copy()
        test_assignment[var_name] = value
        
        for constraint in csp.constraints:
            vars_in_constraint = [v.name for v in constraint.variables]
            if var_name in vars_in_constraint:
                if all(v in test_assignment for v in vars_in_constraint):
                    if not constraint.is_satisfied(test_assignment):
                        return False
        return True

class PC2Solver(BaseSolver):
    def __init__(self, timeout: float = 1800):
        super().__init__("PC2", timeout)

    
    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        self.metrics = CSPMetrics()
        self.metrics.start_timer()
        
        if not self._pc2(csp):
            self.metrics.stop_timer()
            return None
            
        assignment = {}
        result = self._backtrack(csp, assignment)
        
        self.metrics.stop_timer()
        return result
    
    def _pc2(self, csp: CSP) -> bool:
        # Get ordering of variables
        if self.check_timeout():
            return False

        ordering = list(csp.variables.keys())
        n = len(ordering)
        
        # Initialize queue with all possible paths of length 2, (i,k,j) means path from i to j through k
        Q = [(i, k, j) for i in ordering 
                      for k in ordering 
                      for j in ordering 
                      if i != j and i != k and j != k]
        
        while Q:
            if self.check_timeout():
                return False
            
            i, k, j = Q.pop(0)
            self.metrics.revisions += 1
            
            if self._revise_constraint(csp, i, k, j):
                if not csp.variables[i].domain or not csp.variables[j].domain:
                    return False
                    
                new_paths = self._related_paths(i, k, j, ordering)
                Q.extend(path for path in new_paths if path not in Q)
        
        return True
    
    def _revise_constraint(self, csp: CSP, i: str, k: str, j: str) -> bool:
        """Revises constraint between variables i and j using k"""

        if self.check_timeout():
            return False

        changed = False
        var_i = csp.variables[i]
        var_j = csp.variables[j]
        var_k = csp.variables[k]
        
        to_remove_i = set()
        to_remove_j = set()
        
        # Check each value in domain of i
        for a in var_i.domain:
            supported = False
            for c in var_k.domain:
                for b in var_j.domain:
                    self.metrics.constraint_checks += 1
                    assignment = {i: a, k: c, j: b}
                    
                    # Check if this combination satisfies all relevant constraints
                    if all(constraint.is_satisfied(assignment)
                           for constraint in csp.constraints
                           if set(v.name for v in constraint.variables).issubset(assignment.keys())):
                        supported = True
                        break
                if supported:
                    break
            
            if not supported:
                to_remove_i.add(a)
                changed = True
        
        # Remove unsupported values
        var_i.domain -= to_remove_i
        
        return changed
    
    def _related_paths(self, i: str, k: str, j: str, ordering: List[str]) -> List[Tuple[str, str, str]]:
        paths = []
        
        if i < j:  # The first case
            for x in ordering:
                if x not in {i, j, k}:
                    paths.append((i, j, x))
                    paths.append((x, i, j))
                    paths.append((i, x, j))
        elif i == j: # The second case
            for x in ordering:
                if x != i and x != k:
                    for y in ordering:
                        if y != i:
                            paths.append((x, i, y))
        
        return paths
    
    def _backtrack(self, csp: CSP, assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.check_timeout():
            return False

        self.metrics.nodes_explored += 1
        
        if len(assignment) == len(csp.variables):
            return assignment
            
        var = self._select_unassigned_variable(csp, assignment)
        
        # We go through each value in a variable domain and check the consistency at each step
        for value in var.domain:
            self.metrics.constraint_checks += 1
            if self._is_consistent(csp, var.name, value, assignment):
                assignment[var.name] = value
                result = self._backtrack(csp, assignment)
                if result is not None:
                    return result
                assignment.pop(var.name)
                self.metrics.backtracks += 1
                
        return None
    
    def _is_consistent(self, csp: CSP, var_name: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Checks if assignment is consistent with constraints"""
        test_assignment = assignment.copy()
        test_assignment[var_name] = value
        
        for constraint in csp.constraints:
            if var_name in [var.name for var in constraint.variables]:
                if all(v.name in test_assignment for v in constraint.variables):
                    if not constraint.is_satisfied(test_assignment):
                        return False
        return True
    
    def _select_unassigned_variable(self, csp: CSP, assignment: Dict[str, Any]) -> Variable:
        """Selects the first unassigned variable"""
        for var_name, var in csp.variables.items():
            if var_name not in assignment:
                return var
        return None

class ACLookAheadSolver(BaseSolver):
    def __init__(self, timeout: float = 1800):
        super().__init__("AC-Lookahead", timeout)
    
    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        self.metrics = CSPMetrics()
        self.metrics.start_timer()
        
        # Create a deep copy of domains to restore if needed
        self.original_domains = {var.name: var.domain.copy() for var in csp.variables.values()}
        result = self._ac_lookahead(csp, {})
        
        self.metrics.stop_timer()
        return result
    
    def _ac_lookahead(self, csp: CSP, assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.check_timeout():
            return None

        self.metrics.nodes_explored += 1
        
        if len(assignment) == len(csp.variables):
            return assignment
        
        # Select unassigned variable using minimum remaining values
        var = self._select_unassigned_variable(csp, assignment)
        if not var:
            return None
            
        # Try each value in the current domain
        for value in list(var.domain):
            self.metrics.constraint_checks += 1
            
            # Save current domains
            saved_domains = {name: var.domain.copy() 
                           for name, var in csp.variables.items()}
            
            # Assign value and restrict domain
            assignment[var.name] = value
            var.domain = {value}
            
            # Check if assignment is consistent with current constraints
            if self._is_consistent(csp, var.name, value, assignment):
                # Maintain AC in future variables
                if self._maintain_arc_consistency(csp, var.name, assignment):
                    result = self._ac_lookahead(csp, assignment)
                    if result is not None:
                        return result
            
            # Restore domains and backtrack
            for name, domain in saved_domains.items():
                csp.variables[name].domain = domain.copy()
            assignment.pop(var.name)
            self.metrics.backtracks += 1
        
        return None
    
    def _maintain_arc_consistency(self, csp: CSP, current_var: str, 
                                assignment: Dict[str, Any]) -> bool:
        """Maintain arc consistency for future variables"""
        if self.check_timeout():
            return None
        queue = []
        
        # Add arcs from current variable to all unassigned variables
        for var2 in csp.variables.values():
            if var2.name not in assignment and var2.name != current_var:
                queue.append((current_var, var2.name))
                queue.append((var2.name, current_var))
        
        # Add arcs between all pairs of unassigned variables
        unassigned = [var.name for var in csp.variables.values() 
                     if var.name not in assignment]
        for i in range(len(unassigned)):
            for j in range(i + 1, len(unassigned)):
                queue.append((unassigned[i], unassigned[j]))
                queue.append((unassigned[j], unassigned[i]))
        
        while queue:
            xi_name, xj_name = queue.pop(0)
            self.metrics.revisions += 1
            
            if self._revise(csp, csp.variables[xi_name], 
                          csp.variables[xj_name], assignment):
                if not csp.variables[xi_name].domain:
                    return False
                
                # Add neighbors for re-revision
                if xi_name not in assignment:
                    for xk_name in csp.variables:
                        if (xk_name not in assignment and 
                            xk_name != xi_name and 
                            xk_name != xj_name):
                            queue.append((xk_name, xi_name))
        
        return True
    
    def _revise(self, csp: CSP, xi: Variable, xj: Variable, 
                assignment: Dict[str, Any]) -> bool:
        """Revise domain of xi with respect to xj"""
        if self.check_timeout():
            return None
        revised = False
        to_remove = set()
        
        for x in xi.domain:
            # Skip if this variable is already assigned
            if xi.name in assignment and assignment[xi.name] != x:
                continue
                
            satisfiable = False
            for y in xj.domain:
                # Skip if xj is assigned and this isn't its value
                if xj.name in assignment and assignment[xj.name] != y:
                    continue
                    
                # Check if this value pair is consistent
                test_assignment = assignment.copy()
                test_assignment[xi.name] = x
                test_assignment[xj.name] = y
                
                if all(constraint.is_satisfied(test_assignment)
                       for constraint in csp.constraints
                       if all(var.name in test_assignment 
                            for var in constraint.get_variables())):
                    satisfiable = True
                    break
            
            if not satisfiable:
                to_remove.add(x)
                revised = True
        
        # Remove unsupported values
        xi.domain -= to_remove
        return revised

    def _is_consistent(self, csp: CSP, var_name: str, value: Any, 
                      assignment: Dict[str, Any]) -> bool:
        """Check if assignment is consistent with constraints"""
        test_assignment = assignment.copy()
        
        for constraint in csp.constraints:
            vars_in_constraint = [var.name for var in constraint.get_variables()]
            if var_name in vars_in_constraint:
                if all(v in test_assignment for v in vars_in_constraint):
                    if not constraint.is_satisfied(test_assignment):
                        return False
        return True
    
    def _select_unassigned_variable(self, csp: CSP, 
                                  assignment: Dict[str, Any]) -> Optional[Variable]:
        """Select unassigned variable with minimum remaining values"""
        unassigned = [var for var in csp.variables.values() 
                     if var.name not in assignment]
        return min(unassigned, key=lambda var: len(var.domain)) if unassigned else None


class BackjumpingSolver(BaseSolver):
    def __init__(self, timeout: float = 1800):
        super().__init__("Backjumping", timeout)

        
    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        self.metrics = CSPMetrics()
        self.metrics.start_timer()
        
        # Initialize max_check array
        # "max_check[i] stores the number of the highest variable that was checked 
        # against the current instantiation of xi"
        self.max_check = {var.name: -1 for var in csp.variables.values()}
        
        # Create mapping from variable names to their order
        self.var_to_index = {var_name: idx for idx, var_name in enumerate(csp.variables.keys())}
        self.index_to_var = {idx: var_name for idx, var_name in enumerate(csp.variables.keys())}
        
        # Initialize assignment
        assignment = {}
        result = self._bj(0, assignment, csp)
        
        self.metrics.stop_timer()
        return result if isinstance(result, dict) else None
    
    def _bj(self, current: int, assignment: Dict[str, Any], csp: CSP) -> Union[Dict[str, Any], int]:

        if self.check_timeout():
            return current - 1

        self.metrics.nodes_explored += 1
        
        # If all variables assigned, return solution
        if len(assignment) == len(csp.variables):
            return assignment
        
        var_name = self.index_to_var[current]
        var = csp.variables[var_name]
        
        # Store previous max_check value
        prev_max_check = self.max_check[var_name]
        
        # Try each value in the domain
        for value in var.domain:
            self.metrics.constraint_checks += 1
            
            # Reset max_check for this variable
            self.max_check[var_name] = -1
            
            # Check if assignment is consistent
            if self._is_consistent(csp, var, value, assignment):
                assignment[var_name] = value
                
                # Recursively try to complete the assignment
                result = self._bj(current + 1, assignment, csp)
                
                # If solution found, return it
                if isinstance(result, dict):
                    return result
                
                # If we get a backjump level
                if isinstance(result, int):
                    # If backjump level is less than current level, keep backjumping
                    if result < current:
                        self.metrics.backtracks += 1
                        self.max_check[var_name] = prev_max_check
                        return result
                    
                assignment.pop(var_name)
            
        # All values failed - backjump to deepest conflict
        self.metrics.backtracks += 1
        return self.max_check[var_name] if self.max_check[var_name] >= 0 else current - 1
    
    def _is_consistent(self, csp: CSP, var: Variable, value: Any, assignment: Dict[str, Any]) -> bool:
        """Check if assignment is consistent with constraints"""
        test_assignment = assignment.copy()
        test_assignment[var.name] = value
        current_index = self.var_to_index[var.name]
        
        for constraint in csp.constraints:
            vars_in_constraint = [v.name for v in constraint.variables]
            
            # Skip if constraint doesn't involve current variable
            if var.name not in vars_in_constraint:
                continue
            
            # Skip if not all variables in constraint are assigned
            if not all(v in test_assignment for v in vars_in_constraint):
                continue
            
            self.metrics.constraint_checks += 1
            
            if not constraint.is_satisfied(test_assignment):
                # Update max_check with highest conflicting variable
                conflict_vars = [v for v in vars_in_constraint if v != var.name]
                for conflict_var in conflict_vars:
                    conflict_level = self.var_to_index[conflict_var]
                    current_max = self.max_check[var.name]
                    if current_max == -1 or conflict_level > current_max:
                        self.max_check[var.name] = conflict_level
                return False
        
        return True
    

class ForwardCheckingSolver(BaseSolver):
    def __init__(self, timeout: float = 1800):
        super().__init__("Forward Checking", timeout)
    
    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        """Forward checking main procedure"""
        self.metrics = CSPMetrics()
        self.metrics.start_timer()
        
        # Initialize domain store for tracking removed values
        self.domain_store = {
            var.name: {
                value: None for value in var.domain
            } for var in csp.variables.values()
        }
        
        # Create initial domains backup
        self.saved_domains = {
            var.name: var.domain.copy() 
            for var in csp.variables.values()
        }
        
        result = self._forward_checking(csp, {}, 0)
        
        self.metrics.stop_timer()
        return result
    
    def _forward_checking(self, csp: CSP, assignment: Dict[str, Any], depth: int) -> Optional[Dict[str, Any]]:
        if self.check_timeout():
            return None

        self.metrics.nodes_explored += 1
        
        if len(assignment) == len(csp.variables):
            return assignment
        
        # Select variable using MRV heuristic
        var = self._select_unassigned_variable(csp, assignment)
        if not var:
            return None
        
        # Try each value in the current domain
        for value in list(var.domain):
            self.metrics.constraint_checks += 1
            
            # Save current domains state
            saved_domains = {
                name: var.domain.copy() 
                for name, var in csp.variables.items()
            }
            
            # Try assignment
            assignment[var.name] = value
            
            # Check if assignment is consistent
            if self._is_consistent(csp, var.name, value, assignment):
                # Forward check
                if self._forward_check(csp, var.name, assignment):
                    result = self._forward_checking(csp, assignment, depth + 1)
                    if result is not None:
                        return result
            
            # Remove assignment and restore domains
            assignment.pop(var.name)
            self._restore_domains(csp, saved_domains)
            self.metrics.backtracks += 1
        
        return None
    
    def _forward_check(self, csp: CSP, current_var: str, assignment: Dict[str, Any]) -> bool:
        """
        Check future variables and prune inconsistent values.
        Returns True if forward checking succeeds, False if any domain becomes empty.
        """
        if self.check_timeout():
            return False
        
        self.metrics.revisions += 1
        
        # Get all unassigned variables
        future_vars = [var for var in csp.variables.values() 
                    if var.name not in assignment]
        
        for future_var in future_vars:
            # Save domain before pruning
            original_domain = future_var.domain.copy()
            pruned = False
            
            # Check each value in the future variable's domain
            for value in list(future_var.domain):
                test_assignment = assignment.copy()
                test_assignment[future_var.name] = value
                
                # Check if this value is consistent with current assignment
                if not self._check_constraints(csp, future_var.name, value, test_assignment):
                    future_var.domain.remove(value)
                    pruned = True
            
            # If domain became empty after pruning, restore domain and return False
            if pruned and not future_var.domain:
                future_var.domain = original_domain
                return False
                
        return True

    
    def _check_constraints(self, csp: CSP, var_name: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Check if value is consistent with all relevant constraints"""
        test_assignment = assignment.copy()
        test_assignment[var_name] = value
        
        for constraint in csp.constraints:
            vars_in_constraint = [var.name for var in constraint.get_variables()]
            if var_name in vars_in_constraint:
                # Only check constraints where all variables are assigned
                if all(v in test_assignment for v in vars_in_constraint):
                    if not constraint.is_satisfied(test_assignment):
                        return False
        return True

    
    def _is_consistent(self, csp: CSP, var_name: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Check if current assignment is consistent"""
        # Check all constraints involving the current variable
        test_assignment = assignment.copy()
        
        for constraint in csp.constraints:
            vars_in_constraint = [var.name for var in constraint.get_variables()]
            if var_name in vars_in_constraint:
                if all(v in test_assignment for v in vars_in_constraint):
                    if not constraint.is_satisfied(test_assignment):
                        return False
        
        return True
    
    def _select_unassigned_variable(self, csp: CSP, assignment: Dict[str, Any]) -> Optional[Variable]:
        """Select unassigned variable using minimum remaining values (MRV)"""
        unassigned = [
            var for var in csp.variables.values() 
            if var.name not in assignment
        ]
        if not unassigned:
            return None
            
        return min(unassigned, key=lambda var: (
            len(var.domain),  # First criterion: domain size
            -sum(1 for c in csp.constraints  # Second criterion: degree (number of constraints)
                 if var.name in [v.name for v in c.get_variables()])
        ))
    
    def _restore_domains(self, csp: CSP, domains: Dict[str, Set[Any]]):
        """Restore domains to saved state"""
        for var_name, domain in domains.items():
            csp.variables[var_name].domain = domain.copy()