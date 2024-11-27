from typing import Dict, Set, List, Any, Optional
from abc import ABC, abstractmethod
import copy

class Variable:
    def __init__(self, name: str, domain: Set[Any]):
        self.name = name
        self.domain = domain
        self.initial_domain = copy.deepcopy(domain)
    
    def __str__(self):
        return f"{self.name}: {self.domain}"
    
    def reset_domain(self):
        self.domain = copy.deepcopy(self.initial_domain)

class Constraint(ABC):
    def __init__(self, variables: List[Variable]):
        self.variables = variables
    
    @abstractmethod
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Checks if the constraint is satisfied with the given assignment"""
        pass
    
    @abstractmethod
    def get_variables(self) -> List[Variable]:
        """Returns list of variables involved in this constraint"""
        return self.variables

class CSP:
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.assignment: Dict[str, Any] = {}
    
    def add_variable(self, variable: Variable):
        """Adds a variable to the CSP"""
        self.variables[variable.name] = variable
    
    def add_constraint(self, constraint: Constraint):
        """Adds a constraint to the CSP"""
        self.constraints.append(constraint)
    
    def is_consistent(self, var_name: str, value: Any) -> bool:
        """Checks if we maintain consistency if we assign a value to a variable"""
        assignment = self.assignment.copy()
        assignment[var_name] = value
        
        # Check each constraint that involves this variable
        return all(
            constraint.is_satisfied(assignment)
            for constraint in self.constraints
            if var_name in [var.name for var in constraint.get_variables()]
            and all(v.name in assignment for v in constraint.get_variables())
        )
    
    def select_unassigned_variable(self) -> Optional[Variable]:
        """Selects an unassigned variable (grabs the first unassigned variable, by default)"""
        for var_name, variable in self.variables.items():
            if var_name not in self.assignment:
                return variable
        return None
    
    def order_domain_values(self, variable: Variable) -> List[Any]:
        """Order domain values for a variable (We simply grab the list as it is)"""
        return list(variable.domain)
    
    def backtracking_search(self) -> Optional[Dict[str, Any]]:
        """Backtracking search to find a solution"""
        if len(self.assignment) == len(self.variables):
            return self.assignment
        
        var = self.select_unassigned_variable()
        if not var:
            return None
        
        for value in self.order_domain_values(var):
            if self.is_consistent(var.name, value):
                self.assignment[var.name] = value
                result = self.backtracking_search()
                if result is not None:
                    return result
                del self.assignment[var.name]
        
        return None
    
    def solve(self) -> Optional[Dict[str, Any]]:
        self.assignment = {}
        return self.backtracking_search()
    
    def reset(self):
        """Resets the CSP to its initial state"""
        self.assignment = {}
        for variable in self.variables.values():
            variable.reset_domain()

