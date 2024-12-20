from enum import Enum
from typing import Dict, Set, List, Tuple, Any, Optional
from base import Variable, CSP, Constraint
from solvers import BacktrackingSolver, AC3Solver, ForwardCheckingSolver, BackjumpingSolver, ACLookAheadSolver, PC2Solver
from analysis import CSPAnalyzer
from dataclasses import dataclass
from collections import defaultdict
import copy

class ShiftType(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"
    OFF = "off"

@dataclass
class Shift:
    type: ShiftType
    start_time: int
    duration: int

SHIFTS = {
    ShiftType.MORNING: Shift(ShiftType.MORNING, 8, 8),
    ShiftType.AFTERNOON: Shift(ShiftType.AFTERNOON, 16, 8),
    ShiftType.NIGHT: Shift(ShiftType.NIGHT, 0, 8),
    ShiftType.OFF: Shift(ShiftType.OFF, 0, 0)
}

# Add value ordering heuristic
def shifts_ordering(nurse_id: str, day: int, nurse_conditions: Dict[str, Set[str]]) -> List[ShiftType]:
    values = []
    
    # Check if nurse has exemptions
    has_exemption = False
    if nurse_id in nurse_conditions:
        exempt_conditions = {
            "pregnant", "nursing", "medical_condition",
            "reduced_schedule", "grade3_disability"
        }
        has_exemption = any(c in exempt_conditions for c in nurse_conditions[nurse_id])

    # Morning and afternoon shifts are generally preferred
    values.extend([ShiftType.MORNING, ShiftType.AFTERNOON])
    
    # night shift only if not exempt
    if not has_exemption:
        values.append(ShiftType.NIGHT)
    
    # OFF shift as last resort (if no other suitable assignment)
    values.append(ShiftType.OFF)
    
    return values

class NurseRosteringCSP:
    def __init__(self, num_nurses: int, days: int = 7, nurse_conditions: Optional[Dict[str, Set[str]]] = None):
        self.num_nurses = num_nurses
        self.days = days
        self.nurse_conditions = nurse_conditions or {}
        self.csp = CSP()
        self.metrics = {}
        self.initialize_csp()

    def initialize_csp(self):
        # Create variables day-by-day instead of nurse-by-nurse
        for day in range(self.days):
            for nurse in range(self.num_nurses):
                var_name = f"day_{day}_nurse_{nurse}"
                domain = self._get_initial_domain(nurse)
                var = Variable(var_name, domain)
                self.csp.add_variable(var)

        self._add_constraints()
        self.analyzer = CSPAnalyzer(self.csp)

    def _get_initial_domain(self, nurse: int) -> Set[ShiftType]:
        """Get initial domain with consideration for nurse conditions"""
        domain = {ShiftType.MORNING, ShiftType.AFTERNOON, ShiftType.OFF}
        
        # Add night shift only if nurse is not exempt
        if str(nurse) not in self.nurse_conditions or not any(
            c in {"pregnant", "nursing", "medical_condition", "reduced_schedule", "grade3_disability"}
            for c in self.nurse_conditions[str(nurse)]
        ):
            domain.add(ShiftType.NIGHT)
            
        return domain

    def _add_constraints(self):
        # Add simplified but effective constraints
        self._add_shift_coverage_constraint()
        self._add_nurse_workload_constraint()
        self._add_consecutive_shifts_constraint()

    def _add_shift_coverage_constraint(self):
        """Ensure minimum coverage for each shift type"""
        for day in range(self.days):
            day_vars = [var for var in self.csp.variables.values() 
                       if var.name.startswith(f"day_{day}")]
            self.csp.add_constraint(ShiftCoverageConstraint(day_vars, {
                ShiftType.MORNING: 2,
                ShiftType.AFTERNOON: 2,
                ShiftType.NIGHT: 1
            }))

    def _add_nurse_workload_constraint(self):
        """Ensure each nurse works between 36-48 hours per week"""
        for nurse in range(self.num_nurses):
            nurse_vars = [var for var in self.csp.variables.values() 
                         if f"nurse_{nurse}" in var.name]
            self.csp.add_constraint(WeeklyHoursConstraint(nurse_vars))

    def _add_consecutive_shifts_constraint(self):
        """Ensure proper rest periods between shifts"""
        for nurse in range(self.num_nurses):
            for day in range(self.days - 1):
                var1 = self.csp.variables[f"day_{day}_nurse_{nurse}"]
                var2 = self.csp.variables[f"day_{day+1}_nurse_{nurse}"]
                self.csp.add_constraint(RestPeriodConstraint([var1, var2]))

    def print_metrics_summary(self):
        """Print summary of solver metrics"""
        print("\nSolver Performance Summary:")
        print("-" * 100)
        
        headers = ['Solver', 'Time (s)', 'Checks', 'Revisions', 'Nodes', 'Backtracks', 'Found Solution']
        print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<12} {headers[3]:<12} "
            f"{headers[4]:<12} {headers[5]:<12} {headers[6]:<15}")
        print("-" * 100)
        
        for solver_name, metrics in self.metrics.items():
            solution_found = metrics.nodes_explored > 0 and metrics.execution_time < 1800  # Less than timeout
            print(f"{solver_name:<15} {metrics.execution_time:<10.3f} "
                f"{metrics.constraint_checks:<12} {metrics.revisions:<12} "
                f"{metrics.nodes_explored:<12} {metrics.backtracks:<12} "
                f"{'Yes' if solution_found else 'No':<15}")


    def solve(self) -> Optional[Dict[str, ShiftType]]:
        """Solve using multiple solvers and collect metrics"""
        solvers = [
            ForwardCheckingSolver(),
            ACLookAheadSolver(),
            AC3Solver(),
            BackjumpingSolver(),
            BacktrackingSolver(),
            PC2Solver()
        ]

        best_solution = None
        print("\nSolver Performance:")
        print("-" * 50)

        for solver in solvers:
            print(f"\nTrying {solver.name}...")
            csp_copy = copy.deepcopy(self.csp)
            
            try:
                solution = solver.solve(csp_copy)
                metrics = solver.get_metrics()
                self.metrics[solver.name] = metrics
                
                self._print_solver_metrics(solver.name, metrics)
                
                if solution:
                    print("Solution found!")
                    converted_solution = self._convert_solution(solution)
                    if not best_solution:
                        best_solution = converted_solution
                else:
                    print("No solution found.")
                    
            except Exception as e:
                print(f"Error with {solver.name}: {str(e)}")
                continue

        print("\nFinal Results:")
        print("-" * 50)
        self.print_metrics_summary()
        
        if best_solution:
            print("\nBest Solution Found:")
            self._print_solution(best_solution)
            
        return best_solution


    def _convert_solution(self, solution: Dict[str, ShiftType]) -> Dict[str, ShiftType]:
        """Convert day-based solution to nurse-based solution"""
        converted = {}
        for var_name, shift in solution.items():
            _, day, _, nurse = var_name.split('_')
            converted[f"nurse_{nurse}_{day}"] = shift
        return converted

    def _print_solution(self, solution: Dict[str, ShiftType]):
        print("\nRoster Solution:")
        print("-" * 60)
        
        # Calculate hours
        nurse_hours = defaultdict(int)
        nurse_schedules = defaultdict(dict)
        
        for var_name, shift in solution.items():
            nurse_id, day = var_name.split("_")[1:]
            nurse_schedules[nurse_id][int(day)] = shift
            if shift != ShiftType.OFF:
                nurse_hours[nurse_id] += SHIFTS[shift].duration

        # Print schedule
        print(f"{'Nurse':^8}|", end="")
        for day in range(self.days):
            print(f" Day {day:^3} |", end="")
        print(" Hours")
        print("-" * 60)

        for nurse_id, schedule in sorted(nurse_schedules.items()):
            print(f"{nurse_id:^8}|", end="")
            for day in range(self.days):
                shift = schedule.get(day, ShiftType.OFF)
                print(f" {shift.value[:3]:^5} |", end="")
            print(f" {nurse_hours[nurse_id]:>4}")

    def _print_solver_metrics(self, solver_name: str, metrics):
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Constraint checks: {metrics.constraint_checks}")
        print(f"Revisions: {metrics.revisions}")
        print(f"Nodes explored: {metrics.nodes_explored}")
        print(f"Backtracks: {metrics.backtracks}")


class ShiftCoverageConstraint(Constraint):
    def __init__(self, variables: List[Variable], requirements: Dict[ShiftType, int]):
        super().__init__(variables)
        self.requirements = requirements

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        day_vars = set(var.name for var in self.variables)
        assigned_vars = set(assignment.keys())
        
        # If not all variables for this day are assigned, check if requirements can still be met
        if not day_vars.issubset(assigned_vars):
            # Count current assignments
            shift_counts = defaultdict(int)
            for var_name in day_vars.intersection(assigned_vars):
                shift = assignment[var_name]
                if shift != ShiftType.OFF:
                    shift_counts[shift] += 1
            
            # Check if remaining unassigned variables could satisfy requirements
            remaining = len(day_vars - assigned_vars)
            for shift_type, required in self.requirements.items():
                if shift_counts[shift_type] + remaining < required:
                    return False
            return True
            
        # All variables assigned - check requirements
        shift_counts = defaultdict(int)
        for var_name in day_vars:
            shift = assignment[var_name]
            if shift != ShiftType.OFF:
                shift_counts[shift] += 1
                
        return all(shift_counts[shift_type] >= required 
                  for shift_type, required in self.requirements.items())
    
    def get_variables(self) -> List[Variable]:
        return self.variables

class WeeklyHoursConstraint(Constraint):
    def __init__(self, variables: List[Variable], min_hours: int = 40, max_hours: int = 48):
        super().__init__(variables)
        self.min_hours = min_hours
        self.max_hours = max_hours

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        nurse_vars = set(var.name for var in self.variables)
        assigned_vars = nurse_vars.intersection(assignment.keys())
        
        if not assigned_vars:  # No assignments yet
            return True
            
        # Calculate current hours
        current_hours = sum(SHIFTS[assignment[var_name]].duration 
                          for var_name in assigned_vars
                          if assignment[var_name] != ShiftType.OFF)
        
        # If all shifts assigned, check exact requirements
        if nurse_vars.issubset(assignment.keys()):
            return self.min_hours <= current_hours <= self.max_hours
            
        # For partial assignments, check if requirements can still be met
        remaining_shifts = len(nurse_vars - assigned_vars)
        min_possible = current_hours  # No additional hours
        max_possible = current_hours + (remaining_shifts * 8)  # Maximum possible additional hours
        
        return min_possible <= self.max_hours and max_possible >= self.min_hours

    
    def get_variables(self) -> List[Variable]:
        return self.variables


class RestPeriodConstraint(Constraint):
    def __init__(self, variables: List[Variable], min_rest: int = 12):
        super().__init__(variables)
        self.min_rest = min_rest

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        # Get assigned variables that are part of this constraint
        assigned_vars = [var for var in self.variables 
                        if var.name in assignment]
        
        if len(assigned_vars) < 2:
            return True
            
        # Extract day numbers to check adjacent days
        days = [int(var.name.split('_')[1]) for var in assigned_vars]
        nurse = assigned_vars[0].name.split('_')[3]  # Get nurse ID
        
        # Check each pair of consecutive days
        for i in range(len(days)-1):
            if abs(days[i] - days[i+1]) == 1:  # Adjacent days
                shift1 = assignment[f"day_{days[i]}_nurse_{nurse}"]
                shift2 = assignment[f"day_{days[i+1]}_nurse_{nurse}"]
                
                if shift1 == ShiftType.OFF or shift2 == ShiftType.OFF:
                    continue
                    
                end_time = (SHIFTS[shift1].start_time + SHIFTS[shift1].duration) % 24
                start_time = SHIFTS[shift2].start_time
                
                # Calculate rest period with day boundary handling
                if start_time >= end_time:
                    rest_period = start_time - end_time
                else:
                    rest_period = 24 - (end_time - start_time)
                    
                if rest_period < self.min_rest:
                    return False
                    
        return True

    
    def get_variables(self) -> List[Variable]:
        return self.variables

def solve_nurse_rostering(num_nurses: int = 4, days: int = 7, 
                         nurse_conditions: Optional[Dict[str, Set[str]]] = None) -> bool:
    csp = NurseRosteringCSP(num_nurses, days, nurse_conditions)
    solution = csp.solve()
    return solution is not None

if __name__ == "__main__":
    nurse_conditions = {
        "0": {"pregnant"},
        "2": {"medical_condition"}
    }
    
    solve_nurse_rostering(
        num_nurses=10,
        days=7,
        nurse_conditions=nurse_conditions
    )