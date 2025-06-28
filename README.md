# Numerical Optimization - Interior Point Method

This project implements an Interior Point Method solver for constrained optimization problems, as part of Programming Assignment 02 for Numerical Optimization with Python (2025B).

## Project Structure

```
numOptFinalEx/
├── src/
│   └── constrained_min.py          # Main interior point method implementation
├── tests/
│   └── test_constrained_min.py     # Unit tests and visualization
├── examples.py                     # Problem definitions (QP and LP examples)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Implementation

The interior point method uses the log-barrier approach with:
- Initial barrier parameter t = 1
- Barrier parameter increase factor μ = 10
- Convergence tolerance = 1e-8

### Features
- General implementation supporting both equality and inequality constraints
- Log-barrier method for inequality constraints
- SLSQP and L-BFGS-B solvers for inner optimization
- Comprehensive visualization of results
- Unit testing framework

## Problems Solved

### 1. Quadratic Programming (QP) Problem
**Objective:** Minimize x² + y² + (z+1)²  
**Constraints:** 
- x + y + z = 1 (equality)
- x ≥ 0, y ≥ 0, z ≥ 0 (inequality)

**Initial Point:** (0.1, 0.2, 0.7)  
**Interpretation:** Find the closest probability vector to point (0,0,-1)

### 2. Linear Programming (LP) Problem
**Objective:** Maximize x + y  
**Constraints:**
- y ≥ -x + 1
- y ≤ 1
- x ≤ 2
- y ≥ 0

**Initial Point:** (0.5, 0.75)  
**Interpretation:** Find the upper right vertex of a planar polygon

## Usage

### Quick Start
```bash
# Run the assignment demonstration (this runs both test_qp() and test_lp())
python tests/test_constrained_min.py

# Run tests using unittest
python -m unittest tests.test_constrained_min -v
```

### Using the Interior Point Solver
```python
from src.constrained_min import interior_pt
from examples import get_qp_problem

# Get problem configuration
func, ineq_constraints, eq_mat, eq_rhs, x0 = get_qp_problem()

# Solve
x_opt, path, obj_values, success = interior_pt(
    func, ineq_constraints, eq_mat, eq_rhs, x0
)
```

## Results

The implementation generates:
1. **Feasible region plots** with the central path taken by the algorithm
2. **Objective value plots** vs outer iteration number
3. **Final solution** with constraint values

### Expected Solutions
- **QP Problem:** Approximately (0, 0, 1) - the closest point to (0,0,-1) on the probability simplex
- **LP Problem:** Approximately (2, 1) - the upper right vertex maximizing x + y

## Dependencies

- `numpy` >= 1.21.0
- `scipy` >= 1.7.0  
- `matplotlib` >= 3.4.0

Install with:
```bash
pip install -r requirements.txt
```

## Technical Details

### Interior Point Algorithm
1. **Outer Loop:** Increase barrier parameter t by factor μ = 10
2. **Inner Loop:** Solve barrier problem using constrained optimization
3. **Convergence:** Stop when duality gap m/t < tolerance

### Constraint Handling
- **Inequality constraints:** g(x) ≤ 0 handled via log-barrier: -log(-g(x))
- **Equality constraints:** Ax = b handled directly in inner optimization
- **Gradient computation:** Analytical gradients for efficiency
- **Hessian approximation:** Finite differences when not available

## Visualization

The code generates detailed plots showing:
- 3D feasible region and central path (QP problem)
- 2D feasible region with constraint boundaries (LP problem)
- Objective function contours
- Convergence behavior

## Testing

Comprehensive unit tests verify:
- Convergence to correct solutions
- Constraint satisfaction
- Algorithm robustness
- Visualization functionality

## Author

Programming Assignment 02 - Numerical Optimization with Python (2025B)
