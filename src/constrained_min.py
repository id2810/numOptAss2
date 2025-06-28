"""
Interior Point Method for Constrained Optimization
Implementation of log-barrier method for solving constrained optimization problems.
"""

import numpy as np
from scipy.optimize import minimize
import warnings


class InteriorPointSolver:
    """Interior Point Method solver using log-barrier approach."""
    
    def __init__(self, t_init=1.0, mu=10.0, tol=1e-8, max_outer_iter=50):
        """
        Initialize the Interior Point solver.
        
        Args:
            t_init (float): Initial barrier parameter
            mu (float): Factor to increase barrier parameter
            tol (float): Convergence tolerance
            max_outer_iter (int): Maximum outer iterations
        """
        self.t_init = t_init
        self.mu = mu
        self.tol = tol
        self.max_outer_iter = max_outer_iter
        
    def _log_barrier_objective(self, x, func, ineq_constraints, t):
        """
        Compute the log-barrier objective function.
        
        Args:
            x (array): Current point
            func (callable): Original objective function
            ineq_constraints (list): List of inequality constraint functions
            t (float): Barrier parameter
            
        Returns:
            float: Log-barrier objective value
        """
        # Original objective
        f_val, _ = func(x, compute_grad=False)
        obj = t * f_val
        
        # Log-barrier terms for inequality constraints
        for constraint in ineq_constraints:
            c_val, _ = constraint(x, compute_grad=False)
            if c_val >= 0:  # Constraint violated
                return np.inf
            obj -= np.log(-c_val)
            
        return obj
    
    def _log_barrier_gradient(self, x, func, ineq_constraints, t):
        """
        Compute the gradient of the log-barrier objective.
        
        Args:
            x (array): Current point
            func (callable): Original objective function
            ineq_constraints (list): List of inequality constraint functions
            t (float): Barrier parameter
            
        Returns:
            array: Gradient of log-barrier objective
        """
        # Gradient of original objective
        _, f_grad = func(x, compute_grad=True)
        grad = t * f_grad
        
        # Gradient of log-barrier terms
        for constraint in ineq_constraints:
            c_val, c_grad = constraint(x, compute_grad=True)
            if c_val >= 0:  # Constraint violated
                return np.full_like(x, np.inf)
            grad += c_grad / (-c_val)
            
        return grad
    
    def _log_barrier_hessian(self, x, func, ineq_constraints, t):
        """
        Compute the Hessian of the log-barrier objective.
        
        Args:
            x (array): Current point
            func (callable): Original objective function
            ineq_constraints (list): List of inequality constraint functions
            t (float): Barrier parameter
            
        Returns:
            array: Hessian of log-barrier objective
        """
        n = len(x)
        
        # Hessian of original objective
        if hasattr(func, '__call__') and len(func(x, compute_hessian=True)) == 3:
            _, _, f_hess = func(x, compute_hessian=True)
        else:
            # Approximate Hessian with finite differences if not available
            f_hess = self._finite_diff_hessian(func, x)
        
        hess = t * f_hess
        
        # Hessian of log-barrier terms
        for constraint in ineq_constraints:
            c_val, c_grad = constraint(x, compute_grad=True)
            if c_val >= 0:  # Constraint violated
                return np.full((n, n), np.inf)
            
            # Second-order term: -1/c^2 * grad(c) * grad(c)^T
            hess += np.outer(c_grad, c_grad) / (c_val * c_val)
            
            # First-order term (if constraint has Hessian)
            if hasattr(constraint, '__call__') and len(constraint(x, compute_hessian=True)) == 3:
                _, _, c_hess = constraint(x, compute_hessian=True)
                hess -= c_hess / c_val
                
        return hess
    
    def _finite_diff_hessian(self, func, x, eps=1e-8):
        """Approximate Hessian using finite differences."""
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Second partial derivative
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                
                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                
                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                
                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                f_pp, _ = func(x_pp, compute_grad=False)
                f_pm, _ = func(x_pm, compute_grad=False)
                f_mp, _ = func(x_mp, compute_grad=False)
                f_mm, _ = func(x_mm, compute_grad=False)
                
                hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
                hess[j, i] = hess[i, j]
                
        return hess


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    """
    Interior Point Method for constrained optimization.
    
    Args:
        func (callable): Objective function with interface f(x, compute_grad=False, compute_hessian=False)
        ineq_constraints (list): List of inequality constraint functions g_i(x) <= 0
        eq_constraints_mat (array): Matrix A for equality constraints Ax = b
        eq_constraints_rhs (array): Vector b for equality constraints Ax = b
        x0 (array): Initial point (must be strictly feasible)
        
    Returns:
        tuple: (x_opt, path, objective_values, success)
            - x_opt: Optimal point found
            - path: List of points in the central path
            - objective_values: List of objective values at each outer iteration
            - success: Boolean indicating convergence
    """
    solver = InteriorPointSolver()
    
    # Convert inputs to numpy arrays
    x0 = np.array(x0, dtype=float)
    if eq_constraints_mat is not None:
        eq_constraints_mat = np.array(eq_constraints_mat, dtype=float)
    if eq_constraints_rhs is not None:
        eq_constraints_rhs = np.array(eq_constraints_rhs, dtype=float)
    
    # Initialize
    x = x0.copy()
    t = solver.t_init
    path = [x.copy()]
    objective_values = []
    
    # Check initial feasibility
    for constraint in ineq_constraints:
        c_val, _ = constraint(x, compute_grad=False)
        if c_val >= 0:
            raise ValueError(f"Initial point is not strictly feasible: constraint value = {c_val}")
    
    if eq_constraints_mat is not None and eq_constraints_rhs is not None:
        eq_violation = np.linalg.norm(eq_constraints_mat @ x - eq_constraints_rhs)
        if eq_violation > 1e-10:
            raise ValueError(f"Initial point violates equality constraints: violation = {eq_violation}")
    
    print(f"Starting Interior Point Method")
    print(f"Initial point: {x}")
    
    for outer_iter in range(solver.max_outer_iter):
        print(f"\nOuter iteration {outer_iter + 1}, t = {t:.2e}")
        
        # Define the barrier objective and its derivatives
        def barrier_obj_and_grad(x_var):
            obj = solver._log_barrier_objective(x_var, func, ineq_constraints, t)
            grad = solver._log_barrier_gradient(x_var, func, ineq_constraints, t)
            return obj, grad
        
        # Define constraints for the inner optimization
        constraints = []
        
        # Equality constraints
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            def eq_constraint(x_var):
                return eq_constraints_mat @ x_var - eq_constraints_rhs
            
            def eq_constraint_jac(x_var):
                return eq_constraints_mat
            
            constraints.append({
                'type': 'eq',
                'fun': eq_constraint,
                'jac': eq_constraint_jac
            })
        
        # Solve the barrier problem
        try:
            def objective(x_var):
                return barrier_obj_and_grad(x_var)[0]
            
            def gradient(x_var):
                return barrier_obj_and_grad(x_var)[1]
            
            result = minimize(
                objective,
                x,
                method='SLSQP',
                jac=gradient,
                constraints=constraints,
                options={'ftol': solver.tol, 'disp': False}
            )
            
            if not result.success:
                print(f"Inner optimization failed: {result.message}")
                # Try with a different method
                result = minimize(
                    objective,
                    x,
                    method='L-BFGS-B',
                    jac=gradient,
                    options={'ftol': solver.tol, 'disp': False}
                )
            
            x = result.x
            
        except Exception as e:
            print(f"Error in inner optimization: {e}")
            break
        
        # Store current point and objective value
        path.append(x.copy())
        f_val, _ = func(x, compute_grad=False)
        objective_values.append(f_val)
        
        print(f"Current point: {x}")
        print(f"Objective value: {f_val:.6e}")
        
        # Check convergence (duality gap approximation)
        m = len(ineq_constraints)  # Number of inequality constraints
        if m / t < solver.tol:
            print(f"\nConverged! Duality gap estimate: {m/t:.2e}")
            return x, path, objective_values, True
        
        # Update barrier parameter
        t *= solver.mu
    
    print(f"\nMaximum iterations reached")
    return x, path, objective_values, False


# Helper functions for creating constraint functions
def create_inequality_constraint(constraint_func, constraint_grad_func=None):
    """
    Create an inequality constraint function compatible with interior_pt.
    
    Args:
        constraint_func (callable): Function that computes g(x) for constraint g(x) <= 0
        constraint_grad_func (callable): Function that computes gradient of g(x)
        
    Returns:
        callable: Constraint function with the required interface
    """
    def constraint(x, compute_grad=False, compute_hessian=False):
        if compute_hessian:
            # For simplicity, we don't implement constraint Hessians
            val = constraint_func(x)
            if constraint_grad_func is not None:
                grad = constraint_grad_func(x)
            else:
                # Finite difference gradient
                grad = finite_diff_gradient(constraint_func, x)
            return val, grad, np.zeros((len(x), len(x)))
        elif compute_grad:
            val = constraint_func(x)
            if constraint_grad_func is not None:
                grad = constraint_grad_func(x)
            else:
                # Finite difference gradient
                grad = finite_diff_gradient(constraint_func, x)
            return val, grad
        else:
            return constraint_func(x), None
    
    return constraint


def finite_diff_gradient(func, x, eps=1e-8):
    """Compute gradient using finite differences."""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    
    return grad
