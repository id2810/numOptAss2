"""
Example optimization problems for testing the interior point method.
"""

import numpy as np
from src.constrained_min import create_inequality_constraint


def qp_objective_func(x, compute_grad=False, compute_hessian=False):
    """
    Quadratic Programming objective: min x^2 + y^2 + (z+1)^2
    Subject to: x + y + z = 1, x >= 0, y >= 0, z >= 0
    """
    x_val, y_val, z_val = x[0], x[1], x[2]
    
    # Objective value
    obj = x_val**2 + y_val**2 + (z_val + 1)**2
    
    if compute_hessian:
        # Gradient
        grad = np.array([2*x_val, 2*y_val, 2*(z_val + 1)])
        
        # Hessian
        hess = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        
        return obj, grad, hess
    
    elif compute_grad:
        # Gradient
        grad = np.array([2*x_val, 2*y_val, 2*(z_val + 1)])
        return obj, grad
    
    else:
        return obj, None


def qp_inequality_constraints():
    """
    Create inequality constraints for QP: x >= 0, y >= 0, z >= 0
    Converted to g(x) <= 0 format: -x <= 0, -y <= 0, -z <= 0
    """
    constraints = []
    
    # -x <= 0 (i.e., x >= 0)
    def neg_x_constraint(x):
        return -x[0]
    
    def neg_x_gradient(x):
        return np.array([-1, 0, 0])
    
    constraints.append(create_inequality_constraint(neg_x_constraint, neg_x_gradient))
    
    # -y <= 0 (i.e., y >= 0)
    def neg_y_constraint(x):
        return -x[1]
    
    def neg_y_gradient(x):
        return np.array([0, -1, 0])
    
    constraints.append(create_inequality_constraint(neg_y_constraint, neg_y_gradient))
    
    # -z <= 0 (i.e., z >= 0)
    def neg_z_constraint(x):
        return -x[2]
    
    def neg_z_gradient(x):
        return np.array([0, 0, -1])
    
    constraints.append(create_inequality_constraint(neg_z_constraint, neg_z_gradient))
    
    return constraints


def qp_equality_constraints():
    """
    Create equality constraint matrix and RHS for QP: x + y + z = 1
    """
    # Ax = b where A = [1, 1, 1] and b = [1]
    eq_constraints_mat = np.array([[1, 1, 1]])
    eq_constraints_rhs = np.array([1])
    
    return eq_constraints_mat, eq_constraints_rhs


def lp_objective_func(x, compute_grad=False, compute_hessian=False):
    """
    Linear Programming objective: max[x + y] = min[-x - y]
    """
    x_val, y_val = x[0], x[1]
    
    # Objective value (negative since we're maximizing)
    obj = -x_val - y_val
    
    if compute_hessian:
        # Gradient
        grad = np.array([-1, -1])
        
        # Hessian (linear function has zero Hessian)
        hess = np.zeros((2, 2))
        
        return obj, grad, hess
    
    elif compute_grad:
        # Gradient
        grad = np.array([-1, -1])
        return obj, grad
    
    else:
        return obj, None


def lp_inequality_constraints():
    """
    Create inequality constraints for LP:
    y >= -x + 1  =>  x + y - 1 >= 0  =>  -(x + y - 1) <= 0
    y <= 1       =>  y - 1 <= 0
    x <= 2       =>  x - 2 <= 0
    y >= 0       =>  -y <= 0
    """
    constraints = []
    
    # -(x + y - 1) <= 0  (i.e., y >= -x + 1)
    def constraint1(x):
        return -(x[0] + x[1] - 1)
    
    def constraint1_grad(x):
        return np.array([-1, -1])
    
    constraints.append(create_inequality_constraint(constraint1, constraint1_grad))
    
    # y - 1 <= 0  (i.e., y <= 1)
    def constraint2(x):
        return x[1] - 1
    
    def constraint2_grad(x):
        return np.array([0, 1])
    
    constraints.append(create_inequality_constraint(constraint2, constraint2_grad))
    
    # x - 2 <= 0  (i.e., x <= 2)
    def constraint3(x):
        return x[0] - 2
    
    def constraint3_grad(x):
        return np.array([1, 0])
    
    constraints.append(create_inequality_constraint(constraint3, constraint3_grad))
    
    # -y <= 0  (i.e., y >= 0)
    def constraint4(x):
        return -x[1]
    
    def constraint4_grad(x):
        return np.array([0, -1])
    
    constraints.append(create_inequality_constraint(constraint4, constraint4_grad))
    
    return constraints


def lp_equality_constraints():
    """
    No equality constraints for LP problem.
    """
    return None, None


# Problem configurations
def get_qp_problem():
    """Get the complete QP problem configuration."""
    func = qp_objective_func
    ineq_constraints = qp_inequality_constraints()
    eq_constraints_mat, eq_constraints_rhs = qp_equality_constraints()
    x0 = np.array([0.1, 0.2, 0.7])  # Initial point
    
    return func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0


def get_lp_problem():
    """Get the complete LP problem configuration."""
    func = lp_objective_func
    ineq_constraints = lp_inequality_constraints()
    eq_constraints_mat, eq_constraints_rhs = lp_equality_constraints()
    x0 = np.array([0.5, 0.75])  # Initial point
    
    return func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0
