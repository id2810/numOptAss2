"""
Unit tests for the interior point method implementation.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constrained_min import interior_pt
from examples import get_qp_problem, get_lp_problem


class TestConstrainedMin(unittest.TestCase):
    """Test cases for constrained minimization problems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tol = 1e-6
        
    def test_qp(self):
        """Test the quadratic programming example."""
        print("\n" + "="*60)
        print("TESTING QUADRATIC PROGRAMMING PROBLEM")
        print("="*60)
        print("Problem: min x^2 + y^2 + (z+1)^2")
        print("Subject to: x + y + z = 1, x >= 0, y >= 0, z >= 0")
        print("Initial point: (0.1, 0.2, 0.7)")
        
        # Get problem configuration
        func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = get_qp_problem()
        
        # Solve the problem
        x_opt, path, objective_values, success = interior_pt(
            func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0
        )
        
        # Check convergence
        self.assertTrue(success, "Interior point method should converge")
        
        # Verify the solution
        print(f"\nFinal solution: {x_opt}")
        final_obj, _ = func(x_opt, compute_grad=False)
        print(f"Final objective value: {final_obj:.8f}")
        
        # Check equality constraint satisfaction
        eq_violation = np.linalg.norm(eq_constraints_mat @ x_opt - eq_constraints_rhs)
        print(f"Equality constraint violation: {eq_violation:.2e}")
        self.assertLess(eq_violation, self.tol, "Equality constraint should be satisfied")
        
        # Check inequality constraints (non-negativity)
        for i, x_val in enumerate(x_opt):
            print(f"x[{i}] = {x_val:.8f} (should be >= 0)")
            self.assertGreaterEqual(x_val, -self.tol, f"x[{i}] should be non-negative")
        
        # Theoretical solution: closest point to (0,0,-1) on simplex
        # Solution should be approximately (0, 0, 1)
        expected_solution = np.array([0, 0, 1])
        error = np.linalg.norm(x_opt - expected_solution)
        print(f"Distance from expected solution (0, 0, 1): {error:.6f}")
        
        # Create plots
        self._plot_qp_results(path, objective_values, x_opt)
        
        return x_opt, objective_values, success
    
    def test_lp(self):
        """Test the linear programming example."""
        print("\n" + "="*60)
        print("TESTING LINEAR PROGRAMMING PROBLEM")
        print("="*60)
        print("Problem: max x + y")
        print("Subject to: y >= -x + 1, y <= 1, x <= 2, y >= 0")
        print("Initial point: (0.5, 0.75)")
        
        # Get problem configuration
        func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = get_lp_problem()
        
        # Solve the problem
        x_opt, path, objective_values, success = interior_pt(
            func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0
        )
        
        # Check convergence
        self.assertTrue(success, "Interior point method should converge")
        
        # Verify the solution
        print(f"\nFinal solution: {x_opt}")
        final_obj, _ = func(x_opt, compute_grad=False)
        original_obj = -final_obj  # Convert back to maximization
        print(f"Final objective value (maximization): {original_obj:.8f}")
        
        # Check inequality constraints
        constraint_values = []
        constraint_names = [
            "-(x + y - 1) <= 0 (y >= -x + 1)",
            "y - 1 <= 0 (y <= 1)",
            "x - 2 <= 0 (x <= 2)",
            "-y <= 0 (y >= 0)"
        ]
        
        for i, constraint in enumerate(ineq_constraints):
            c_val, _ = constraint(x_opt, compute_grad=False)
            constraint_values.append(c_val)
            print(f"Constraint {i+1}: {constraint_names[i]}, value = {c_val:.8f}")
            self.assertLessEqual(c_val, self.tol, f"Inequality constraint {i+1} should be satisfied")
        
        # Theoretical solution: upper right vertex should be (2, 1)
        expected_solution = np.array([2, 1])
        error = np.linalg.norm(x_opt - expected_solution)
        print(f"Distance from expected solution (2, 1): {error:.6f}")
        
        # Create plots
        self._plot_lp_results(path, objective_values, x_opt)
        
        return x_opt, objective_values, success
    
    def _plot_qp_results(self, path, objective_values, x_opt):
        """Create plots for QP results."""
        path = np.array(path)
        
        # Plot 1: 3D feasible region and central path
        fig = plt.figure(figsize=(15, 6))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot the feasible region (simplex)
        simplex_vertices = np.array([
            [1, 0, 0],  # (1, 0, 0)
            [0, 1, 0],  # (0, 1, 0)
            [0, 0, 1]   # (0, 0, 1)
        ])
        
        # Draw the simplex faces
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        simplex_faces = [simplex_vertices]
        ax1.add_collection3d(Poly3DCollection(simplex_faces, alpha=0.3, facecolor='lightblue', edgecolor='blue'))
        
        # Plot the central path
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-', markersize=4, linewidth=2, label='Central Path')
        
        # Plot initial and final points
        ax1.scatter(*path[0], color='green', s=100, label='Initial Point')
        ax1.scatter(*x_opt, color='red', s=100, label='Final Solution')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title('QP: Feasible Region and Central Path')
        ax1.legend()
        
        # Plot 2: Objective value vs iteration
        ax2 = fig.add_subplot(122)
        ax2.plot(range(len(objective_values)), objective_values, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('QP: Objective Value vs Iteration')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('qp_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_lp_results(self, path, objective_values, x_opt):
        """Create plots for LP results."""
        path = np.array(path)
        
        # Plot 1: 2D feasible region and central path
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Define the feasible region
        x_vals = np.linspace(-0.5, 2.5, 100)
        
        # Plot constraints
        # y >= -x + 1
        y1 = -x_vals + 1
        ax1.plot(x_vals, y1, 'b-', label='y = -x + 1', linewidth=2)
        ax1.fill_between(x_vals, y1, 3, alpha=0.1, color='blue')
        
        # y <= 1
        ax1.axhline(y=1, color='red', linestyle='-', linewidth=2, label='y = 1')
        ax1.fill_between(x_vals, -1, 1, alpha=0.1, color='red')
        
        # x <= 2
        ax1.axvline(x=2, color='green', linestyle='-', linewidth=2, label='x = 2')
        ax1.fill_betweenx(np.linspace(-1, 3, 100), -1, 2, alpha=0.1, color='green')
        
        # y >= 0
        ax1.axhline(y=0, color='orange', linestyle='-', linewidth=2, label='y = 0')
        ax1.fill_between(x_vals, 0, 3, alpha=0.1, color='orange')
        
        # Feasible region vertices
        vertices = np.array([
            [0, 0],    # Origin
            [0, 1],    # (0, 1)
            [1, 1],    # (1, 1) 
            [2, 1],    # (2, 1) - optimal point
            [2, 0]     # (2, 0)
        ])
        
        # Plot feasible region boundary
        vertices_closed = np.vstack([vertices, vertices[0]])
        ax1.fill(vertices_closed[:, 0], vertices_closed[:, 1], alpha=0.3, color='lightgray', label='Feasible Region')
        ax1.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'k-', linewidth=2)
        
        # Plot the central path
        ax1.plot(path[:, 0], path[:, 1], 'ro-', markersize=6, linewidth=2, label='Central Path')
        
        # Plot initial and final points
        ax1.scatter(*path[0], color='green', s=100, label='Initial Point', zorder=10)
        ax1.scatter(*x_opt, color='red', s=100, label='Final Solution', zorder=10)
        
        # Plot objective function contours
        x_grid = np.linspace(-0.5, 2.5, 50)
        y_grid = np.linspace(-0.5, 1.5, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = X + Y  # Objective function (we're maximizing x + y)
        contours = ax1.contour(X, Y, Z, levels=10, colors='purple', alpha=0.5)
        ax1.clabel(contours, inline=True, fontsize=8)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('LP: Feasible Region and Central Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, 2.5)
        ax1.set_ylim(-0.5, 1.5)
        
        # Plot 2: Objective value vs iteration (convert back to maximization)
        max_objectives = [-obj for obj in objective_values]
        ax2.plot(range(len(max_objectives)), max_objectives, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value (Maximization)')
        ax2.set_title('LP: Objective Value vs Iteration')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('lp_results.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
