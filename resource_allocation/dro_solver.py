import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import gamma
import itertools

class DROParameters:
    """Class to handle DRO problem parameters"""
    def __init__(self, x_dim, num_ball, R, params, epsilon=None, ball_weights=None, ball_centres=None):
        """
        Initialize DRO parameters.
        
        Args:
            x_dim (int): Dimension of decision variable x (number of resources)
            R (float): Total resource constraint
            params (dict): Fixed parameters for each resource i
            epsilon (float): Radius of ambiguity set
            ball_weights (np.array): Weights for each Euclidean ball
            ball_centres (np.array): Centres for each Euclidean ball
        """
        self.x_dim = x_dim # number of resources
        self.R = R # resource budget
        self.epsilon = epsilon # radius of ambiguity set
        self.num_balls = num_balls # number of balls
        self.ball_weights = ball_weights # weights for each ball
        self.ball_centres = ball_centres # centres for each ball
        # Store fixed parameters
        self.a = np.array([params[i]['a_i'] for i in range(x_dim)])
        self.b = np.array([params[i]['b_i'] for i in range(x_dim)])
        self.c = np.array([params[i]['c_i'] for i in range(x_dim)])
        self.d = np.array([params[i]['d_i'] for i in range(x_dim)])
 
    def update_epsilon(self, epsilon):
        """Update ambiguity set radius"""
        self.epsilon = epsilon
        
    def update_ball_weights(self, weights):
        """Update ball weights"""
        self.ball_weights = weights

class DROSolver:
    def __init__(self, bounds, num_balls, x_dim):
        """
        Initialize the DRO solver.  
        Args:
            bounds (np.array): Shape (d, 2) containing min and max bounds for each dimension
            num_balls (int): Number of balls to use for covering
        """
        self.bounds = bounds # uncertainty set bounds for each resource
        self.num_balls = num_balls # number of balls
        self.d = x_dim 
        self.ball_weights = None # weights for each ball
        self.ball_centres = None # centres for each ball
        self.radius = None  # Will be computed by find_minimum_radius

    def compute_coverage(self, centers, radius, num_grid_points=50):
        """
        Compute whether the hyperrectangle is completely covered by the balls.
        
        Args:
            centers (np.array): Ball centers
            radius (float): Ball radius
            num_grid_points (int): Number of grid points per dimension
            
        Returns:
            bool: True if hyperrectangle is completely covered, False otherwise
        """
        # Generate grid points with higher density near edges and corners
        grid_points = []
        
        # Add corners of hyperrectangle
        corners = np.array(list(itertools.product(*[[self.bounds[i,0], self.bounds[i,1]] 
                                                  for i in range(self.d)])))
        grid_points.append(corners)
        
        # Add regular grid points
        regular_grid = np.array(np.meshgrid(
            *[np.linspace(self.bounds[i,0], self.bounds[i,1], num_grid_points) 
              for i in range(self.d)]
        )).reshape(self.d, -1).T
        grid_points.append(regular_grid)
        
        # Combine all points
        grid_points = np.unique(np.vstack(grid_points), axis=0)
        
        # Calculate distances from each grid point to nearest ball center
        distances = cdist(grid_points, centers, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Check if all points are covered
        return np.all(min_distances <= radius)
    
    def find_minimum_radius(self, tolerance=1e-4, max_iterations=100):
        """
        Find minimum radius needed for complete coverage using symmetry of hyperrectangle.
        """
        # Initialize bounds for binary search
        # Lower bound: minimum radius based on volume of one orthant
        volume_orthant = np.prod(self.bounds[:,1] - self.bounds[:,0]) / (2**self.d)
        volume_unit_ball = np.pi**(self.d/2) / gamma(self.d/2 + 1)
        balls_per_orthant = max(1, self.num_balls // (2**self.d))
        r_lower = (volume_orthant / (balls_per_orthant * volume_unit_ball))**(1/self.d)
        
        # Upper bound: radius from center to corner of orthant
        r_upper = np.sqrt(np.sum((self.bounds[:,1])**2))
        
        best_centers = None
        best_radius = r_upper
        
        while (r_upper - r_lower) > tolerance:
            r_mid = (r_lower + r_upper) / 2
            found_covering = False
            
            # Try multiple initial positions in first orthant
            for _ in range(max_iterations):
                # Try to find covering with current radius
                def objective(centers_flat):
                    centers = centers_flat.reshape(-1, self.d)
                    # Generate grid points in first orthant
                    grid_points = np.array(np.meshgrid(
                        *[np.linspace(0, self.bounds[i,1], 10) 
                          for i in range(self.d)]
                    )).reshape(self.d, -1).T
                    
                    distances = cdist(grid_points, centers, metric='euclidean')
                    min_distances = np.min(distances, axis=1)
                    return np.sum((np.maximum(0, min_distances - r_mid))**2)
                
                # Initial centers in first orthant
                initial_centers = np.random.uniform(
                    0, self.bounds[:,1], 
                    size=(balls_per_orthant, self.d)
                )
                
                # Optimize ball positions in first orthant
                result = minimize(
                    objective,
                    initial_centers.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, self.bounds[i%self.d,1]) 
                           for i in range(balls_per_orthant * self.d)],
                    options={'maxiter': 1000}
                )
                
                orthant_centers = result.x.reshape(-1, self.d)
                
                # Generate all centers using symmetry
                all_centers = []
                for signs in itertools.product([-1, 1], repeat=self.d):
                    signs = np.array(signs)
                    reflected_centers = orthant_centers * signs
                    all_centers.append(reflected_centers)
                centers = np.vstack(all_centers)
                
                # Take only the required number of balls
                centers = centers[:self.num_balls]
                
                # Check if this is a valid covering
                if self.compute_coverage(centers, r_mid):
                    found_covering = True
                    best_centers = centers
                    best_radius = r_mid
                    r_upper = r_mid
                    break
            
            if not found_covering:
                r_lower = r_mid
        
        if best_centers is None:
            raise ValueError("Failed to find valid covering")
        
        self.radius = best_radius
        self.ball_centers = best_centers
        return self.radius

    def generate_ball_covering(self):
        """
        Generate an optimal covering of a hyperrectangle using Euclidean balls.
        """
        if self.radius is None:
            self.find_minimum_radius()
        return self.ball_centers
    
    
    def solve_max_problem(self, x_current, weights_current, dro_params, max_iter_fw=1, u_current=None, v_current=None):
        """Solve the maximization problem using Frank-Wolfe with efficient projections"""
        
        def project_onto_constraints(u, v):
            """
            Project (u,v) onto the feasible set defined by:
            - Non-negativity: u,v >= 0
            - Bound constraints: u + v <= bounds
            - Weighted sum constraint: sum(weights_k^2 * (u_k + v_k)) = epsilon
            """
            # Define decision variables for projection
            u_proj = cp.Variable(u.shape)
            v_proj = cp.Variable(v.shape)
            
            # Objective: minimize the distance to the original (u, v)
            objective = cp.Minimize(cp.sum_squares(u_proj - u) + cp.sum_squares(v_proj - v))
            
            # Constraints
            constraints = [
                u_proj >= 0,
                v_proj >= 0
            ]
            
            # Hyperrectangle constraints: 0 <= s_u + s_v <= bounds
            for i in range(dro_params.x_dim):
                constraints.append(u_proj[:, i] + v_proj[:, i] <= self.bounds[i, 1])
            
            # Weighted sum constraint
            weights_squared = np.sqrt(weights_current)
            weighted_sum = cp.sum(cp.multiply(weights_squared.reshape(-1, 1), u_proj + v_proj))
            constraints.append(weighted_sum == dro_params.epsilon)
            
            # Solve the quadratic program
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.SCS)  # You can choose another solver if needed
                
                if problem.status != cp.OPTIMAL:
                    raise ValueError(f"Projection problem failed to solve optimally. Status: {problem.status}")
                
                return u_proj.value, v_proj.value
                
            except cp.error.SolverError:
                print("Warning: Projection failed. Returning input values.")
                return u, v
        
        def compute_gradients(u, v):
            """Compute gradients for the max problem objective with respect to u and v"""
            grad_u = np.zeros_like(u)
            grad_v = np.zeros_like(v)
            
            for i in range(dro_params.x_dim):
                for k in range(self.num_balls):
                    # Current perturbed center for ball k
                    perturbed_center = dro_params.ball_centres[k] + u[k] + v[k]

                    # Gradient components
                    grad_u[k,i] = weights_current[k] * (
                        2 * dro_params.c[i] * x_current[i] * 
                        (perturbed_center[i] * x_current[i] - 1) -
                        2 * dro_params.d[i] * perturbed_center[i]
                    )
                    grad_v[k,i] = grad_u[k,i]  # Same gradient for v
                    
            return grad_u, grad_v
        
        def linear_oracle(grad_u, grad_v, weights_current, dro_params):
            """
            Solve the linear optimization subproblem for Frank-Wolfe:
            min_{s_u,s_v} <grad_u,s_u> + <grad_v,s_v>
            subject to:
                sum_{k,i} weights_k^2*(s_u_k_i + s_v_k_i) = epsilon*Delta
                0 <= s_u_k_i + s_v_k_i <= bounds[:,1]
                s_u_k_i, s_v_k_i >= 0
            
            Returns:
                tuple: (s_u, s_v) optimal vertex of the feasible set
            """
            # Decision variables
            s_u = cp.Variable((self.num_balls, dro_params.x_dim))
            s_v = cp.Variable((self.num_balls, dro_params.x_dim))
            
            # Objective: minimize inner product with gradients
            objective = cp.sum(cp.multiply(grad_u, s_u) + cp.multiply(grad_v, s_v))
            
            # Constraints
            constraints = []
            
            # Non-negativity constraints
            constraints.append(s_u >= 0)
            constraints.append(s_v >= 0)
            
            # Hyperrectangle constraints: 0 <= s_u + s_v <= bounds
            for i in range(dro_params.x_dim):
                constraints.append(s_u[:, i] + s_v[:, i] <= self.bounds[i, 1])
            
            # Weighted sum constraint
            weights_squared = weights_current**2
            weighted_sum = cp.sum(cp.multiply(weights_squared.reshape(-1, 1), s_u + s_v))
            constraints.append(weighted_sum == dro_params.epsilon)
            
            # Solve the linear program
            problem = cp.Problem(cp.Minimize(objective), constraints)
            try:
                problem.solve(solver=cp.MOSEK)  # or another LP solver
                
                if problem.status != cp.OPTIMAL:
                    raise ValueError(f"Linear oracle failed to solve optimally. Status: {problem.status}")
                
                return s_u.value, s_v.value
                
            except cp.error.SolverError:
                print("Warning: Solver failed. Returning zeros.")
                return np.zeros_like(grad_u), np.zeros_like(grad_v)
       
        # Initialize if not provided
        if u_current is None:
            u_current = np.zeros((self.num_balls, dro_params.x_dim))
        if v_current is None:
            v_current = np.zeros((self.num_balls, dro_params.x_dim))
        
        for t in range(max_iter_fw):
            # Compute gradients
            grad_u, grad_v = compute_gradients(u_current, v_current)
            
            # Get feasible descent direction using linear oracle
            s_u, s_v = linear_oracle(
                -grad_u,  # Negative because we're maximizing
                -grad_v,
                weights_current,
                dro_params
            )
            
            # Update with step size
            gamma = 0.5
            u_next = u_current + gamma * (s_u - u_current)
            v_next = v_current + gamma * (s_v - v_current)
            
            # Project to ensure feasibility
            u_next, v_next = project_onto_constraints(u_next, v_next)
            
            u_current, v_current = u_next, v_next
        
        return u_current, v_current


    def solve_min_problem_gd(self, u_prev, v_prev, weights_current, dro_params, max_iter=1, learning_rate=0.01, tol=1e-6, x_current=None):
        """Solve the minimization problem using gradient descent."""
        # Compute perturbed centers
        perturbed_centers = np.array([
            dro_params.ball_centres[k] + u_prev[k] + v_prev[k]
            for k in range(self.num_balls)
        ])
        
        def objective(x):
            """Compute objective value"""
            total_cost = 0
            for i in range(dro_params.x_dim):
                for k in range(self.num_balls):
                    total_cost += weights_current[k] * (
                        dro_params.a[i] * (x[i] - dro_params.b[i])**2 + 
                        dro_params.c[i] * (perturbed_centers[k,i] * x[i] - 1)**2 - 
                        dro_params.d[i] * perturbed_centers[k,i]**2
                    )
            return total_cost
        
        def gradient(x):
            """Compute gradient of objective"""
            grad = np.zeros(dro_params.x_dim)
            for i in range(dro_params.x_dim):
                for k in range(self.num_balls):
                    # Derivative of quadratic terms
                    grad[i] += weights_current[k] * (
                        2 * dro_params.a[i] * (x[i] - dro_params.b[i]) + 
                        2 * dro_params.c[i] * perturbed_centers[k,i] * (perturbed_centers[k,i] * x[i] - 1)
                    )
            return grad
        
        def project_onto_constraints(x):
            """Project onto non-negative orthant and resource constraint"""
            x = np.maximum(0, x)  # Non-negativity
            if np.sum(x) > dro_params.R:  # Resource constraint
                x = x * (dro_params.R / np.sum(x))
            return x
                
        # Gradient descent iterations (1 step)
        for iter in range(max_iter):           
            # Compute gradient
            grad = gradient(x_current)
            # Update with gradient step
            x_next = x_current - learning_rate * grad
            # Project onto feasible set
            x_next = project_onto_constraints(x_next)
        
        # Return both next point and objective value
        return x_next, objective(x_next)


    def visualize_covering(self):
        """
        Visualize the ball covering (only for 2D problems).
        """
        if self.d != 2:
            raise ValueError("Visualization only supported for 2D problems")
            
        if self.ball_centers is None:
            self.generate_ball_covering()
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot rectangle bounds
        ax.add_patch(plt.Rectangle(
            (self.bounds[0,0], self.bounds[1,0]),
            self.bounds[0,1] - self.bounds[0,0],
            self.bounds[1,1] - self.bounds[1,0],
            fill=False, color='black'
        ))
        
        # Plot balls
        for center in self.ball_centers:
            circle = plt.Circle(center, self.radius, alpha=0.3)
            ax.add_patch(circle)
            
        ax.set_xlim(self.bounds[0,0] - self.radius, self.bounds[0,1] + self.radius)
        ax.set_ylim(self.bounds[1,0] - self.radius, self.bounds[1,1] + self.radius)
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()

    def get_optimal_covering(self):
        """
        Get the optimal covering parameters.
        
        Returns:
            tuple: (ball_centers, minimum_radius)
        """
        if self.radius is None:
            self.find_minimum_radius()
        return self.ball_centers, self.radius

    def generate_data_and_weights(self, N, seed=None):
        """
        Generate samples from mixture distribution and compute ball weights.
        
        Args:
            N (int): Number of samples to generate
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            tuple: (samples, weights)
                - samples: Generated and clipped samples
                - weights: Weight for each ball (proportion of samples in each ball)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate samples from mixture
        samples = self.sample_from_mixture(size=N)
        
        # Ensure ball centers are computed
        if self.ball_centers is None:
            self.generate_ball_covering()
        
        # Compute distances between samples and ball centers
        distances = cdist(samples, self.ball_centers, metric='euclidean')
        
        # Assign each sample to nearest ball center
        ball_assignments = np.argmin(distances, axis=1)
        
        # Compute weights as proportion of samples in each ball
        weights = np.zeros(self.num_balls)
        for i in range(self.num_balls):
            weights[i] = np.sum(ball_assignments == i) / N
            
        return samples, weights

    def update_weights_with_new_sample(self, sample, current_weights, N, beta, C, diam):
        """
        Update ball weights and radius when a new sample arrives.
        
        Args:
            sample (np.array): New data point
            current_weights (np.array): Current weights of balls
            N (int): Total number of samples including the new one
            beta (float): Confidence parameter
            C (float): Constant for radius computation
            diam (float): Diameter of uncertainty set
            
        Returns:
            tuple: (new_weights, new_radius)
        """
        # Ensure ball centers are available
        if self.ball_centers is None:
            self.generate_ball_covering()
        
        # Ensure sample is 2D array (1 x d)
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        # Find closest ball center to new sample
        distances = cdist(sample, self.ball_centers, metric='euclidean')
        assigned_ball = np.argmin(distances)
        
        # Update weights
        new_weights = current_weights.copy()
        new_weights = new_weights * (N - 1) / N  # Scale down old counts
        new_weights[assigned_ball] += 1/N  # Add new sample
        
        # Update radius based on new N
        new_radius = diam * (C/N + np.sqrt((2*np.log(1/beta)))/np.sqrt(N))
        
        return new_weights, new_radius



    def visualize_samples_and_covering(self, initial_samples, online_samples=None):
        """
        Visualize the ball covering and sample distributions (only for 2D problems).
        
        Args:
            initial_samples (np.array): Initial samples used for weight initialization
            online_samples (np.array, optional): Samples collected during online learning
        """
        if self.d != 2:
            raise ValueError("Visualization only supported for 2D problems")
        
        if self.ball_centers is None:
            self.generate_ball_covering()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Initial samples
        ax1.set_title("Initial Samples Distribution")
        
        # Plot rectangle bounds
        ax1.add_patch(plt.Rectangle(
            (self.bounds[0,0], self.bounds[1,0]),
            self.bounds[0,1] - self.bounds[0,0],
            self.bounds[1,1] - self.bounds[1,0],
            fill=False, color='black'
        ))
        
        # Plot balls
        for center in self.ball_centers:
            circle = plt.Circle(center, self.radius, alpha=0.2, color='blue')
            ax1.add_patch(circle)
        
        # Plot initial samples
        ax1.scatter(initial_samples[:,0], initial_samples[:,1], 
                   c='red', alpha=0.6, label='Initial samples')
        
        # Plot ball centers
        ax1.scatter(self.ball_centers[:,0], self.ball_centers[:,1], 
                   c='blue', marker='x', s=100, label='Ball centers')
        
        ax1.set_xlim(self.bounds[0,0] - self.radius, self.bounds[0,1] + self.radius)
        ax1.set_ylim(self.bounds[1,0] - self.radius, self.bounds[1,1] + self.radius)
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: All samples
        ax2.set_title("All Samples Distribution")
        
        # Plot rectangle bounds
        ax2.add_patch(plt.Rectangle(
            (self.bounds[0,0], self.bounds[1,0]),
            self.bounds[0,1] - self.bounds[0,0],
            self.bounds[1,1] - self.bounds[1,0],
            fill=False, color='black'
        ))
        
        # Plot balls
        for center in self.ball_centers:
            circle = plt.Circle(center, self.radius, alpha=0.2, color='blue')
            ax2.add_patch(circle)
        
        # Plot initial samples
        ax2.scatter(initial_samples[:,0], initial_samples[:,1], 
                   c='red', alpha=0.6, label='Initial samples')
        
        # Plot online samples if provided
        if online_samples is not None:
            ax2.scatter(online_samples[:,0], online_samples[:,1], 
                       c='green', alpha=0.6, label='Online samples')
        
        # Plot ball centers
        ax2.scatter(self.ball_centers[:,0], self.ball_centers[:,1], 
                   c='blue', marker='x', s=100, label='Ball centers')
        
        ax2.set_xlim(self.bounds[0,0] - self.radius, self.bounds[0,1] + self.radius)
        ax2.set_ylim(self.bounds[1,0] - self.radius, self.bounds[1,1] + self.radius)
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def sample_from_mixture(self, size=1): # now they are equally weighted
        """
        Sample from a mixture of Gaussians centered in different quadrants.
        
        Args:
            size (int): Number of samples to generate
            
        Returns:
            np.array: Generated and clipped samples
        """
        samples = []
        for _ in range(size):
            r = np.random.random()
            if r < 0.25:  # Upper right quadrant
                mu_quad = np.mean([self.bounds[:,1], (self.bounds[:,0] + self.bounds[:,1])/2], axis=0)
                sigma_quad = np.diag((self.bounds[:,1] - self.bounds[:,0])/4)**2
                sample = np.random.multivariate_normal(mu_quad, sigma_quad, size=1)
            elif r < 0.5:  # Lower left quadrant
                mu_quad = np.mean([self.bounds[:,0], (self.bounds[:,0] + self.bounds[:,1])/2], axis=0)
                sigma_quad = np.diag((self.bounds[:,1] - self.bounds[:,0])/4)**2
                sample = np.random.multivariate_normal(mu_quad, sigma_quad, size=1)
            elif r < 0.75:  # Upper left quadrant
                mu_quad = np.array([(self.bounds[0,0] + self.bounds[0,1])/4, 
                                  3*(self.bounds[1,0] + self.bounds[1,1])/4])
                sigma_quad = np.diag((self.bounds[:,1] - self.bounds[:,0])/4)**2
                sample = np.random.multivariate_normal(mu_quad, sigma_quad, size=1)
            else:  # Lower right quadrant
                mu_quad = np.array([3*(self.bounds[0,0] + self.bounds[0,1])/4, 
                                  (self.bounds[1,0] + self.bounds[1,1])/4])
                sigma_quad = np.diag((self.bounds[:,1] - self.bounds[:,0])/4)**2
                sample = np.random.multivariate_normal(mu_quad, sigma_quad, size=1)
            samples.append(sample.flatten())
        
        samples = np.array(samples)
        # Ensure samples stay within bounds
        samples = np.clip(samples, self.bounds[:,0], self.bounds[:,1])
        return samples

    def solve_max_problem_exact(self, x_current, weights_current, dro_params):
        """
        Solve the maximization problem exactly using CVXPY.
        
        Args:
            x_current (np.array): Current resource allocation
            weights_current (np.array): Current ball weights
            dro_params (DROParameters): Problem parameters
            
        Returns:
            tuple: (u_opt, v_opt) optimal perturbation vectors
        """
        # Decision variables
        y = cp.Variable((self.num_balls, dro_params.x_dim))
        
        # Build objective function
        obj_terms = []
        for i in range(dro_params.x_dim):
            for k in range(self.num_balls):
                # Perturbed center for ball k
                perturbed_center_ki = dro_params.ball_centres[k, i] + y[k, i]
                
                # Add weighted objective terms
                obj_terms.append(
                    weights_current[k] * (
                        dro_params.a[i] * (x_current[i] - dro_params.b[i])**2 + 
                        dro_params.c[i] * perturbed_center_ki * (x_current[i] - 1)**2 - 
                        dro_params.d[i] * perturbed_center_ki**2
                    )
                )
        
        # Maximize the sum of objective terms
        objective = cp.Maximize(cp.sum(obj_terms))
        
        # Constraints
        constraints = []
        
        # Hyperrectangle constraints for perturbed centers
        for k in range(self.num_balls):
            for i in range(dro_params.x_dim):
                constraints.append(dro_params.ball_centres[k, i] + y[k, i] >= self.bounds[i, 0])
                constraints.append(dro_params.ball_centres[k, i] + y[k, i] <= self.bounds[i, 1])
        
        # Weighted sum of squared norms constraint
        weighted_norms = cp.sum([
            weights_current[k] * cp.norm(y[k, :], 2)**2 
            for k in range(self.num_balls)
        ])
        constraints.append(weighted_norms <= dro_params.epsilon)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()  # or another suitable solver
        
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Max problem failed to solve optimally. Status: {problem.status}")
            
            # Split y into u and v for compatibility with rest of code
            y_val = y.value
            u_val = np.maximum(y_val, 0)
            v_val = np.maximum(-y_val, 0)
            return u_val, v_val
            
        except cp.error.SolverError:
            print("Warning: Max problem solver failed.")
            return np.zeros((self.num_balls, dro_params.x_dim)), np.zeros((self.num_balls, dro_params.x_dim))

    def solve_min_problem_exact(self, y_current,weights_current, dro_params):
        """
        Solve the minimization problem exactly using CVXPY.
        
        Args:
            u_current (np.array): Current u perturbation
            v_current (np.array): Current v perturbation
            weights_current (np.array): Current ball weights
            dro_params (DROParameters): Problem parameters
            
        Returns:
            tuple: (x_opt, obj_value) optimal allocation and objective value
        """
        # Compute perturbed centers
        perturbed_centers = np.array([
            dro_params.ball_centres[k] + y_current[k]
            for k in range(self.num_balls)
        ])
        
        # Decision variable
        x = cp.Variable(dro_params.x_dim)
        
        # Build objective function
        obj_terms = []
        for i in range(dro_params.x_dim):
            # Resource allocation cost
            obj_terms.append(
                cp.sum([
                    weights_current[k] * (
                        dro_params.a[i] * (x[i] - dro_params.b[i])**2 + 
                        dro_params.c[i] * perturbed_centers[k,i] * (x[i] - 1)**2 - 
                        dro_params.d[i] * perturbed_centers[k,i]**2
                    )
                    for k in range(self.num_balls)
                ])
            )
        
        # Minimize the sum of costs
        objective = cp.Minimize(cp.sum(obj_terms))
        
        # Constraints
        constraints = [
            x >= 0,  # Non-negativityI
            cp.sum(x) <= dro_params.R  # Resource budget
        ]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()  # or another suitable solver
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Min problem failed to solve optimally. Status: {problem.status}")
            
            return x.value, problem.value
            
        except cp.error.SolverError:
            print("Warning: Min problem solver failed.")
            return np.zeros(dro_params.x_dim), np.inf

# Example usage
if __name__ == "__main__":
    # Algorithm selection flag
    use_exact_solver = True  # Set to False to use gradient-based methods
    
    # Set up uncertainty set and covering
    bounds = np.array([[-1, 1], [-1, 1]])
    num_balls = 4
    x_dim = 2
    solver = DROSolver(bounds, num_balls, x_dim)    
    
    # Preprocessing: Get the ball centers and radius
    solver.generate_ball_covering()  # This will set both ball_centers and radius
    print(f"Optimal ball radius: {solver.radius}")
    print(f"Ball centers:\n{solver.ball_centers}")
    
    # Initial problem parameters
    R = 5
    T = 10  # Number of timesteps
    
    # Distribution parameters for sampling
    mu = np.zeros(x_dim)  # Mean of underlying normal distribution
    sigma = 0.05*np.eye(x_dim)  # Covariance of underlying normal distribution
    
    # Initial sample size and DRO parameters
    N = 10  # initial number of samples
    diam = 2*np.sqrt(x_dim)
    C = 1/np.sqrt(2)*3
    beta = 0.1
    
    # Generate initial samples and weights
    samples, weights = solver.generate_data_and_weights(N, seed=42)
    
    # Initial radius computation
    radius_init = diam*(C/N + np.sqrt((2*np.log(1/beta)))*1/np.sqrt(N))
    initial_epsilon = radius_init + solver.radius  # Use solver.radius instead of ball_radius
    
    # Fixed parameters for each resource
    fixed_params = {
        0: {'a_i': 1.0, 'b_i': 2.0, 'c_i': 0.5, 'q_i': 0.3, 'd_i': 0.8},
        1: {'a_i': 1.2, 'b_i': 1.5, 'c_i': 0.6, 'q_i': 0.4, 'd_i': 0.7},
    }
    
    # Create DRO parameters object
    dro_params = DROParameters(x_dim, num_balls, R, fixed_params, 
                             epsilon=initial_epsilon,
                             ball_weights=weights,
                             ball_centres=solver.ball_centers)  # Use solver.ball_centers
    
    # Initialize solutions
    x_current = np.zeros(x_dim)
    u_current = np.ones((num_balls, x_dim)) / x_dim
    v_current = np.ones((num_balls, x_dim)) / x_dim
    
    # History for analysis
    history = {
        'x': [],
        'obj_values': [],
        'epsilon': [],
        'weights': []
    }

    # Plot initial configuration
    solver.visualize_samples_and_covering(samples, online_samples=None)
    
    # Main iteration loop
    online_samples = []  # Store online samples
    
    print(f"Running with {'exact' if use_exact_solver else 'gradient-based'} solver")

    for t in range(T):
        print(f"\nTimestep {t+1}/{T}")
        
        if use_exact_solver:
            # Use exact solvers
            u_next, v_next = solver.solve_max_problem_exact(
                x_current,
                weights,
                dro_params
            )
            
            x_next, min_obj = solver.solve_min_problem_exact(
                u_next + v_next,
                weights,
                dro_params
            )
        else:
            # Use gradient-based methods
            u_next, v_next = solver.solve_max_problem(
                x_current,
                weights,
                dro_params,
                max_iter_fw=10,  # Increase iterations for better convergence
                u_current=u_current,
                v_current=v_current
            )
            
            x_next, min_obj = solver.solve_min_problem_gd(
                u_next,
                v_next,
                weights,
                dro_params,
                max_iter=50,  # Increase iterations for better convergence
                x_init=x_current
            )
        
        # Rest of the loop remains the same
        new_sample = solver.sample_from_mixture(size=1)
        new_sample = np.clip(new_sample, bounds[:,0], bounds[:,1])
        online_samples.append(new_sample.flatten())
        
        N += 1
        weights, new_radius = solver.update_weights_with_new_sample(
            new_sample, weights, N, beta, C, diam
        )
        
        new_epsilon = new_radius + solver.radius
        dro_params.update_epsilon(new_epsilon)
        dro_params.update_ball_weights(weights)
        
        x_current = x_next
        u_current = u_next
        v_current = v_next
        
        history['x'].append(x_current)
        history['obj_values'].append(min_obj)
        history['epsilon'].append(new_epsilon)
        history['weights'].append(weights.copy())
        
        print(f"Current allocation: {x_current}")
        print(f"Current epsilon: {new_epsilon}")
        print(f"Weight sum: {np.sum(weights)}")
    
    # Convert online samples to numpy array
    online_samples = np.array(online_samples)

    # Create visualizations
    solver.visualize_samples_and_covering(samples, online_samples)

    # Original results plotting
    plt.figure(figsize=(15, 4))  # Made wider to accommodate new plot

    plt.subplot(141)  # Changed to 4 subplots
    plt.plot([x[0] for x in history['x']], label='Resource 1')
    plt.plot([x[1] for x in history['x']], label='Resource 2')
    plt.xlabel('Timestep')
    plt.ylabel('Resource Allocation')
    plt.legend()

    plt.subplot(142)
    plt.plot(history['epsilon'])
    plt.xlabel('Timestep')
    plt.ylabel('Epsilon')

    plt.subplot(143)
    plt.plot(np.array(history['weights']))
    plt.xlabel('Timestep')
    plt.ylabel('Ball Weights')

    plt.subplot(144)  # New subplot for objective values
    plt.plot(history['obj_values'], 'r-')
    plt.xlabel('Timestep')
    plt.ylabel('Min Problem Objective')
    plt.title('Optimal Value Evolution')

    plt.tight_layout()
    plt.show()
    
