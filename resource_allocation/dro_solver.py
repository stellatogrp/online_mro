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
        self.num_balls = num_ball # number of balls
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
    
    
    def solve_max_problem(self, x_current, weights_current, dro_params, max_iter_fw=1, y_current=None):
        """
        Solve the maximization problem using Frank-Wolfe algorithm.
        
        Args:
            x_current: Current resource allocation
            weights_current: Current ball weights
            dro_params: Problem parameters
            max_iter_fw: Number of Frank-Wolfe iterations
            y_current: Current perturbation (warm start) = u_current + v_current
        """
        
        def compute_gradient(y):
            """Compute gradient of the objective with respect to y"""
            grad = np.zeros_like(y)
            
            for i in range(dro_params.x_dim):
                for k in range(self.num_balls):
                    # Current perturbed center for ball k
                    perturbed_center_ki = dro_params.ball_centres[k, i] + y[k, i]
                    
                    # Gradient component
                    grad[k, i] = weights_current[k] * (
                        2 * dro_params.c[i] * x_current[i] * 
                        (perturbed_center_ki * x_current[i] - 1) -
                        2 * dro_params.d[i] * perturbed_center_ki
                    )
            return grad
        
        def linear_oracle(grad):
            """
            Solve the linear minimization problem over the feasible set:
            min_s <grad, s> 
            s.t. sum(weights_k * ||s_k||_2^2) <= epsilon
                 ball_centres + s in hyperrectangle
            """
            # Decision variables
            s = cp.Variable((self.num_balls, dro_params.x_dim))
            
            # Objective: maximize inner product with gradient
            objective = cp.Maximize(cp.sum(cp.multiply(grad, s)))
            
            # Constraints
            constraints = []
            
            # Weighted sum of squared norms constraint
            weighted_norms = cp.sum([
                weights_current[k] * cp.norm(s[k, :], 2)**2 
                for k in range(self.num_balls)
            ])
            constraints.append(weighted_norms <= dro_params.epsilon)
            
            # Hyperrectangle constraints for perturbed centers
            for k in range(self.num_balls):
                for i in range(dro_params.x_dim):
                    constraints.append(
                        dro_params.ball_centres[k, i] + s[k, i] >= self.bounds[i, 0]
                    )
                    constraints.append(
                        dro_params.ball_centres[k, i] + s[k, i] <= self.bounds[i, 1]
                    )
            
            # Solve the problem
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.MOSEK)  # or another suitable solver
                
                if problem.status != cp.OPTIMAL:
                    raise ValueError(f"Linear oracle failed to solve optimally. Status: {problem.status}")
                
                return s.value
                
            except cp.error.SolverError:
                print("Warning: Linear oracle solver failed.")
                return np.zeros_like(grad)
        
        # Frank-Wolfe iterations
        for t in range(max_iter_fw):
            # Compute gradient at current point
            grad = compute_gradient(y_current)
            
            # Get descent direction by solving linear oracle
            s = linear_oracle(-grad)  # Negative because we're maximizing
            
            # Fixed step size
            gamma = 0.5  
            
            # Update
            y_next = y_current + gamma * (s - y_current)
            
            # Update current point
            y_current = y_next

    
        return y_next

    def solve_min_problem_gd(self, y_prev, weights_current, dro_params, max_iter=1, learning_rate=0.01, tol=1e-6, x_current=None):
        """Solve the minimization problem using gradient descent."""
        # Compute perturbed centers
        perturbed_centers = np.array([
            dro_params.ball_centres[k] + y_prev[k]
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

    def solve_min_problem_exact(self, y_current, weights_current, dro_params):

        # Compute perturbed centers
        perturbed_centers = np.array([
            dro_params.ball_centres[k] + y_current[k]
            for k in range(self.num_balls)
        ])
        
        # Decision variable
        x = cp.Variable(dro_params.x_dim)
        
        # Build objective function - reformulated to be DCP compliant
        obj_terms = []
        for i in range(dro_params.x_dim):
            resource_terms = []
            for k in range(self.num_balls):
                # Split the quadratic terms to ensure DCP compliance
                allocation_cost = dro_params.a[i] * cp.power(x[i] - dro_params.b[i], 2)
                service_cost = dro_params.c[i] * cp.power(x[i], 2) * cp.power(perturbed_centers[k,i], 2)
                linear_term = -2 * dro_params.c[i] * perturbed_centers[k,i] * x[i]
                constant_term = dro_params.c[i] - dro_params.d[i] * cp.power(perturbed_centers[k,i], 2)
                
                resource_terms.append(
                    weights_current[k] * (allocation_cost + service_cost + linear_term + constant_term)
                )
            obj_terms.extend(resource_terms)
        
        # Minimize the sum of costs
        objective = cp.Minimize(cp.sum(obj_terms))
        
        # Constraints
        constraints = [
            x >= 0,  # Non-negativity
            cp.sum(x) <= dro_params.R  # Resource budget
        ]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve() 
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Min problem failed to solve optimally. Status: {problem.status}")
            
            return x.value, problem.value
        
        except cp.error.SolverError:
            print("Warning: Min problem solver failed.")
            return np.zeros(dro_params.x_dim), np.inf

    

    def compute_saa_solution(self, num_samples=1000, seed=42):
        """
        Compute the Sample Average Approximation (SAA) solution using a large number of samples.
        This serves as an approximation of the true stochastic optimization solution.
        
        Args:
            num_samples (int): Number of samples to use for SAA 
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (x_saa, obj_value)
                - x_saa: Optimal resource allocation
                - obj_value: Optimal objective value
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate samples from true distribution
        samples = self.sample_from_mixture(size=num_samples)
        samples = np.clip(samples, self.bounds[:,0], self.bounds[:,1])
        
        # Define decision variables
        x = cp.Variable(self.d)
        
        # Compute average cost over all samples
        total_cost = 0
        for i in range(self.d):  # For each resource
            sample_costs = 0
            for sample in samples:
                # Cost for current sample and resource
                sample_costs += (1/num_samples) * (
                    dro_params.a[i] * (x[i] - dro_params.b[i])**2 + 
                    dro_params.c[i] * (sample[i] * x[i] - 1)**2 - 
                    dro_params.d[i] * sample[i]**2
                )
            total_cost += sample_costs
        
        # Define constraints
        constraints = [
            cp.sum(x) <= dro_params.R,  # Total resource constraint
            x >= 0  # Non-negativity
        ]
        
        # Define and solve the problem
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        try:
            problem.solve(solver=cp.MOSEK)  # or another suitable solver
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"SAA problem failed to solve optimally. Status: {problem.status}")
            
            return x.value, problem.value
            
        except cp.error.SolverError:
            print("Warning: SAA solver failed.")
            return None, None

    def compute_cumulative_regret(self, history, dro_params, online_samples, num_eval_samples=1000, seed=42):
        """
        Compute cumulative regret by comparing online decisions against optimal DRO solution in hindsight.
        At each time t, use the same samples that were available to the online policy.
        
        Args:
            history (dict): History of online decisions and parameters
            dro_params (DROParameters): Problem parameters
            online_samples (np.array): Array of observed samples
            num_eval_samples (int): Number of samples to use for SAA evaluation
            seed (int): Random seed for reproducibility
        """
        T = len(online_samples)  # Number of timesteps
        regret = np.zeros(T)  # Instantaneous regret at each timestep
        cumulative_regret = np.zeros(T)  # Cumulative regret up to each timestep
        optimal_hindsight_solutions = []
        
        # Generate evaluation samples from true distribution for cost computation
        np.random.seed(seed)
        eval_samples = self.sample_from_mixture(size=num_eval_samples)
        eval_samples = np.clip(eval_samples, self.bounds[:,0], self.bounds[:,1])
        
        def evaluate_expected_cost(x):
            """Compute expected cost under true distribution using SAA"""
            total_cost = 0
            for sample in eval_samples:
                cost = 0
                for i in range(self.d):
                    cost += (
                        dro_params.a[i] * (x[i] - dro_params.b[i])**2 + 
                        dro_params.c[i] * (sample[i] * x[i] - 1)**2 - 
                        dro_params.d[i] * sample[i]**2
                    )
                total_cost += cost / num_eval_samples
            return total_cost
        
        # For each timestep t
        for t in range(T):
            # Use the same weights and epsilon that were used by online policy at time t
            weights = history['weights'][t]
            epsilon = history['epsilon'][t]
            
            # Create DRO parameters using historical values
            dro_params_t = DROParameters(
                self.d, 
                self.num_balls, 
                dro_params.R,
                {i: {'a_i': dro_params.a[i], 
                     'b_i': dro_params.b[i], 
                     'c_i': dro_params.c[i], 
                     'd_i': dro_params.d[i]} for i in range(self.d)},
                epsilon=epsilon,
                ball_weights=weights,
                ball_centres=dro_params.ball_centres
            )
            
            # Get optimal solution in hindsight at time t
            x_opt, _ = self.solve_min_problem_exact(
                np.zeros((self.num_balls, self.d)),  # Initial y
                weights,
                dro_params_t
            )
            
            optimal_hindsight_solutions.append(x_opt)
            
            # Compute instantaneous regret at time t using true distribution
            online_cost = evaluate_expected_cost(history['x'][t])
            optimal_cost = evaluate_expected_cost(x_opt)
            regret[t] = online_cost - optimal_cost
            
            # Update cumulative regret
            if t == 0:
                cumulative_regret[t] = regret[t]
            else:
                cumulative_regret[t] = cumulative_regret[t-1] + regret[t]
        
        return cumulative_regret, optimal_hindsight_solutions, regret

    def plot_regret_analysis(self, cumulative_regret, optimal_hindsight_solutions, instantaneous_regret, history):
        """Plot regret analysis results."""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Cumulative Regret
        plt.subplot(121)
        T = len(cumulative_regret)
        t_range = np.arange(1, T+1)
        
        # Plot actual cumulative regret
        plt.plot(t_range, cumulative_regret, 'b-', label='Actual Cumulative Regret')
        
        # Plot theoretical bound (sum of 1/sqrt(t))
        theoretical_cumulative = np.cumsum(1/np.sqrt(t_range))
        plt.plot(t_range, theoretical_cumulative, 'r--', 
                 label='Theoretical O(Σ 1/√t)')
        
        plt.xlabel('Timestep (t)')
        plt.ylabel('Cumulative Regret')
        plt.title('Cumulative Regret Analysis')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Instantaneous Regret
        plt.subplot(122)
        plt.plot(t_range, instantaneous_regret, 'g-', label='Instantaneous Regret')
        
        # Plot theoretical bound (1/sqrt(t))
        theoretical_instant = 1/np.sqrt(t_range)
        plt.plot(t_range, theoretical_instant, 'r--', 
                 label='Theoretical O(1/√t)')
        
        plt.xlabel('Timestep (t)')
        plt.ylabel('Instantaneous Regret')
        plt.title('Instantaneous Regret Evolution')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Algorithm selection flag
    use_exact_solver = False  # Set to False to use gradient-based methods

    # Set random seed for reproducibility
    simulation_seed = 1111  # Change this value to get different random sequences
    np.random.seed(simulation_seed)

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
    T = 5000  # Number of timesteps

    # Initial sample size and DRO parameters
    N = 10  # initial number of samples
    diam = 2*np.sqrt(x_dim)
    C = 1/np.sqrt(2)*3
    beta = 0.1

    # Generate initial samples and weights with seed
    samples, weights = solver.generate_data_and_weights(N, seed=simulation_seed)
    
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
                             ball_centres=solver.ball_centers)

    # Initialize solutions
    x_current = np.zeros(x_dim)
    y_current = np.zeros((num_balls, x_dim))

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
    online_samples = []  # Initialize as list

    print(f"Running with {'exact' if use_exact_solver else 'gradient-based'} solver")

    for t in range(T):
        print(f"\nTimestep {t+1}/{T}")
        
        if use_exact_solver:
            # Use exact solvers
            y_next = solver.solve_max_problem_exact(
                x_current,
                weights,
                dro_params
            )
            
            x_next, min_obj = solver.solve_min_problem_exact(
                y_next,
                weights,
                dro_params
            )
        else:
            # Use gradient-based methods
            y_next = solver.solve_max_problem(
                x_current,
                weights,
                dro_params,
                max_iter_fw=1,
                y_current=y_current
            )
            
            x_next, min_obj = solver.solve_min_problem_gd(
                y_next,
                weights,
                dro_params,
                max_iter=1,
                x_current=x_current
            )
        
        # Rest of the loop remains the same
        new_sample = solver.sample_from_mixture(size=1)
        new_sample = np.clip(new_sample, bounds[:,0], bounds[:,1])
        online_samples.append(new_sample.flatten())  # Append to list
        
        N += 1
        weights, new_radius = solver.update_weights_with_new_sample(
            new_sample, weights, N, beta, C, diam
        )
        
        new_epsilon = new_radius + solver.radius
        dro_params.update_epsilon(new_epsilon)
        dro_params.update_ball_weights(weights)
        
        x_current = x_next
        y_current = y_next
        
        history['x'].append(x_current)
        history['obj_values'].append(min_obj)
        history['epsilon'].append(new_epsilon)
        history['weights'].append(weights.copy())
        
        print(f"Current allocation: {x_current}")
        print(f"Current epsilon: {new_epsilon}")
        print(f"Weight sum: {np.sum(weights)}")

    # After all iterations complete, create visualizations
    solver.visualize_samples_and_covering(samples, np.array(online_samples))

    # Results plotting
    plt.figure(figsize=(15, 4))

    plt.subplot(141)
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

    plt.subplot(144)
    plt.plot(history['obj_values'], 'r-')
    plt.xlabel('Timestep')
    plt.ylabel('Min Problem Objective')
    plt.title('Optimal Value Evolution')

    plt.tight_layout()
    plt.show()

    # Compute and plot regret analysis
    cumulative_regret, optimal_hindsight, instantaneous_regret = solver.compute_cumulative_regret(
        history, 
        dro_params, 
        np.array(online_samples),
        seed=simulation_seed  # Use same seed for consistent evaluation
    )

    # Plot regret analysis
    solver.plot_regret_analysis(
        cumulative_regret, 
        optimal_hindsight, 
        instantaneous_regret,
        history
    )
    
