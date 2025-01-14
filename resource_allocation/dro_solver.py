import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.cluster import KMeans
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

def createproblem_resource_allocation(m, dro_params,R,N,dat,eps,w):
    """Create the DRO problem in cvxpy
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample = number of resources
    dro_params: DROParameters
        Object containing all DRO parameters (a_i, b_i, c_i, d_i)
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
   
    # VARIABLES #
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    tau = cp.Variable((N,4))
    xi_l = cp.Variable((N,m))
    xi_c = cp.Variable((N,m))
    bbb = cp.Variable(N)
     
    # OBJECTIVE
    objective = eps*lam + w@s

    # CONSTRAINTS # Note: hard-coded for m = 2 here for visualization purposed
    constraints = []
    for j in range(N):
        ell_convex_conj = sum(
            dro_params.a[i] * (x[i] - dro_params.b[i])**2 + (xi_l[j,i] + dro_params.c[i]*(x[i]-1))**2/(4*dro_params.d[i]) 
            for i in range(m)
        )
        ####cost_convex_conj = lam*(dat[j,:]@dat[j,:] + (xi_c[j]/lam + 2*dat[j,:])@(xi_c[j]/lam + 2*dat[j,:])/4)
        constraints += [ 1/4*cp.quad_over_lin(xi_c[j,:],lam) <= bbb[j]]
        cost_convex_conj = 2*lam*(dat[j,:]@dat[j,:]) + dat[j,:]@xi_c[j,:] + bbb[j]
        f_convex_conj = tau[j,0]*1 + tau[j,1]*1 + tau[j,2]*1 + tau[j,3]*1

        constraints += [ell_convex_conj + cost_convex_conj + f_convex_conj <= s[j]]
        constraints += [xi_l[j,:] + xi_c[j,:] + tau[j,0]*(np.array([1,0])).T + tau[j,1]*(np.array([0,1]).T) + tau[j,2]*(np.array([-1,0]).T) + tau[j,3]*(np.array([0,-1]).T) == 0]
    constraints += [x >= 0, x <= R]
    constraints += [lam - 0.00001 >= 0]
    constraints += [tau[j,k] >= 0 for j in range(N) for k in range(4)]
    
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return problem.objective.value, x.value, problem.solver_stats.solve_time


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
    
    
    def solve_max_problem(self, x_prev, weights_prev, dro_params, rad_prev, max_iter_fw=1, y_prev=None):
        """Solve the maximization problem using Frank-Wolfe algorithm."""
        def compute_gradient(y):
            """Compute gradient of the objective with respect to y evaluated at y_current"""
            grad = np.zeros_like(y)
            
            for i in range(dro_params.x_dim):
                for k in range(self.num_balls):
                    # Current perturbed center for ball k
                    perturbed_center_ki = dro_params.ball_centres[k, i] + y_prev[k, i]
                    
                    # Gradient component evaluated at x_prev
                    grad[k, i] = weights_prev[k] * (
                        dro_params.c[i] * (x_prev[i] - 1) -
                        2 * dro_params.d[i] * perturbed_center_ki
                    )
            return grad
        
        def linear_oracle(grad):
            """
            Solve the linear minimization problem over the feasible set:
            max_s <grad, s> 
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
                weights_prev[k] * cp.norm(s[k, :], 2)**2 
                for k in range(self.num_balls)
            ])
            constraints.append(weighted_norms <= rad_prev)
            
            # Hyperrectangle constraints for perturbed centers
            for k in range(self.num_balls):
                for i in range(dro_params.x_dim):
                    constraints.append(
                        dro_params.ball_centres[k, i] + s[k, i] >= self.bounds[i, 0]
                    )
                    constraints.append(
                        dro_params.ball_centres[k, i] + s[k, i] <= self.bounds[i, 1]
                    )
            
            # Solve the problem and track solver time
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve()
                solver_time = problem.solver_stats.solve_time
                
                
                if problem.status != cp.OPTIMAL:
                    raise ValueError(f"Linear oracle failed to solve optimally. Status: {problem.status}")
                
                return s.value, solver_time
                
            except cp.error.SolverError:
                print("Warning: Linear oracle solver failed.")
                return np.zeros_like(grad), 0.0
        
        # Frank-Wolfe iterations
        total_solver_time = 0
        for t in range(max_iter_fw):
            grad = compute_gradient(y_prev)
            s, solver_time = linear_oracle(grad) 
            total_solver_time += solver_time
            
            gamma = 0.5
            y_next = y_prev + gamma * (s - y_prev)
            y_prev = y_next

        # Notel: worst-case online distribution at time t is given by weights t-1 and delta's at dro_params.ball_centres[k, i] + y_next[k, i]
        return y_next, total_solver_time

    def solve_min_problem_gd(self, y_prev, weights_prev, dro_params, max_iter=1, learning_rate=0.01, tol=1e-6, x_prev=None):
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
                    total_cost += weights_prev[k] * (
                        dro_params.a[i] * (x[i] - dro_params.b[i])**2 + 
                        dro_params.c[i] * perturbed_centers[k,i] * (x[i] - 1) - 
                        dro_params.d[i] * perturbed_centers[k,i]**2
                    )
            return total_cost
        
        def gradient(x):
            """Compute gradient of objective"""
            grad = np.zeros(dro_params.x_dim)
            for i in range(dro_params.x_dim):
                for k in range(self.num_balls):
                    # Derivative of quadratic terms
                    grad[i] += weights_prev[k] * (
                        2 * dro_params.a[i] * (x[i] - dro_params.b[i]) + 
                        dro_params.c[i] * perturbed_centers[k,i]
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
            grad = gradient(x_prev)
            # Update with gradient step
            x_next = x_prev - learning_rate * grad
            # Project onto feasible set
            x_next = project_onto_constraints(x_next)
        
        # Return both next point and objective value
        return x_next

    def evaluate_online_cost(self, y_next, x_next, weights_prev, dro_params):
        """Evaluate online cost"""
        # Worst case dstirbution is given by weights_prev and delta's at dro_params.ball_centres[k, i] + y_next[k, i]
        wc_centers = dro_params.ball_centres + y_next
                            
        # Evaluate cost at x_next
        cost = 0
        for k in range(self.num_balls):
            tmp = 0
            for i in range(dro_params.x_dim):
                tmp +=  (
                    dro_params.a[i] * (x_next[i] - dro_params.b[i])**2 + 
                    dro_params.c[i] * wc_centers[k,i] * (x_next[i] - 1) - 
                    dro_params.d[i] * wc_centers[k,i]**2
                )
            cost += weights_prev[k] * tmp
        return cost

    def visualize_covering(self):
        """
        Visualize the ball covering (only for 2D problems).
        """
        if self.d != 2:
            raise ValueError("Visualization only supported for 2D problems")
            
        if self.ball_centers is None:
            self.generate_ball_covering()
            
        fig, ax = plt.subplots(figsize=(12,5))
        
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



    def visualize_samples_and_covering(self, samples, online_samples=None):
        """Visualize samples and ball covering with LaTeX formatting."""
        if self.d != 2:
            raise ValueError("Visualization only supported for 2D problems")
        
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12
        })
        
        fig, ax = plt.subplots(figsize=(12,5), dpi=300)
        
        # Plot rectangle bounds
        ax.add_patch(plt.Rectangle(
            (self.bounds[0,0], self.bounds[1,0]),
            self.bounds[0,1] - self.bounds[0,0],
            self.bounds[1,1] - self.bounds[1,0],
            fill=False, color='black', linewidth=2
        ))
        
        # Plot balls
        for center in self.ball_centers:
            circle = plt.Circle(center, self.radius, alpha=0.3)
            ax.add_patch(circle)
        
        # Plot samples
        if samples is not None:
            plt.scatter(samples[:,0], samples[:,1], c='blue', alpha=0.5, 
                       label=r'Initial Samples')
        
        if online_samples is not None:
            plt.scatter(online_samples[:,0], online_samples[:,1], c='red', 
                       alpha=0.5, label=r'Online Samples')
        
        ax.set_xlim(self.bounds[0,0] - self.radius, self.bounds[0,1] + self.radius)
        ax.set_ylim(self.bounds[1,0] - self.radius, self.bounds[1,1] + self.radius)
        ax.set_aspect('equal')
        
        plt.xlabel(r'$\xi_1$')
        plt.ylabel(r'$\xi_2$')
        plt.title(r'Sample Distribution and Ball Covering')
        plt.grid(True, alpha=0.3)
        plt.legend(framealpha=0.9)
        
        # Save high-quality figures
        #plt.savefig('covering_visualization.pdf', bbox_inches='tight', dpi=300)
        #plt.savefig('covering_visualization.png', bbox_inches='tight', dpi=300)
        
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


    def evaluate_expected_cost_SAA(self,x_current,data,dro_params,N):
        """Compute expected cost under SAA approach by sampling many data from true distribution"""
    
        obj = 0
        for i in range(dro_params.x_dim):
            for k in range(N):
                    obj = obj + (
                        dro_params.a[i] * (x_current[i] - dro_params.b[i])**2 + 
                        dro_params.c[i] * data[k,i] * (x_current[i] - 1) - 
                        dro_params.d[i] * data[k,i]**2
                    )
        obj = obj/N
                        
        return obj
            
            

    def plot_regret_analysis(self, history):
        """Plot regret analysis with LaTeX formatting."""
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14
        })
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.semilogy(history['cumulative_regret'], 'b-', linewidth=2, label='W.C. cumulative Regret wrt DRO')
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Cumulative Regret')
        plt.title(r'Cumulative Regret over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
 

         # Create figure
        plt.subplot(122)               
        plt.semilogy(history['cumulative_regret_SAA'], 'b-', linewidth=2, label='SAA Cumulative Regret wrt DRO')
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Cumulative Regret')
        plt.title(r'Cumulative Regret over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()



    def plot_results(self, history):
        """Plot results with LaTeX formatting."""
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12
        })
        
        # Create figure with higher DPI
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Resource Allocation
        plt.subplot(141)
        plt.plot([x[0] for x in history['x_online']], 'b-', linewidth=2, label=r'Resource 1')
        plt.plot([x[1] for x in history['x_online']], 'g-', linewidth=2, label=r'Resource 2')
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Resource Allocation')
        plt.grid(True, alpha=0.3)
        plt.legend(framealpha=0.9)
        
        # Plot 2: Epsilon Evolution
        plt.subplot(142)
        plt.plot(history['epsilon'], 'r-', linewidth=2)
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'$\epsilon$')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Ball Weights
        plt.subplot(143)
        plt.plot(np.array(history['weights']), linewidth=2)
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Ball Weights')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Objective Values
        plt.subplot(144)
        plt.plot(history['online_obj_values'], 'r-', linewidth=2)
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Min Problem Objective')
        plt.title(r'Optimal Value Evolution')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout(pad=2.0)
        
        plt.show()

    def plot_computation_times(self, history):
        """Plot CVXPY solver and ambiguity set update time analysis with LaTeX formatting."""
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12
        })
        
        # Calculate total time statistics
        total_times = history['online_computation_times']['step_time']
        total_times_DRO = history['DRO_computation_times']['solver_time']

        # consider exponential
        total_times = np.exp(total_times)
        total_times_DRO = np.exp(total_times_DRO)
        mean_total = np.mean(total_times)
        std_total = np.std(total_times)
        mean_total_DRO = np.mean(total_times_DRO)
        std_total_DRO = np.std(total_times_DRO)


        # Print total time statistics
        print(f"\nTotal Computation Time Statistics:")
        print(f"Mean: {mean_total:.3f} seconds")
        print(f"Standard Deviation: {std_total:.3f} seconds")
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Prepare data for boxplot
        data = [
            total_times, total_times_DRO
        ]
        
        # Create boxplot
        bp = plt.boxplot(data, labels=[
            r'Total Time: Online', r'Total Time: DRO'
        ])
        
        # Customize boxplot colors
        plt.setp(bp['boxes'], color='blue')
        plt.setp(bp['whiskers'], color='blue')
        plt.setp(bp['caps'], color='blue')
        plt.setp(bp['medians'], color='red')

        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.ylabel(r'Compuation time')
        plt.yscale("log")
        #plt.savefig('time.pdf', bbox_inches='tight', dpi=300)
        
        plt.show()

    def plot_computation_times_iter(self,history,T):
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12
        })
        t_range = np.arange(1, T+1)
        plt.figure(figsize=(12, 5))
        plt.plot(t_range, history['online_computation_times']['step_time'], 'b-', linewidth=2, label = "online weight update")
        plt.plot(t_range,  history['DRO_computation_times']['solver_time'], color ='black', linewidth=2, label = "DRO")
        plt.legend()
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Compuation time')
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        #plt.savefig('time_iters.pdf', bbox_inches='tight', dpi=300)
        plt.show()

# Example usage (only for gradient based version)
if __name__ == "__main__":
    # Set random seed for reproducibility
    simulation_seed = 1111
    np.random.seed(simulation_seed)

    # Set up uncertainty set and covering
    bounds = np.array([[-1, 1], [-1, 1]])
    num_balls = 4 # cluster availables
    x_dim = 2 
    m = 2
    solver = DROSolver(bounds, num_balls, x_dim)    

    # Preprocessing: Get the ball centers and radius
    solver.generate_ball_covering()
    print(f"Optimal ball radius: {solver.radius}")
    print(f"Ball centers:\n{solver.ball_centers}")

    # Initial parameters
    T = 100  # Number of timesteps
    N = 10  # initial number of samples
    diam = 2*np.sqrt(x_dim)
    C = 1/np.sqrt(2)*3
    beta = 0.1
    total_data = []

    # History for analysis
    history = {
        'x_online': [],
        'online_obj_values': [],
        'DRO_x': [],
        'DRO_obj_values': [],
        'MRO_x': [],
        'MRO_obj_values': [],
        'epsilon': [],
        'weights': [],
        'online_computation_times': {
            'weight_update': [],
            'step_time': [],
            'total_iteration': []
        },
        'DRO_computation_times':{
        'solver_time':[]
        },
        'istantaneous_regret':[],
        'istantaneous_regret_SAA':[],
        'cumulative_regret':[],
        'cumulative_regret_SAA':[],
    }

    # Generate initial sample and weights
    samples, weights = solver.generate_data_and_weights(N, seed=simulation_seed)
    total_data = samples  # Initialize as numpy array directly
    
    # Initial radius computation for first sample
    radius_init = diam*(C/N + np.sqrt((2*np.log(1/beta)))*1/np.sqrt(N))
    initial_epsilon = radius_init + solver.radius
    

    # Generates lots of sample to test with SAA approach
    N_SAA = 1000
    samples_SAA, notUsed = solver.generate_data_and_weights(N_SAA, seed=simulation_seed)

    # Now we can create DRO parameters and solve problems using this sequence
    R = 5  # resource budget
    fixed_params = {
        0: {'a_i': 1.0, 'b_i': 2.0, 'c_i': 0.5, 'd_i': 0.8},
        1: {'a_i': 1.2, 'b_i': 1.5, 'c_i': 0.6, 'd_i': 0.7},
    }

    # Create DRO parameters object
    dro_params = DROParameters(x_dim, num_balls, R, fixed_params, 
                             epsilon=initial_epsilon,
                             ball_weights=weights,
                             ball_centres=solver.ball_centers)


    # Plot initial configuration
    solver.visualize_samples_and_covering(samples, online_samples=None)

    # Initialization (t = -1)
    x_prev = np.zeros(x_dim)
    y_prev = np.zeros((num_balls,x_dim)) # meaning we start from the initial nominal distribution
    weights_prev = weights
    rad_prev = radius_init

    # Main iteration loop
    for t in range(T):
        print(f"\nTimestep {t+1}/{T}")
        
        ############ONLINE UPDATES ################
        # solve online problem
        
        y_t, time_max = solver.solve_max_problem(
                x_prev,
                weights_prev,
                dro_params,
                rad_prev=rad_prev,
                max_iter_fw=1,
                y_prev=y_prev
            )
        tic = time.time()
        x_t = solver.solve_min_problem_gd(
                y_prev,
                weights_prev,
                dro_params,
                max_iter=1,
                x_prev=x_prev
            )
        toc = time.time()
        # evaluate online cost
        online_cost = solver.evaluate_online_cost(y_t, x_t, weights_prev, dro_params)
        time_step = (toc - tic) + time_max

        # Store solutions
        history['online_computation_times']['step_time'].append(time_step)
        history['online_obj_values'].append(online_cost)
        history['x_online'].append(x_t)
        tmp_SAA = solver.evaluate_expected_cost_SAA(x_t,samples_SAA,dro_params,N_SAA)
    
        # New sample arrives
        new_sample = solver.sample_from_mixture(size=1)
        new_sample = np.clip(new_sample, bounds[:,0], bounds[:,1])
        total_data = np.vstack([total_data, new_sample])  # Stack as numpy array
                
        N += 1  # total counter offline+online
        tic = time.time()
        weights, new_radius = solver.update_weights_with_new_sample(
            new_sample, weights, N, beta, C, diam
        )
        # Update the clustered set for later
        new_epsilon = new_radius + solver.radius  # note: new_radius is the current radius at time t (knowing that a new samples arrived)
        toc = time.time()
        time_update = toc - tic
        history['online_computation_times']['weight_update'].append(time_update)
        history['online_computation_times']['total_iteration'].append(time_update + time_step)
        dro_params.update_epsilon(new_epsilon)
        dro_params.update_ball_weights(weights)
        history['epsilon'].append(new_epsilon)
        history['weights'].append(weights)

        # Update solutions for next round
        weights_prev = weights
        x_prev = x_t
        y_prev = y_t
        rad_prev = new_epsilon

        # Solve DRO full with updated ambiguity set
        value_DRO, x_DRO, time_DRO = createproblem_resource_allocation(m, dro_params,R,N,total_data,new_radius,(1/N) * np.ones(N))

        tmp_DRO_eval_SAA = solver.evaluate_expected_cost_SAA(x_DRO,samples_SAA,dro_params,N_SAA)

        history['DRO_x'].append(x_DRO)  
        history['DRO_obj_values'].append(value_DRO)
        history['istantaneous_regret'].append(online_cost - value_DRO)
        history['istantaneous_regret_SAA'].append(tmp_SAA - tmp_DRO_eval_SAA)
        history['DRO_computation_times']['solver_time'].append(time_DRO)
      
        print(f"Current allocation: {x_t}")
        print(f"Current epsilon: {new_epsilon}")
        print(f"Weight sum: {np.sum(weights)}")
        print(f"Current perturbation: {y_t}")


    # After all iterations complete, create visualizations
    solver.visualize_samples_and_covering(samples, np.array(total_data))

    # After the loop, compute cumulative regret
    cumulative_regret = np.cumsum(history['istantaneous_regret'])
    history['cumulative_regret'] = cumulative_regret
    cumulative_regret_SAA = np.cumsum(history['istantaneous_regret_SAA'])
    history['cumulative_regret_SAA'] = cumulative_regret_SAA

    # Plot results with consistent styling
    solver.plot_results(history)

    # After all iterations complete
    solver.plot_regret_analysis(history)

    # Plot time
    solver.plot_computation_times(history)
    solver.plot_computation_times_iter(history,T)
 
    
    
