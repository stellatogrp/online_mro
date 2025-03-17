import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from scipy.special import gamma
from sklearn.cluster import KMeans
import itertools
import time
import mosek


def createproblem_portMIP(N, m):
    """Create the problem in cvxpy, minimize CVaR
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    z = cp.Variable(m, boolean=True)
    tau = cp.Variable()
    # OBJECTIVE #
    objective = tau + eps*lam + w@s
    # + cp.quad_over_lin(a*x, 4*lam)
    # CONSTRAINTS #
    constraints = []
    constraints += [a*tau + a*dat@x <= s]
    constraints += [s >= 0]
    constraints += [cp.norm(a*x, 2) <= lam]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    constraints += [lam >= 0]
    constraints += [x - z <= 0, cp.sum(z) <= 10]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, lam, dat, eps, w

class DROParameters:
    """Class to handle DRO problem parameters"""
    def __init__(self, x_dim, num_ball,  epsilon=None, ball_weights=None, ball_centres=None):
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
        self.epsilon = epsilon # radius of ambiguity set
        self.num_balls = num_ball # number of balls
        self.ball_weights = ball_weights # weights for each ball
        self.ball_centres = ball_centres # centres for each ball
        # Store fixed parameters
 
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
        min_dist = np.min(distances, axis = 1)
        
        # Assign each sample to nearest ball center
        ball_assignments = np.argmin(distances, axis=1)
        
        # Compute weights as proportion of samples in each ball
        weights = np.zeros(self.num_balls)
        for i in range(self.num_balls):
            weights[i] = np.sum(ball_assignments == i) / N
            
        return samples, weights, min_dist

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
        min_dist = np.min(distances)
        assigned_ball = np.argmin(distances)
        
        # Update weights
        new_weights = current_weights.copy()
        new_weights = new_weights * (N - 1) / N  # Scale down old counts
        new_weights[assigned_ball] += 1/N  # Add new sample
        
        # Update radius based on new N
        new_radius = diam * (C/N + np.sqrt((2*np.log(1/beta)))/np.sqrt(N))
        
        return new_weights, new_radius, min_dist



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
        
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        
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
        
        plt.xlabel(r'$u_1$')
        plt.ylabel(r'$u_2$')
        plt.title(r'Sample Distribution and Ball Covering')
        plt.grid(True, alpha=0.3)
        plt.legend(framealpha=0.9)
        
        # Save high-quality figures
        plt.savefig('covering_visualization.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('covering_visualization.png', bbox_inches='tight', dpi=300)
        
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


    def compute_cumulative_regret(self, history, online_samples, num_eval_samples=1000, seed=1111):
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

        def evaluate_expected_cost(d_eval, x, tau):
            return np.mean(
                np.maximum(-5*d_eval@x - 4*tau, tau)) 
        
        T = len(online_samples)  # Number of timesteps
        regret = np.zeros(T)  # Instantaneous regret at each timestep
        MRO_regret = np.zeros(T)
        cumulative_regret = np.zeros(T)  # Cumulative regret up to each timestep
        MRO_cumulative_regret = np.zeros(T) 
        theoretical = np.zeros(T)

        eval_values = np.zeros(T)
        MRO_eval_values = np.zeros(T)
        DRO_eval_values = np.zeros(T)
        
        # Generate evaluation samples from true distribution for cost computation
        np.random.seed(seed)
        eval_samples = self.sample_from_mixture(size=num_eval_samples)
        eval_samples = np.clip(eval_samples, self.bounds[:,0], self.bounds[:,1])
        
        # For each timestep t
        for t in range(T):            
            # Compute instantaneous regret at time t using true distribution
            online_cost = evaluate_expected_cost(eval_samples, history['x'][t],history['tau'][t])
            MRO_cost = evaluate_expected_cost(eval_samples, history['MRO_x'][t],history['MRO_tau'][t])
            optimal_cost = evaluate_expected_cost(eval_samples, history['DRO_x'][t],history['DRO_tau'][t])
            regret[t] = history['worst_values'][t] - history['DRO_obj_values'][t]
            eval_values[t] = online_cost
            MRO_eval_values[t] = MRO_cost
            DRO_eval_values[t] = optimal_cost
            
            # Update cumulative regret
            if t == 0:
                cumulative_regret[t] = regret[t]
                theoretical[t] = 1/np.sqrt(10)
            else:
                theoretical[t] = theoretical[t-1] + 1/np.sqrt(t+10)
                cumulative_regret[t] = cumulative_regret[t-1] + regret[t]
        
        return cumulative_regret, regret, eval_values, MRO_eval_values, DRO_eval_values, theoretical

    def plot_regret_analysis(self, cumulative_regret, regret, theo):
        """Plot regret analysis results with LaTeX formatting and log scales."""
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 22
        })
        
        # Create figure with 2x2 subplots
    
        T = len(cumulative_regret)
        t_range = np.arange(1, T+1)
        plt.figure(figsize=(9, 4), dpi=300)
        plt.semilogy(t_range, cumulative_regret, 'b-', linewidth=2, label = "online weight update")
        # plt.semilogy(t_range, MRO_cumulative_regret, 'r-', linewidth=2, label = "online clustering")
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Cumulative Regret')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('regret_analysis_cumulative.pdf', bbox_inches='tight', dpi=300)

        plt.figure(figsize=(9, 4), dpi=300)
        plt.semilogy(t_range, regret, 'b-', linewidth=2, label = "online weight update")
        # plt.semilogy(t_range, MRO_cumulative_regret, 'r-', linewidth=2, label = "online clustering")
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Instantaneous Regret')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('regret_analysis_inst.pdf', bbox_inches='tight', dpi=300)

        fig, ax = plt.subplots(1,1, figsize=(9, 4), dpi=300)
        ax.semilogy(t_range, cumulative_regret, 'b-', linewidth=2, label = "actual cumulative regret")
        ax.semilogy(t_range, theo, 'r-', linewidth=2, label = "theoretical regret")
        axins = zoomed_inset_axes(ax, 6, loc="lower right")
        axins.set_xlim(3700, 4000)
        axins.set_ylim(7, 10)
        axins.plot(t_range, cumulative_regret, 'b-',linewidth=2)
        axins.set_xticks(ticks=[])
        axins.set_yticks(ticks=[])
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
        ax.set_xlabel(r'Time step $(t)$')
        ax.set_ylabel(r'Cumulative Regret')
        # ax.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('regret_analysis_comp.pdf', bbox_inches='tight', dpi=300)


    def plot_eval(self,eval, MRO_eval, DRO_eval):
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 22
        })
        T = len(eval)
        t_range = np.arange(1, T+1)
        plt.figure(figsize=(7, 4), dpi=300)
        plt.plot(t_range, eval, 'b-', linewidth=2, label = "online weight update")
        plt.plot(t_range, MRO_eval, 'r-', linewidth=2, label = "online clustering")
        plt.plot(t_range, DRO_eval, color ='black', linewidth=2, label = "DRO")
        plt.legend()
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Evaluation value (out of sample)')
        plt.grid(True, alpha=0.3)
        plt.savefig('eval_analysis.pdf', bbox_inches='tight', dpi=300)


    def plot_results(self, history):
        """Plot results with LaTeX formatting."""
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 22
        })
        
        # Create figure with higher DPI
        plt.figure(figsize=(11, 4), dpi=300)

        # Plot 2: Epsilon Evolution
        plt.subplot(121)
        plt.plot(history['epsilon'], 'r-', linewidth=2)
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'$\epsilon$')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Ball Weights
        plt.subplot(122)
        plt.plot(np.array(history['weights']), linewidth=2)
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Ball Weights')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout(pad=2.0)
        plt.savefig('radius.pdf', bbox_inches='tight', dpi=300)


    def plot_computation_times(self, history):
        """Plot computation time analysis with LaTeX formatting."""
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 22
        })
        
        # Create figure
        plt.figure(figsize=(8, 3), dpi=300)
        
        # Prepare data for boxplot
        data = [
            history['online_computation_times']['total_iteration'], history['MRO_computation_times']['total_iteration'],history['DRO_computation_times']['total_iteration'] 
        ]
        np.save("online",history['online_computation_times']['total_iteration'])
        np.save("mro",history['MRO_computation_times']['total_iteration'])
        np.save("dro",history['DRO_computation_times']['total_iteration'])

        # Create boxplot
        bp = plt.boxplot(data, labels=[

            r'weight update', r'reclustering', r'DRO' 
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
        plt.savefig('time.pdf', bbox_inches='tight', dpi=300)

    def plot_computation_times_iter(self,history):
        # Set up LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 22
        })
        T = len(eval)
        t_range = np.arange(1, T+1)
        plt.figure(figsize=(9, 4), dpi=300)
        plt.plot(t_range, history['online_computation_times']['total_iteration'], 'b-', linewidth=2, label = "online weight update")
        plt.plot(t_range, history['MRO_computation_times']['total_iteration'], 'r-', linewidth=2, label = "online clustering")
        plt.plot(t_range, history['DRO_computation_times']['total_iteration'], color ='black', linewidth=2, label = "DRO")
        plt.legend()
        plt.xlabel(r'Time step $(t)$')
        plt.ylabel(r'Compuation time')
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.savefig('time_iters.pdf', bbox_inches='tight', dpi=300)


# Example usage
if __name__ == "__main__":
    # Algorithm selection flag
    use_exact_solver = True  # Set to False to use gradient-based methods

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
    T = 100  # Number of timesteps

    # Initial sample size and DRO parameters
    N = 10  # initial number of samples
    diam = 2*np.sqrt(x_dim)
    C = 1/np.sqrt(2)*3
    beta = 0.1

    # Generate initial samples and weights with seed
    samples, weights, min_dist = solver.generate_data_and_weights(N, seed=simulation_seed)
    min_dist = list(min_dist)

    # Initial radius computation
    radius_init = diam*(C/N + np.sqrt((2*np.log(1/beta)))*1/np.sqrt(N))
    initial_epsilon = radius_init + solver.radius  # Use solver.radius instead of ball_radius
    new_radius = radius_init # radius for DRO

    # Create DRO parameters object
    dro_params = DROParameters(x_dim, num_balls,
                            epsilon=initial_epsilon,
                            ball_weights=weights,
                            ball_centres=solver.ball_centers)

    # Create CVXPY problems 
    online_problem, online_x, online_s, online_tau, online_lmbda, data_train, eps_train, w_train = createproblem_portMIP(num_balls, x_dim)

    # Initialize solutions
    x_current = np.zeros(x_dim)

    # History for analysis
    history = {
        'x': [],
        'tau': [],
        'obj_values': [],
        'MRO_x': [],
        'MRO_tau': [],
        'MRO_obj_values': [],
        'DRO_x': [],
        'DRO_tau': [],
        'DRO_obj_values': [],
        'worst_values': [],
        'epsilon': [],
        'weights': [],
        'online_computation_times': {
            'weight_update': [],
            'min_problem': [],
            'total_iteration': []
        },
        'MRO_computation_times':{
        'clustering': [],
        'min_problem': [],
        'total_iteration':[]
        },
        'DRO_computation_times':{
        'total_iteration':[]
        },
        'distances':[]
    }

    # Plot initial configuration
    solver.visualize_samples_and_covering(samples, online_samples=None)

    # Main iteration loop
    online_samples = []  # Initialize as list
    running_samples = samples.copy()


    for t in range(T):
        print(f"\nTimestep {t+1}/{T}")
        
        # solve online MRO problem
        data_train.value = dro_params.ball_centres
        eps_train.value = new_radius
        w_train.value = dro_params.ball_weights
        online_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
            mosek.dparam.optimizer_max_time:  1500.0})
        x_current = online_x.value
        tau_current = online_tau.value
        min_obj = online_problem.objective.value
        min_time = online_problem.solver_stats.solve_time

        # Store timing information
        history['online_computation_times']['min_problem'].append(min_time)

        # solve MRO problem with new clusters
        start_time = time.time()
        kmeans = KMeans(n_clusters=num_balls).fit(running_samples)
        new_centers = kmeans.cluster_centers_
        wk = np.bincount(kmeans.labels_) / N
        cluster_time = time.time()-start_time
        history['MRO_computation_times']['clustering'].append(cluster_time)

        data_train.value = new_centers
        w_train.value = wk
        # eps_train.value = new_radius
        online_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
            mosek.dparam.optimizer_max_time:  1500.0})
        MRO_x_current = online_x.value
        MRO_tau_current = online_tau.value
        MRO_min_obj = online_problem.objective.value
        MRO_min_time = online_problem.solver_stats.solve_time
        
        history['MRO_computation_times']['min_problem'].append(MRO_min_time)
        history['MRO_computation_times']['total_iteration'].append(MRO_min_time+cluster_time)

        # solve DRO problem 
        DRO_problem, DRO_x, DRO_s, DRO_tau, DRO_lmbda, DRO_data, DRO_eps, DRO_w = createproblem_portMIP(N, x_dim)
        DRO_data.value = running_samples
        DRO_w.value = (1/N)*np.ones(N)
        DRO_eps.value = new_radius
        DRO_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
            mosek.dparam.optimizer_max_time:  1500.0})
        DRO_x_current = DRO_x.value
        DRO_tau_current = DRO_tau.value
        DRO_min_obj = DRO_problem.objective.value
        DRO_min_time = DRO_problem.solver_stats.solve_time
        history['DRO_computation_times']['total_iteration'].append(DRO_min_time)

        # compute online MRO worst value (wrt non clustered data)
        orig_cons = DRO_problem.constraints
        orig_obj = DRO_problem.objective
        new_cons = orig_cons + [DRO_x == x_current, DRO_tau == tau_current]
        new_problem = cp.Problem(orig_obj, new_cons)
        new_problem.solve(ignore_dpp=True, solver=cp.MOSEK, verbose=False, mosek_params={
            mosek.dparam.optimizer_max_time:  1500.0})
        new_worst = new_problem.objective.value
        history['worst_values'].append(new_worst)
            
        # New sample
        new_sample = solver.sample_from_mixture(size=1)
        new_sample = np.clip(new_sample, bounds[:,0], bounds[:,1])
        online_samples.append(new_sample.flatten())  # Append to list
        running_samples = np.vstack([running_samples,new_sample.flatten()])
        
        N += 1
        start_time = time.time()
        weights, new_radius, new_min = solver.update_weights_with_new_sample(
            new_sample, weights, N, beta, C, diam
        )
        min_dist.append(new_min)
        weight_update_time = time.time()-start_time
        history['online_computation_times']['weight_update'].append(weight_update_time)
        history['online_computation_times']['total_iteration'].append(weight_update_time + min_time)

        new_epsilon = new_radius + solver.radius
        dro_params.update_epsilon(new_epsilon)
        dro_params.update_ball_weights(weights)
        
        
        history['x'].append(x_current)
        history['tau'].append(tau_current)
        history['obj_values'].append(min_obj)
        history['MRO_x'].append(MRO_x_current)
        history['MRO_tau'].append(MRO_tau_current)
        history['MRO_obj_values'].append(MRO_min_obj)
        history['DRO_x'].append(DRO_x_current)
        history['DRO_tau'].append(DRO_tau_current)
        history['DRO_obj_values'].append(DRO_min_obj)
        history['epsilon'].append(new_epsilon)
        history['weights'].append(weights.copy())
        
        print(f"Current allocation: {x_current}")
        print(f"Current epsilon: {new_epsilon}")
        print(f"Weight sum: {np.sum(weights)}")

    min_dist.append(solver.radius)
    history["distances"].append(min_dist)
    np.save("min_dist1", min_dist)
    np.save('recluster_obj1',history['obj_values'] )
    np.save('dro_obj1', history['DRO_obj_values'])
    np.save('recluster_worst1',history['worst_values'] )

    # After all iterations complete, create visualizations
    solver.visualize_samples_and_covering(samples, np.array(online_samples))

    # Plot results with consistent styling
    solver.plot_results(history)

    # Compute and plot regret analysis
    cumulative_regret, instantaneous_regret, eval, MRO_eval, DRO_eval, theo = solver.compute_cumulative_regret(
        history, 
        np.array(online_samples),
        seed=simulation_seed
    )

    # Plot regret analysis
    solver.plot_regret_analysis(
        cumulative_regret, 
        instantaneous_regret,theo
    )
    np.save("regret2", cumulative_regret)

    # After all other plots
    solver.plot_computation_times(history)

    solver.plot_eval(eval, MRO_eval, DRO_eval)

    solver.plot_computation_times_iter(history)

    