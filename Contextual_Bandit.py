from _util import *
from scipy.optimize import minimize
##!!! logY and log(budget)

class Contextual_Bandit_agent():
    @autoargs()
    def __init__(self, global_model = None, sigma_eps = 1, M = None, K = None, N = None,
                 prior_mean_powerlaw_coef = None, prior_cov_powerlaw_coef = None,
                 Xs = None, with_intercept = True, order = None, augment_size = 1000, B = None, true_gamma = None,
                 log_log = True, true_prior_mean_powerlaw_coef = None):
        
        self.seed = 42
        self.Phi = Xs # [M, K*(N+1), p]
        #transform the features matrix to include only the campaign feature/adline feature-->Phi = [M,K,dm+dk]
        self.preprocess_Phi()
        self._init_data_storate(K, M)
        self.sample_fail = 0

    def preprocess_Phi(self):
        """
        the original Phi is (M, K*(N+1), 1+dm+dk+1) including intercept and exact budget for each (m,k,n) arm. However, in this agent, we need only the campaign & adline features.
        """
        if self.with_intercept:
            #omit the intercept and the budget which is the in last column
            self.Phi = self.Phi[:,:,1:-1]
        else:
            #omit the budget which is the in last column
            self.Phi = self.Phi[:,:,:-1]
        # get the feature of each(m,k), which is the same across all N budget divisions, the final Phi is a M*K*(d_m+d_k) matrix
        self.Phi = self.Phi[:,list(range(0,self.K*(self.N+1), self.N+1))]
        self.p = self.Phi.shape[2]+1
        
    
    def _init_data_storate(self, K, M):
        """ initialize data storage and components required for computing the posterior
        """
        #initialize the records of Rewards to facilitate the future rearragment
        self.Y_each_task = [[np.zeros(0) for _ in range(K)] for _ in range(M)]
        self.A_each_task = [[np.zeros(0) for a in range(K)] for _ in range(M)]
        
        #initialize the records of all interactions to facilitate the global model training
        self.Phi_all = np.zeros((0,self.p))
        self.Y_all = np.zeros(0)
        self.A_all = np.zeros(0)

    def _init_posterior(self):
        #initialize a posteriors of coefficients for each (campaign, adline)
        #self.coef_post_mean = [[self.prior_mean_powerlaw_coef[m][k] for k in range(self.K)] for m in range(self.M)]
        self.coef_post_mean = [[self.prior_mean_powerlaw_coef for k in range(self.K)] for m in range(self.M)]
        self.coef_post_cov = [[self.prior_cov_powerlaw_coef for k in range(self.K)] for m in range(self.M)]
    ################################################################################################################################################
    ################################################################################################################################################
    def receive_reward(self, i, t, S, obs_Y, X):
        """update_data
        """
        x = X[S]
        # receive observation, update the feature matrix Phi, the reward vector R, and the records of the number of interactions
        for j in range(len(S)):
            if x[j][-1] > 0:
                this_R = obs_Y[j] #log(Y+1)
                self.Y_each_task[i][j] = np.append(self.Y_each_task[i][j], this_R) 
                self.A_each_task[i][j] =  np.append(self.A_each_task[i][j], x[j][-1]) 
                if self.with_intercept:
                    self.Phi_all = np.vstack([self.Phi_all, x[j][1:]])
                else:
                    self.Phi_all = np.vstack([self.Phi_all, x[j]])
                self.Y_all = np.append(self.Y_all, this_R)
                self.A_all = np.append(self.A_all, x[j][-1])

    def take_action(self, i, t, X = None):
        """
        In the concurrent setting, the gamma will be sampled every day after observing rewards from all campaigns that day.
        """
        np.random.seed(self.seed)
        self.seed += 1
        if t == 0 and i == 0:
            self._init_posterior()
        elif self.order == "concurrent" and i == 0 and self.Phi_all.shape[0]>0:
            #step1: update the global model
            self.fit_global()
            #step2: augment data for each adline
            self.data_augmentation()
            #step3: update the posterior of parameters for each adline
            self.update_concurrent_post()
        elif self.order == "sequential" and t == 0 and self.Phi_all.shape[0]>0:
            #step1: update the global model
            self.fit_global()
            #step2: augment data for each adline
            self.data_augmentation_i(i)
            #step3: update the posterior of parameters for each adline
            self.update_campaign_i_post(i)                
            
        # sampling coefs from corresponding posterior distribution
        coef_list1, coef_list2 = [], []
        for k in range(self.K):
            fail_sample = False
            sample_t = 1
            coef1, coef2 = np.random.multivariate_normal(self.coef_post_mean[i][k], self.coef_post_cov[i][k])
            #if i == 0:
                #print(i,k,(self.coef_post_mean[i][k]))#-self.true_prior_mean_powerlaw_coef[i][k]))
            #if self.log_log:
            #    while(coef1 < 0 or coef2 > 1 or coef2 <= 0):
            #        sample_t += 1
            #        if sample_t > 20:
            #            fail_sample = True
            #            coef1, coef2 = np.random.multivariate_normal(self.prior_mean_powerlaw_coef, self.prior_cov_powerlaw_coef)
            #        else:
            #            coef1, coef2 = np.random.multivariate_normal(self.coef_post_mean[i][k], self.coef_post_cov[i][k])
            #    if fail_sample:
            #        self.sample_fail += 1    
            coef_list1.append(coef1)
            coef_list2.append(coef2)
        # get the optimal budget allocation
        A = self._optimize(i, coef_list1, coef_list2, log_log = self.log_log)
                
        for j in range(self.K):
            A[j] = j*(self.N+1) + self.find_closest_point(self.N, A[j]/self.B[i])
        #print((A%(self.N+1))/self.N)
        return A.astype(int)
        
    def find_closest_point(self, N, ratio):
        points = list(np.array(range(0,N+1))/N)
        # Calculate the distance from each point to the ratio
        distances = [abs(point - ratio) for point in points]
        # Return the index of the closest point
        return distances.index(min(distances))



    def _optimize(self, i, coef_list1, coef_list2, log_log=True):
        # cvxpy
        #A = cp.Variable(self.K)
        #objective_terms = np.array([cp.power(A[i]+1, coef_list2[i]) for i in range(len(coef_list2))])
        #objective = cp.Minimize(- np.exp(coef_list1) @ objective_terms)
        #constraints = [cp.sum(A) == self.B[i], A>=0] #constraints: sum<=B, and A[i] non-negative
        #problem = cp.Problem(objective, constraints)
        #self.opt = problem.solve(solver=cp.ECOS)
        #print('cv:', A.value)
        optimizer = YourOptimizer(K=self.K, B= self.B[i], N = self.N, coef_list1=coef_list1, coef_list2=coef_list2, log_log = log_log)
        optimizer.optimize()
        A = optimizer.opt
        return  A # in the original budget scale

    ################################################################################################################################################
    ###################################################### receive_reward #################################################################
    ################################################################################################################################################
    def fit_global(self):
        self.global_model.fit(self.Phi_all, self.Y_all)
        #print(self.global_model.intercept_+np.sum(self.global_model.coef_))
        #print(self.Phi_all.shape, np.mean((self.true_gamma[1:]-self.global_model.coef_)**2),(self.true_gamma[0]-self.global_model.intercept_)**2)
    
    def update_concurrent_post(self):
        # step1: initialize the BLR for each adline and campaign
        # step2: update the posteriors of parameters for each adline and campaign
        self.coef_post_mean = [[[] for k in range(self.K)] for i in range(self.M)]
        self.coef_post_cov  = [[[] for k in range(self.K)] for i in range(self.M)]
        
        for i in range(self.M):
            for k in range(self.K):
                #BLR = BayesianLinearRegression(self.prior_mean_powerlaw_coef[i][k], self.prior_cov_powerlaw_coef, self.sigma_eps)
                BLR = BayesianLinearRegression(self.prior_mean_powerlaw_coef, self.prior_cov_powerlaw_coef, self.sigma_eps)
                BLR.update_posterior(features = np.array(self.A_augmented_each_task[i][k]), targets = np.array(self.Y_augmented_each_task[i][k]))
                self.coef_post_mean[i][k] = np.double(np.squeeze(BLR.post_mean))
                self.coef_post_cov[i][k] = np.double(BLR.post_cov) 
                
    def update_campaign_i_post(self, i):
        # step1: initialize the BLR for each adline and campaign
        # step2: update the posteriors of parameters for each adline and campaign
        self.coef_post_mean = [[[] for k in range(self.K)] for i in range(self.M)]
        self.coef_post_cov  = [[[] for k in range(self.K)] for i in range(self.M)]
        
        for k in range(self.K):
            BLR = BayesianLinearRegression(self.prior_mean_powerlaw_coef, self.prior_cov_powerlaw_coef, self.sigma_eps)
            BLR.update_posterior(features = np.array(self.A_augmented_each_task[i][k]), targets = np.array(self.Y_augmented_each_task[i][k]))
            self.coef_post_mean[i][k] = np.double(np.squeeze(BLR.post_mean))
            self.coef_post_cov[i][k] = np.double(BLR.post_cov) 
        
    def data_augmentation(self):
        """
        Augmenting the observed history for each adline with a series of predicted returns generated by the global model
        """
        #print(min(self.Phi_all[:,-1]), max(self.Phi_all[:,-1]))
        if self.log_log:
            A_grid = [np.arange(min(self.Phi_all[:,-1]), max(self.Phi_all[:,-1]) + .001, (max(self.Phi_all[:,-1])-min(self.Phi_all[:,-1]) + .001)/self.augment_size) for i in range(self.M)]
            #A_grid = [np.arange(.001, np.log(self.B[i]+1), (np.log(self.B[i]+1)-.001)/self.augment_size) for i in range(self.M)]
        else:
            A_grid = [np.arange(min(self.Phi_all[:,-1]), max(self.Phi_all[:,-1]) + .001, (max(self.Phi_all[:,-1])-min(self.Phi_all[:,-1]) + .001)/self.augment_size) for i in range(self.M)]
            #A_grid = [np.arange(.001, self.B[i], (self.B[i]-.001)/self.augment_size) for i in range(self.M)]
        #print(A_grid)
        self.A_augmented_each_task = [[self.A_each_task[i][k].tolist() + A_grid[i].tolist() for k in range(self.K)] for i in range(self.M)]
        self.Y_augmented_each_task = [[self.Y_each_task[i][k].tolist() + self.get_predictedR(self.Phi[i][k], A_grid[i]).tolist() for k in range(self.K)] for i in range(self.M)]
        #print(self.Y_augmented_each_task[0])
        
    def data_augmentation_i(self, i):
        """
        Augmenting the observed history for each adline with a series of predicted returns generated by the global model
        """
        if self.log_log:
            A_grid = np.arange(min(self.Phi_all[:,-1]), max(self.Phi_all[:,-1]) + .001, (max(self.Phi_all[:,-1])-min(self.Phi_all[:,-1]) + .001)/self.augment_size)
        else:
            A_grid = np.arange(min(self.Phi_all[:,-1]), max(self.Phi_all[:,-1]) + .001, (max(self.Phi_all[:,-1])-min(self.Phi_all[:,-1]) + .001)/self.augment_size)
            
        self.A_augmented_each_task = [[[] for k in range(self.K)] for i in range(self.M)]
        self.Y_augmented_each_task = [[[] for k in range(self.K)] for i in range(self.M)]
        self.A_augmented_each_task[i] = [self.A_each_task[i][k].tolist() + A_grid.tolist() for k in range(self.K)]
        self.Y_augmented_each_task[i] = [self.Y_each_task[i][k].tolist() + self.get_predictedR(self.Phi[i][k], A_grid).tolist() for k in range(self.K)]
        
        
    def get_predictedR(self, feature_mk, A_grid):
        """
        feature_mk is a (dm_dk,) feature vector
        """
        X = np.tile(feature_mk, (A_grid.shape[0], 1))
        X = np.hstack((X, A_grid.reshape(-1,1)))
        #print(self.global_model.predict(np.array([[1,1,1]])))
        return self.global_model.predict(X)
        #return X.dot(self.true_gamma[1:])+self.true_gamma[0]
        
    
    
class BayesianLinearRegression:
    """ Bayesian linear regression
    
    Args:
        prior_mean: Mean values of the prior distribution (m_0)
        prior_cov: Covariance matrix of the prior distribution (S_0)
        noise_var: Variance of the noise distribution
    """
    
    def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray, noise_var: float):
        self.prior_mean = prior_mean[:, np.newaxis] # column vector of shape (1, d)
        self.prior_cov = prior_cov # matrix of shape (d, d)
        # We initalize the prior distribution over the parameters using the given mean and covariance matrix
        # In the formulas above this corresponds to m_0 (prior_mean) and S_0 (prior_cov)
        self.prior = np.random.multivariate_normal(prior_mean, prior_cov)
        
        # We also know the variance of the noise
        self.noise_var = noise_var # single float value
        self.noise_precision = 1 / noise_var
        
        # Before performing any inference the parameter posterior equals the parameter prior
        self.param_posterior = self.prior
        # Accordingly, the posterior mean and covariance equal the prior mean and variance
        self.post_mean = self.prior_mean # corresponds to m_N in formulas
        self.post_cov = self.prior_cov # corresponds to S_N in formulas
        
    def update_posterior(self, features: np.ndarray, targets: np.ndarray):
        """
        Update the posterior distribution given new features and targets
        
        Args:
            features: numpy array of features
            targets: numpy array of targets
        """
        # Reshape targets to allow correct matrix multiplication
        # Input shape is (N,) but we need (N, 1)
        targets = targets[:, np.newaxis]
        
        # Compute the design matrix, shape (N, 2)
        design_matrix = self.compute_design_matrix(features)

        # Update the covariance matrix, shape (2, 2)
        design_matrix_dot_product = design_matrix.T.dot(design_matrix)
        inv_prior_cov = np.linalg.inv(self.prior_cov)
        self.post_cov = np.linalg.inv(inv_prior_cov +  self.noise_precision * design_matrix_dot_product)
        
        # Update the mean, shape (2, 1)
        self.post_mean = self.post_cov.dot( 
                         inv_prior_cov.dot(self.prior_mean) + 
                         self.noise_precision * design_matrix.T.dot(targets))

        
        # Update the posterior distribution
        self.param_posterior = np.random.multivariate_normal(self.post_mean.flatten(), self.post_cov)
                
    def compute_design_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the design matrix. To keep things simple we use simple linear
        regression and add the value phi_0 = 1 to our input data.
        
        Args:
            features: numpy array of features
        Returns:
            design_matrix: numpy array of transformed features
            
        >>> compute_design_matrix(np.array([2, 3]))
        np.array([[1., 2.], [1., 3.])
        """
        n_samples = len(features)
        phi_0 = np.ones(n_samples)
        design_matrix = np.stack((phi_0, features), axis=1)
        return design_matrix
    
    
# optimizer allowed nonconvex optimization
class YourOptimizer:
    def __init__(self, K, B, N, coef_list1, coef_list2, log_log = True):
        self.K = K
        self.B = B
        self.N = N
        self.coef_list1 = coef_list1
        self.coef_list2 = coef_list2
        self.log_log = log_log
        if log_log:
            # trasfer back to k1 in the original power-law form 
            self.coef_list1 = np.exp(self.coef_list1)

    def objective(self, A):
        if self.log_log:
            objective_terms = np.sum([(self.coef_list1[i] * np.power(A[i]+1, self.coef_list2[i])) if A[i] >= (self.B/self.N) else 0 
                                   for i in range(len(self.coef_list2))])
        else:
            objective_terms =  np.sum([(self.coef_list1[i] + A[i] * self.coef_list2[i]) if A[i] >= (self.B/self.N) else 0 
                                   for i in range(len(self.coef_list2))])
        return - objective_terms


    def constraint(self, A):
        return np.sum(A) - self.B

    def optimize(self):
        # Initial guess
        x0 = np.zeros(self.K)

        # Define constraints and bounds
        cons = {'type': 'eq', 'fun': self.constraint}
        bnds = [(0, None) for _ in range(self.K)]# Non-negativity constraints

        # Run the optimizer
        result = minimize(self.objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        self.opt = result.x
        return result
