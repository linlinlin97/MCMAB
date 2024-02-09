from _util import *
from _optimizer import *
import scipy.linalg
from scipy.linalg import cholesky, cho_solve
from sklearn.metrics.pairwise import rbf_kernel

class GP_agent():
    """ "Lec 9: linear bandits and TS"
        1. prior_theta = prior_theta
        2. magnitute of the errors is the marginalized one
    * already incremental
        3. assume all campaigns' contextual information is obtained & logged at the beginning, before the experiment starts
    """
    @autoargs()
    def __init__(self, M, K, N, Xs = None, prior_f_mu = None, kernel = "RBF", kernel_gamma = 1, 
                 sigma = 1, exp_episode = 2, cholesky_decomp = True, order = None, 
                 refresh_memory = False, refresh_threshold = 5000, log_log = False, approximate = True, 
                 B_max = None, with_intercept = False):
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        
        d: the dimension of covariates in X
        %Xall: [M,K*(N+1),d]-dimensional
        prior_f_mu: the prior mean of f function (Gaussian Process), which is a function of 
        prior_f_kernel: prior kernel of gaussian process 
        kernel_gamma: a parameter in gaussian process. One can set kernel_gamma = 1 by default
        
        Assume all ad campaigns share the same information? -- No, but M is not used in contextual info matrix X
        We calculate the posterior of each campaign independently -- assume covariance = 0 between campaigns
        and use the same kernel for all campaigns
        """
        self.M = M
        self.K = K
        self.N = N
        Xs = Xs.copy()
        if self.log_log:
            Xs[:,:,-1] /= np.log(self.B_max+1)
        else:
            Xs[:,:,-1] /= self.B_max
        if with_intercept:
            Xs = Xs[:,:,1:]
        self.d = d = np.shape(Xs)[2]
        self.Phi = Xs # [M, K*(N+1), p]
        self.Phi_flatten = np.vstack(Xs)
        self.items_tot = K * (N + 1)
        self.cnts = np.zeros((M, self.items_tot))
        
        temp = prior_f_mu
        def prior_f_mu_init(self, i, x, use = "0"): # incorporate (self, i) in argument list for later usage
            return temp(x)
        self.prior_f_mu = copy.copy(prior_f_mu_init)
        
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        self.seed = 42
        self.timer = 0
        self.j = None
        self.H_all = np.array([], dtype=np.int64).reshape(0,d) # updated once receiving a new sample
        self.H_all_batch = np.array([], dtype=np.int64).reshape(0,d) # updated by batch
        self.R_all = np.array([], dtype=np.int64) # updated once receiving a new sample
        self.R_all_batch = np.array([], dtype=np.int64) # updated by batch
        self.Z_all = np.array([], dtype=np.int64).reshape(self.M * self.items_tot, 0)
        self.Z_all_batch = np.array([], dtype=np.int64).reshape(self.M * self.items_tot, 0)
        self.cholesky_decomp = cholesky_decomp
        self.order = order
        self.refresh_memory = refresh_memory
        self.refresh_threshold = refresh_threshold
        self.memory_clear_times = 0
        self.approximate = approximate
        self._init_posterior()
        self._init_memory_check()
        self.updated = False
        
        
    def _init_posterior(self):
      
        if (self.kernel == "RBF"):
            def prior_f_kernel_RBF(self, i, j, x, x_prime = [], use = "0"):
                if len(x_prime) == 0:
                    return rbf_kernel(X = x, Y = None, gamma = self.kernel_gamma)
                else:
                    return rbf_kernel(X = x, Y = x_prime, gamma = self.kernel_gamma) 
                    # note: x needs to be [items, d] dimensional
            self.prior_f_kernel = prior_f_kernel_RBF
        elif (self.prior_f_kernel == "linear"):
            def prior_f_kernel_linear(self, i, j, x, x_prime = [], use = "0"):
                if len(x_prime) == 0:
                    x_prime = x
                return x.dot(x_prime.T)
            self.prior_f_kernel = prior_f_kernel_linear
        else:
            print("Error: kernel name not found.")
       
        self.post_f_mu = copy.copy(self.prior_f_mu) # [items_tot = K * (N + 1)]-dimensional
        self.post_f_kernel = copy.copy(self.prior_f_kernel)

    def _init_memory_check(self):

        # store the most up-to-date prior after each round of memory check
        self.f_MKN_prior_mu = self.prior_f_mu(self, 0, self.Phi_flatten)    
        self.f_MKN_prior_kernel = self.prior_f_kernel(self, 0, 0, self.Phi_flatten) 


    def receive_reward(self, i, t, A, obs_R, X):
        X = X.copy()
        if self.log_log:
            X[:,-1] /= np.log(self.B_max+1)
        else:
            X[:,-1] /= self.B_max
        if self.with_intercept:
            X = X[:, 1:]
        current_time = now()

        # update_data. update posteriors
        X_updated = X[A]
        # obs_R is K-dimensional
        
        # rows_w_budget = np.where(X_updated[:, -1] != 0)[0]
        rows_w_budget = np.linspace(0,(self.K-1), self.K).astype(int)
        for k in range(self.K):
            if A[k] == (k*self.N + k):
                rows_w_budget[k] = -1
        rows_w_budget = np.delete(rows_w_budget, np.where(rows_w_budget == -1))
        
        
        X_updated = X_updated[rows_w_budget]
        obs_R = obs_R[rows_w_budget]
        A = A[rows_w_budget]
        
        self.updated_num = len(rows_w_budget) # the real number of ad lines that are updated (after deleting zero proportions)

        
        self.R_all = np.concatenate((self.R_all, obs_R))
        self.H_all = np.vstack((self.H_all, X_updated))
        self.items_updating = np.shape(self.H_all)[0]
        
        self.Z_all = np.hstack((self.Z_all, np.zeros((self.M * self.items_tot, self.updated_num))))
        for k in range(self.updated_num):
            self.Z_all[i*self.items_tot + A[k], self.items_updating-self.updated_num + k] = 1
            
        

        # self.update_f()
        
        self.cnts[i, A] += 1
        self.timer += now() - current_time
        
        
    def take_action(self, i, t, X):
        """
        X = [items_tot, p]
        """ 
        X = X.copy()
        if self.log_log:
            X[:,-1] /= np.log(self.B_max+1)
        else:
            X[:,-1] /= self.B_max
        if self.with_intercept:
            X = X[:, 1:]
        np.random.seed(self.seed)
        self.seed += 1
        
        if self.approximate:
            if self.order == "concurrent":
                if i == 0 and t != 0:
                    self.H_all_batch = self.H_all.copy()
                    self.R_all_batch = self.R_all.copy()
                    self.Z_all_batch = self.Z_all.copy()
                    self.update_f(i) 
                    if self.refresh_memory == True and np.sum(self.cnts)>self.refresh_threshold:
                        self.clear_memory()
            elif self.order == "sequential":
                if t == 0 and i != 0:
                    self.H_all_batch = self.H_all.copy()
                    self.R_all_batch = self.R_all.copy()
                    self.Z_all_batch = self.Z_all.copy()
                    self.update_f(i) 
                    if self.refresh_memory == True and np.sum(self.cnts)>self.refresh_threshold:
                        self.clear_memory()
        else:
            if np.sum(self.cnts) != 0:
                self.H_all_batch = self.H_all.copy()
                self.R_all_batch = self.R_all.copy()
                self.Z_all_batch = self.Z_all.copy()
                self.update_f(i) 
                if self.refresh_memory == True and np.sum(self.cnts)>self.refresh_threshold:
                    self.clear_memory()
        
        #if self.order == "concurrent":
        #    if i % self.M == 0 and t != 0:
        #        self.H_all_batch = self.H_all.copy()
        #        self.R_all_batch = self.R_all.copy()
        #        self.Z_all_batch = self.Z_all.copy()
        #        self.update_f(i) 
        #else:
        #    if np.sum(self.cnts) != 0:
        #        self.update_f(i)
     
        if len(self.R_all_batch) == 0: # this may happen when (1)t=0 and i=0, (2)right after clearing historical data
            cov = self.prior_f_kernel(self, i, self.j, X)
            self.sampled_f = np.random.multivariate_normal(self.prior_f_mu(self, i, X), cov + 1e-10 * np.eye(cov.shape[0])) 
        else:
            cov = self.post_f_kernel(self, i, self.j, X)                                               
            self.sampled_f = np.random.multivariate_normal(self.post_f_mu(self, i, X), cov + 1e-10 * np.eye(cov.shape[0])) 
        # sample theta (or f) from posterior
        
        Rs = self.sampled_f
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)
        
        return A
    
    def clear_memory(self):
        ######### NEW memory check: clear everything if the number of historical data exceed refresh threshold######
        self.memory_clear_times += 1
        print("GP-FD: clear memory for ", self.memory_clear_times, " times")
            
        # step 1: store historical data to memory for the update of prior in step 2
        self.f_MKN_prior_mu_previous = self.f_MKN_prior_mu.copy()
        self.f_MKN_prior_kernel_previous = self.f_MKN_prior_kernel.copy()
              
        # step 2: update the new prior according to previous data
        temp1 = self.f_MKN_prior_kernel_previous.dot(self.Z_all_batch)
        temp2 = self.f_inv_mid.dot(self.R_all_batch-self.Z_all_batch.T.dot(self.f_MKN_prior_mu_previous))

        self.f_MKN_prior_mu = self.f_MKN_prior_mu_previous + temp1.dot(temp2)
        self.f_MKN_prior_kernel = self.f_MKN_prior_kernel_previous - temp1.dot(self.f_inv_mid).dot(temp1.T)
            
        def prior_f_mu_func(self, i, x, use = "i"):
            if use == "Z_all":
                return self.f_MKN_prior_mu.dot(self.Z_all_batch)
            else:
                return self.f_MKN_prior_mu[i*self.items_tot:(i+1)*self.items_tot]
        def prior_f_kernel_func(self, i, j, x, x_prime = [], use = "1"):
            if j == None:
                j = i
            if use == "1":
                return self.f_MKN_prior_kernel[i*self.items_tot:(i+1)*self.items_tot, j*self.items_tot:(j+1)*self.items_tot]
            if use == "2":
                return self.f_MKN_prior_kernel[i*self.items_tot:(i+1)*self.items_tot, :].dot(self.Z_all_batch)
            if use == "3":
                return self.Z_all_batch.T.dot(self.f_MKN_prior_kernel[:,j*self.items_tot:(j+1)*self.items_tot])
            if use == "4":
                return self.Z_all_batch.T.dot(self.f_MKN_prior_kernel).dot(self.Z_all_batch)
                
                
            return self.f_MKN_prior_kernel[i*self.items_tot:(i+1)*self.items_tot, i*self.items_tot:(i+1)*self.items_tot]
            
        self.prior_f_mu = copy.copy(prior_f_mu_func)
        self.prior_f_kernel = copy.copy(prior_f_kernel_func)
            
            
        # step 3: abandon all historical data after updating prior            
        self.H_all = np.array([], dtype=np.int64).reshape(0,self.d)
        self.R_all = np.array([], dtype=np.int64)
        self.H_all_batch = np.array([], dtype=np.int64).reshape(0,self.d)
        self.R_all_batch = np.array([], dtype=np.int64)
        self.Z_all = np.array([], dtype=np.int64).reshape(self.M * self.items_tot, 0)
        self.Z_all_batch = np.array([], dtype=np.int64).reshape(self.M * self.items_tot, 0)
            
        self.cnts = np.zeros((self.M, self.K*(self.N+1)))


    def update_f(self, i):
        
        if (self.cholesky_decomp == True):
            A = self.prior_f_kernel(self, i, 0, self.H_all_batch, [], "4") + self.sigma**2 * np.identity(self.items_updating)
            L = cholesky(A, lower=True, check_finite=False)
            self.f_inv_mid = cho_solve((L, True), np.identity(A.shape[0]), check_finite=False)
        else:
            self.f_inv_mid = inv(self.prior_f_kernel(self, i, 0, self.H_all_batch, [], "4") + self.sigma**2 * np.identity(self.items_updating))
        
        # calculate the posterior mean of gaussian process f
        def post_f_mu_updating(self, i, x):
            return self.prior_f_mu(self, i, x, "i") + self.prior_f_kernel(self, i, 0, x, self.H_all_batch, "2").dot(self.f_inv_mid).dot(self.R_all_batch - self.prior_f_mu(self, i, self.H_all_batch, "Z_all"))
        
        # calculate the posterior kernel of gaussian process f
        def post_f_kernel_updating(self, i, j, x, x_prime = []):
            if len(x_prime) == 0:
                x_prime = x
            if j == None:
                j = i
            return self.prior_f_kernel(self, i, j, x, x_prime, "1") -self.prior_f_kernel(self, i, 0, x, self.H_all_batch, "2").dot(self.f_inv_mid).dot(self.prior_f_kernel(self, 0, j, self.H_all_batch, x_prime, "3"))
        
        # update posterior mean and kernel
        self.post_f_mu = copy.copy(post_f_mu_updating)
        self.post_f_kernel = copy.copy(post_f_kernel_updating)
    
    def _optimize(self, Rs, t):
        """ 
        the dynamic programming to return the optimal subset of (m,k) tuples selected.
        """
        #the reward should be non-negative
        Rs[Rs < 0] = 0
        #the reward for budget = 0 should be 0
        Rs[list(range(0,self.K*(self.N+1), self.N+1))] = 0
        A = knapsack_optimizer(Rs, self.K, self.N)
        
        if len(A) == 0:
            A = np.zeros(self.K)
            for i in range(self.K):
                if t < self.exp_episode:
                    A[i] = i*(self.N+1) + self.N//self.K # evenly assign the budget
                else:
                    A[i] = i*(self.N+1) # assign 0 to all
        return A.astype(int)
