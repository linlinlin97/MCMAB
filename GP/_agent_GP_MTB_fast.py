from _util import *
from _optimizer import *
import scipy.linalg
from scipy.linalg import cholesky, cho_solve
from sklearn.metrics.pairwise import rbf_kernel
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class GP_MTB_agent():
    @autoargs()
    def __init__(self, sigma_2 = 1, prior_f_mu = None, kernel = "RBF", kernel_gamma = 1,
                 sigma_1 = None, M = None, K = None, N = None, Xs = None, 
                 cholesky_decomp = True, order = None, exp_episode = 2, 
                 refresh_memory = False, refresh_threshold = 5000, log_log = False,  
                 approximate = True, B_max = None, with_intercept = False):
        
        self.seed = 42
        self.M = M
        self.K = K
        self.N = N
        self.items_tot = K*(N+1)
        #self.p = p = len(gamma_prior_mean)
        self.delta_cov = sigma_1**2*identity(K*(N+1))
        Xs = Xs.copy()
        if self.log_log:
            Xs[:,:,-1] /= np.log(self.B_max+1)
        else:
            Xs[:,:,-1] /= self.B_max
        if with_intercept:
            Xs = Xs[:,:,1:]
        self.Phi = Xs # [M, K*(N+1), p]
        self.Phi_flatten = np.vstack(Xs)
        self.p = np.shape(Xs)[2]
        self.approximate = approximate
        self.cnts = np.zeros((M, K*(N+1)))
        self.timer = 0
        
        # set prior mean and variance for f
        temp = prior_f_mu
        def prior_f_mu_init(self, i, x, use = "0"): # incorporate (self, i) in argument list for later usage
            return temp(x)
        self.prior_f_mu = copy.copy(prior_f_mu_init)
        
        # self.prior_f_mu = prior_f_mu
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        self._init_prior_var()

        # setup the initial data for storage
        self._init_data_storate(K*(N+1), M)        
        self._init_memory_check()
        
        self.cholesky_decomp = cholesky_decomp
        #self.gamma_prior_cov_inv = inv(gamma_prior_cov)
        
        self.j = None
        self.refresh_memory = refresh_memory
        self.refresh_threshold = refresh_threshold
        self.memory_clear_times = 0 
        #initialize the records of the number of interactions with each item to facilitate the future rearragment
        self.Sigma_idx = np.array(self._init_empty("num"))
        
    def _init_data_storate(self, tot_items_m, M):
        """ initialize data storage and components required for computing the posterior
        """
        #initialize the records of Rewards to facilitate the future rearragment
        self.R_each_task = [[np.zeros(0) for a in range(tot_items_m)] for _ in range(M)] # [M,tot_items_m,0]
        
        #initialize the records of features to facilitate the future rearragment
        self.Phi_obs = self._init_empty("row", self.p) # [M,K*(N+1),0,p]
        self.Phi_obs_i = [np.zeros((0, self.p)) for i in range(M)] # [M,0,p]
        
    def _init_memory_check(self):

        # store the most up-to-date prior after each round of memory check
        self.f_MKN_prior_mu = self.prior_f_mu(self, 0, self.Phi_flatten)    
        self.f_MKN_prior_kernel = self.prior_f_kernel(self, 0, 0, self.Phi_flatten) 

        
    def _init_prior_var(self):   
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

        
    def _init_posterior(self, tot_items_m, M):
        # step1: initialize a gamma, sampling from the prior distribution
        # self.sampled_gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
        
        #step2: initialize the priors of thetas given the sampled gamma
        self.theta_mean_prior = [np.random.multivariate_normal(self.prior_f_mu(self, i, self.Phi[i]), self.prior_f_kernel(self, i, i, self.Phi[i])) for i in range(M)]
        #self.theta_mean_prior = [self.Phi[i].dot(self.sampled_gamma) for i in range(M)]
        self.theta_cov_prior = [self.delta_cov for i in range(M)]
        self.post_theta_num_wo_prior = [zeros(tot_items_m) for i in range(M)] # assign space for posterior of theta
        self.post_theta_den_wo_prior = [zeros(tot_items_m) for i in range(M)]
        
        #step3: calculate the corresponding posteriors of thetas
        self.theta_num_post0 = [self.theta_mean_prior[i]/np.diag(self.delta_cov) for i in range(M)]
        self.theta_den_post0 = [1 / np.diag(self.delta_cov) for i in range(M)]
        self.theta_num_post = [self.theta_num_post0[i] + self.post_theta_num_wo_prior[i] for i in range(M)]
        self.theta_den_post = [self.theta_den_post0[i] + self.post_theta_den_wo_prior[i] for i in range(M)]
        self.theta_mean_post = [self.theta_num_post[i]/self.theta_den_post[i] for i in range(M)]
        self.theta_cov_post = [1/self.theta_den_post[i] for i in range(M)]
        
        # to save storage so as to save time
        self.inv = {}
        

    ################################################################################################################################################
    ################################################################################################################################################
    def receive_reward(self, i, t, S, obs_R, X = None):
        """update_data
        """
        X = X.copy()
        if self.log_log:
            X[:,-1] /= np.log(self.B_max+1)
        else:
            X[:,-1] /= self.B_max
        if self.with_intercept:
            X = X[:, 1:]
        x = X[S] # S: the K-dimensional action vector [0,101,...]
        
        
        # rows_w_budget = np.where(x[:, -1] != 0)[0]
        rows_w_budget = np.linspace(0,(self.K-1), self.K).astype(int)
        for k in range(self.K):
            if S[k] == (k*self.N + k):
                rows_w_budget[k] = -1
        rows_w_budget = np.delete(rows_w_budget, np.where(rows_w_budget == -1))
        
        x = x[rows_w_budget]
        obs_R = obs_R[rows_w_budget]
        S = S[rows_w_budget]
        
        # receive observation, update the feature matrix Phi, the reward vector R, and the records of the number of interactions
        self.cnts[i,S] += 1
        self.update_Phi(i, S, x)
        self.update_R_each_task(i, obs_R, S)
        self.Sigma_idx[i][S] += 1
        
        # update the posteriors of the arms played
        self.post_theta_num_wo_prior[i][S] += (obs_R / self.sigma_2 ** 2)
        self.post_theta_den_wo_prior[i][S] += (1 / self.sigma_2 ** 2)
        self.theta_num_post[i][S] = self.theta_num_post0[i][S] + self.post_theta_num_wo_prior[i][S]
        self.theta_den_post[i][S] = self.theta_den_post0[i][S] + self.post_theta_den_wo_prior[i][S]
        self.theta_mean_post[i][S] = self.theta_num_post[i][S]/self.theta_den_post[i][S]
        self.theta_cov_post[i][S] = 1/self.theta_den_post[i][S]            


    def take_action(self, i, t, X = None):
        """
        In the concurrent setting, the f will be sampled every day after observing rewards from all campaigns that day.
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
        current_time = now()
        
        if t == 0 and i == 0:
            self._init_posterior(self.K*(self.N+1), self.M)
        else:
            if self.approximate:
                if i == 0 and self.order == "concurrent":
                    #step1: update the posterior of f
                    self.update_f(i)  
                    if self.refresh_memory == True and np.sum(self.cnts)>self.refresh_threshold:
                        self.clear_memory()
                    #step2: update the posterior of thetas given the new posterior of f
                    self.update_concurrent_post()

                elif t == 0 and self.order == "sequential":
                    #step1: update the posterior of f
                    self.update_f(i)  
                    if self.refresh_memory == True and np.sum(self.cnts)>self.refresh_threshold:
                        self.clear_memory()
                    #step2: update the posterior of thetas given the new posterior of f
                    self.update_campaign_i_post(i)
            else:
                #step1: update the posterior of f
                self.update_f(i)  
                if self.refresh_memory == True and np.sum(self.cnts)>self.refresh_threshold:
                    self.clear_memory()
                #step2: update the posterior of thetas given the new posterior of f
                self.update_campaign_i_post(i)            
        
        
        
        #if self.order == "concurrent":
        #    if t == 0 and i % self.M == 0:
        #        self._init_posterior(self.K*(self.N+1), self.M)
        #    elif i % self.M == 0:
        #        #step1: update the posterior of f
        #        self.update_f(i)  
        #        #step2: update the posterior of thetas given the new posterior of f
        #        self.update_concurrent_post()

                
                

        self.timer += now() - current_time    
        Rs = np.random.multivariate_normal(self.theta_mean_post[i], np.diag(self.theta_cov_post[i]))
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)

        return A
    
    def clear_memory(self):
        self.memory_clear_times += 1  
        print("GP-FG: clear memory for ", self.memory_clear_times, " times")

        # step 1: store historical data to memory for the update of prior in step 2
        self.f_MKN_prior_mu_previous = self.f_MKN_prior_mu.copy()
        self.f_MKN_prior_kernel_previous = self.f_MKN_prior_kernel.copy()

        # step 2: update the new prior according to previous data
        temp1 = self.f_MKN_prior_kernel_previous.dot(self.Z_all)
        temp2 = self.f_inv_mid.dot(self.R_obs_flatten-self.Z_all.T.dot(self.f_MKN_prior_mu_previous))

        self.f_MKN_prior_mu = self.f_MKN_prior_mu_previous + temp1.dot(temp2)
        self.f_MKN_prior_kernel = self.f_MKN_prior_kernel_previous - temp1.dot(self.f_inv_mid).dot(temp1.T)            

        def prior_f_mu_func(self, i, x, use = "i"):
            if use == "Z_all":
                return self.f_MKN_prior_mu.dot(self.Z_all)
            else:
                return self.f_MKN_prior_mu[i*self.items_tot:(i+1)*self.items_tot]
        def prior_f_kernel_func(self, i, j, x, x_prime = [], use = "1"):
            if j == None:
                j = i
            if use == "1":
                return self.f_MKN_prior_kernel[i*self.items_tot:(i+1)*self.items_tot, j*self.items_tot:(j+1)*self.items_tot]
            if use == "2":
                return self.f_MKN_prior_kernel[i*self.items_tot:(i+1)*self.items_tot, :].dot(self.Z_all)
            if use == "3":
                return self.Z_all.T.dot(self.f_MKN_prior_kernel[:,j*self.items_tot:(j+1)*self.items_tot])
            if use == "4":
                return self.Z_all.T.dot(self.f_MKN_prior_kernel).dot(self.Z_all)


            return self.f_MKN_prior_kernel[i*self.items_tot:(i+1)*self.items_tot, i*self.items_tot:(i+1)*self.items_tot]

        self.prior_f_mu = copy.copy(prior_f_mu_func)
        self.prior_f_kernel = copy.copy(prior_f_kernel_func)


        # step 3: abandon all historical data after updating prior            
        self._init_data_storate(self.K*(self.N+1), self.M)
        self.cnts = np.zeros((self.M, self.K*(self.N+1)))

    def _optimize(self, Rs, t):
        #the reward should be non-negative
        Rs[Rs < 0] = 0
        #the reward for budget = 0 should be 0
        Rs[list(range(0,self.K*(self.N+1), self.N+1))] = 0
        A = knapsack_optimizer(Rs, self.K, self.N)
        
        ## If all of the estimated rewards are zero, we would allocate the budget evenly if it is still within the exploration period, or zero to all otherwise.
        if len(A) == 0:
            A = np.zeros(self.K)
            for i in range(self.K):
                if t < self.exp_episode:
                    A[i] = i*(self.N+1) + self.N//self.K # evenly assign the budget
                else:
                    A[i] = i*(self.N+1) # assign 0 to all
        return A.astype(int)

    ################################################################################################################################################
    ###################################################### receive_reward #################################################################
    ################################################################################################################################################
    def update_Phi(self, i, S, x_S):
        for j in range(len(S)):
            A = S[j]
            x = x_S[j]
            self.Phi_obs[i][A] = self.vstack([self.Phi_obs[i][A], x])
            if self.Phi_obs[i][A].ndim == 1:
                self.Phi_obs[i][A] = self.Phi_obs[i][A][np.newaxis, :]
                
        self.Phi_obs_i[i] = np.vstack(self.Phi_obs[i])
        self.Phi_all = np.vstack(self.Phi_obs_i)

    def update_R_each_task(self, i, obs_R, S):
        for j in range(len(S)):
            A = S[j]
            this_R = obs_R[j]
            self.R_each_task[i][A] = np.append(self.R_each_task[i][A], this_R)


############################################################################################################### 
        
    def update_f(self, i):
        
        self.data_flatten()
        # print("self.R_obs_flatten: ", self.R_obs_flatten)
        self.items_updating = len(self.R_obs_flatten)
        self.Sigma = np.dot(self.Z_all.T, self.Z_all) * self.sigma_1**2
        
        # print("self.items_updating: ", self.items_updating)
  
        ## calculate the inverse: time comsuming part
        if (self.cholesky_decomp == True):
            A = self.prior_f_kernel(self, i, 0, self.Phi_obs_flatten, [], "4") + self.sigma_2**2 * np.identity(self.items_updating) + self.Sigma
            L = cholesky(A, lower=True, check_finite=False)
            self.f_inv_mid = cho_solve((L, True), np.identity(A.shape[0]), check_finite=False)
        else:
            self.f_inv_mid = inv(self.prior_f_kernel(self, i, 0, self.Phi_obs_flatten, [], "4") + self.sigma_2**2 * np.identity(self.items_updating) + self.Sigma)
         
        # calculate the posterior mean of gaussian process f
        def post_f_mu_updating(self, i, x):
            return self.prior_f_mu(self, i, x, "i") + self.prior_f_kernel(self, i, 0, x, self.Phi_obs_flatten, "2").dot(self.f_inv_mid).dot(self.R_obs_flatten - self.prior_f_mu(self, i, self.Phi_obs_flatten, "Z_all"))
        
        # calculate the posterior kernel of gaussian process f
        def post_f_kernel_updating(self, i, j, x, x_prime = []):
            if (len(x_prime) == 0):
                x_prime = x
            if j == None:
                j = i
            return self.prior_f_kernel(self, i, j, x, x_prime, "1") -self.prior_f_kernel(self, i, 0, x, self.Phi_obs_flatten, "2").dot(self.f_inv_mid).dot(self.prior_f_kernel(self, 0, j, self.Phi_obs_flatten, x_prime, "3"))
           
        # update posterior mean and kernel
        self.post_f_mu = copy.copy(post_f_mu_updating)
        self.post_f_kernel = copy.copy(post_f_kernel_updating)        
      
               
    def data_flatten(self, data_type = "both", Z_cal = True):
        '''
        data_type: can be "both", "Phi_obs" or "R_obs"
        '''
        self.Phi_obs_flatten = np.array([], dtype=np.int64).reshape(0,self.p)
        self.R_obs_flatten = np.array([], dtype=np.int64)
        self.Z_all = np.array([], dtype=np.int64).reshape(self.M * self.K * (self.N+1), 0)
        
        for i in range(self.M):
            for s in range(self.K*(self.N+1)):
                if Z_cal == True:
                    if (self.cnts[i,s] > 0):
                        temp = np.zeros((self.M * self.K*(self.N+1), int(self.cnts[i,s])))
                        temp[(i*self.K*(self.N+1)+s),:] = 1
                        self.Z_all = np.hstack((self.Z_all, temp))
                           
                if data_type == "both":
                    self.Phi_obs_flatten = np.vstack((self.Phi_obs_flatten, self.Phi_obs[i][s]))
                    self.R_obs_flatten = np.concatenate((self.R_obs_flatten, self.R_each_task[i][s]))
                elif data_type == "Phi_obs":
                    self.Phi_obs_flatten = np.vstack((self.Phi_obs_flatten, self.Phi_obs[i][s]))
                elif data_type == "R_obs":
                    self.R_obs_flatten = np.concatenate((self.R_obs_flatten, self.R_each_task[i][s]))
                else: 
                    print("Wrong data type specified")
                
        
        

###############################################################################################################         
        
    def update_concurrent_post(self):
        # step1: update the prior of thetas given the newly sampled f
        self.theta_mean_prior = [np.random.multivariate_normal(self.post_f_mu(self, i, self.Phi[i]), self.post_f_kernel(self, i, i, self.Phi[i])) for i in range(self.M)]
        #self.theta_mean_prior = [self.Phi[i].dot(self.sampled_gamma) for i in range(self.M)]
        self.theta_cov_prior = [self.delta_cov for i in range(self.M)]
        self.theta_num_post0 = [self.theta_mean_prior[i]/np.diag(self.delta_cov) for i in range(self.M)]
        self.theta_den_post0 = [1 / np.diag(self.delta_cov) for i in range(self.M)]
        
        # step2: update the posterior of thetas
        self.theta_num_post = [self.theta_num_post0[i] + self.post_theta_num_wo_prior[i] for i in range(self.M)]
        self.theta_den_post = [self.theta_den_post0[i] + self.post_theta_den_wo_prior[i] for i in range(self.M)]
        self.theta_mean_post = [self.theta_num_post[i]/self.theta_den_post[i] for i in range(self.M)]
        self.theta_cov_post = [1/self.theta_den_post[i] for i in range(self.M)]

        
    def update_campaign_i_post(self,i):
        # step1: update the prior of thetas given the newly sampled f
        self.theta_mean_prior[i] = np.random.multivariate_normal(self.post_f_mu(self, i, self.Phi[i]), self.post_f_kernel(self, i, i, self.Phi[i]))
        # self.theta_mean_prior[i] = self.Phi[i].dot(self.sampled_gamma)
        self.theta_cov_prior[i] = self.delta_cov
        self.theta_num_post0[i] = self.theta_mean_prior[i]/np.diag(self.delta_cov)
        self.theta_den_post0[i] = 1 / np.diag(self.delta_cov)
        
        # step2: update the posterior of thetas
        self.theta_num_post[i] = self.theta_num_post0[i] + self.post_theta_num_wo_prior[i]
        self.theta_den_post[i] = self.theta_den_post0[i] + self.post_theta_den_wo_prior[i]
        self.theta_mean_post[i] = self.theta_num_post[i]/self.theta_den_post[i]
        self.theta_cov_post[i] = 1/self.theta_den_post[i]
        
        
###Utility funcs
    def vstack_list_of_list(self, list_of_list):
        return np.vstack([np.vstack([a for a in lis]) for lis in list_of_list])
    
    def conca_list_of_list(self, list_of_list):
        return np.concatenate([np.concatenate([a for a in lis ]) for lis in list_of_list])

    def vstack(self, C):
        A, B = C
        if A.shape[0] == 0:
            return B
        else:
            return np.vstack([A, B])
        
    def _init_empty(self, aa = None, p = None):
        K = self.K
        N = self.N
        M = self.M
        if aa == "num":
            return [[0 for a in range(K*(N+1))] for _ in range(M)]
        if aa == "col":
            return [np.zeros((K*(N+1), 0)) for i in range(M)] 
        if aa == "row":
            return [[np.zeros((0, p)) for a in range(K*(N+1))] for i in range(M)]
        if aa == "null_scalar":
            return [[np.zeros(0) for a in range(K*(N+1))] for i in range(M)]
        return [[np.zeros((0, 0)) for a in range(K*(N+1))] for i in range(M)]