from _util import *
from _optimizer import *
import scipy.linalg
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class MTB_agent():
    @autoargs()
    def __init__(self, sigma_2 = 1, gamma_prior_mean = None, gamma_prior_cov = None,
                 sigma_1 = None, M = None, K = None, N = None,
                 Xs = None, order = None, exp_episode = 2, log_log = False, true_gamma = None,
                 approximate = True, real = False, transfer_x_type = None):
        
        self.seed = 42
        self.p = p = len(gamma_prior_mean)
        self.delta_cov = sigma_1**2*identity(K*(N+1))
        if real:
            Xs = np.array([self._transfer_x(Xs[i]) for i in range(M)])
        self.Phi = Xs # [M, K*(N+1), p]
        self._init_data_storate(K*(N+1), p, M)
        self.gamma_prior_cov_inv = inv(gamma_prior_cov)
        #self.cnt_0 = 0
        #self.cnt_1 = 0

    def _init_data_storate(self, tot_items_m, p, M):
        """ initialize data storage and components required for computing the posterior
        """
        #initialize the records of Rewards to facilitate the future rearragment
        self.R_each_task = [[np.zeros(0) for a in range(tot_items_m)] for _ in range(M)]
        
        #initialize the records of fatures to facilitate the future rearragment
        self.Phi_obs = self._init_empty("row", p)
        self.Phi_obs_i = [np.zeros((0, p)) for i in range(M)]
        
        #initialize the records of the number of interactions with each item to facilitate the future rearragment
        self.Sigma_idx = np.array(self._init_empty("num"))
        
        ### inverse part (\sigma_1^2 Z^TZ + \sigma_2^2 I)^{-1}
        self.J_inv_to_be_updated = set()
        self.J_sigmaI_inv = self._init_empty()
        self.J_sigmaI_inv_dot_Phi_each = self._init_empty("row", p)
        self.J_sigmaI_inv_dot_R_each = self._init_empty('null_scalar')

    def _init_posterior(self, tot_items_m, p, M):
        #step1: initialize a gamma, sampling from the prior distribution
        self.sampled_gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
        
        #step2: initialize the priors of thetas given the sampled gamma
        self.theta_mean_prior = [self.Phi[i].dot(self.sampled_gamma) for i in range(M)]
        self.theta_cov_prior = [self.delta_cov for i in range(M)]
        self.post_theta_num_wo_prior = [zeros(tot_items_m) for i in range(M)]
        self.post_theta_den_wo_prior = [zeros(tot_items_m) for i in range(M)]
        
        #step3: calculate the corresponding posteiors of thetas
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
        x = X[S]
        
        #self.cnt_0 += len(np.where(x[:, -1] == 0)[0])
        #self.cnt_1 += len(S)
        #if i == 0:
        #    print(self.cnt_0, self.cnt_1, self.cnt_0/self.cnt_1)
            
        rows_w_budget = np.where(x[:, -1] != 0)[0]
        x = x[rows_w_budget]
        obs_R = obs_R[rows_w_budget]
        S = S[rows_w_budget]
        
        if self.real:
            x = self._transfer_x(x)
        # receive observation, update the feature matrix Phi, the reward vector R, and the records of the number of interactions
        self.update_Phi(i, S, x)
        self.update_R_each_task(i, obs_R, S)
        self.Sigma_idx[i][S] += 1
        
        # Keep track of the items that interacted after the last gamma posterior update
        for j in range(len(S)):
            A = S[j]
            self.J_inv_to_be_updated.add((i,A))
        
        # update the posteriors of the arms played
        self.post_theta_num_wo_prior[i][S] += (obs_R / self.sigma_2 ** 2)
        self.post_theta_den_wo_prior[i][S] += (1 / self.sigma_2 ** 2)
        self.theta_num_post[i][S] = self.theta_num_post0[i][S] + self.post_theta_num_wo_prior[i][S]
        self.theta_den_post[i][S] = self.theta_den_post0[i][S] + self.post_theta_den_wo_prior[i][S]
        self.theta_mean_post[i][S] = self.theta_num_post[i][S]/self.theta_den_post[i][S]
        self.theta_cov_post[i][S] = 1/self.theta_den_post[i][S]            


    def take_action(self, i, t, X = None):
        """
        In the concurrent setting, the gamma will be sampled every day after observing rewards from all campaigns that day.
        """
        np.random.seed(self.seed)
        self.seed += 1
        if t == 0 and i == 0:
            self._init_posterior(self.K*(self.N+1), self.p, self.M)
        else:
            if self.approximate:
                if i == 0 and self.order == "concurrent":
                    #step1: update the posterior of gamma
                    self.compute_inverse()
                    #step2: sample a gamma
                    self.sample_gamma()
                    #step3: update the posterior of thetas given the newly sampled gamma
                    self.update_concurrent_post(self.M)
                elif t == 0 and self.order == "sequential":
                    #step1: update the posterior of gamma
                    self.compute_inverse()
                    #step2: sample a gamma
                    self.sample_gamma()
                    #step3: update the posterior of thetas given the newly sampled gamma
                    self.update_campaign_i_post(i)
            else:
                #step1: update the posterior of gamma
                self.compute_inverse()
                #step2: sample a gamma
                self.sample_gamma()
                #step3: update the posterior of thetas given the newly sampled gamma
                self.update_campaign_i_post(i)                
                
        Rs = np.random.multivariate_normal(self.theta_mean_post[i], np.diag(self.theta_cov_post[i]))
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)
        return A

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
            
    def compute_inverse(self):
        for i, A in self.J_inv_to_be_updated:
            N_a = self.Sigma_idx[i][A]
            self.J_sigmaI_inv[i][A] = self.fast_inv(N_a) 
            self.J_sigmaI_inv_dot_Phi_each[i][A] = self.J_sigmaI_inv[i][A].dot(self.Phi_obs[i][A])
            self.J_sigmaI_inv_dot_R_each[i][A] = self.J_sigmaI_inv[i][A].dot(self.R_each_task[i][A])
        # clear out
        self.J_inv_to_be_updated = set()
        
        self.inv['J_sigmaI_inv_dot_Phi_all'] = self.vstack_list_of_list(self.J_sigmaI_inv_dot_Phi_each)
        self.inv['J_sigmaI_inv_dot_R'] = self.conca_list_of_list(self.J_sigmaI_inv_dot_R_each)

    def fast_inv(self, l):
        """
        (J_ia + sigma I)^{-1}
        = sigma ** -2 * identity(N_ia) - sigma ** -4 * (sigma1 ** -2 + sigma ** -2 * N_ia) ** -1 * 11'
        """
        sigma_2, sigma_1 = self.sigma_2, self.sigma_1
        return sigma_2 ** -2 * identity(l) - sigma_2 ** -4 * (sigma_1 ** -2 + sigma_2 ** -2 * l) ** -1 * np.ones((l,1)).dot(np.ones((1,l)))
    
    
    def sample_gamma(self):
        V_Phi = self.inv['J_sigmaI_inv_dot_Phi_all']
        V_R = self.inv['J_sigmaI_inv_dot_R']
        sigma_tilde= inv(self.Phi_all.T.dot(V_Phi)+inv(self.gamma_prior_cov))
        mean = sigma_tilde.dot(self.Phi_all.T.dot(V_R)+self.gamma_prior_cov_inv.dot(self.gamma_prior_mean))
        self.sampled_gamma = np.random.multivariate_normal(mean, sigma_tilde) 
        
        if self.true_gamma:
            print(np.mean((mean - self.true_gamma)**2))

    
    def update_concurrent_post(self,M):
        # step1: update the prior of thetas given the newly sampled gamma
        self.theta_mean_prior = [self.Phi[i].dot(self.sampled_gamma) for i in range(M)]
        self.theta_cov_prior = [self.delta_cov for i in range(M)]
        self.theta_num_post0 = [self.theta_mean_prior[i]/np.diag(self.delta_cov) for i in range(M)]
        self.theta_den_post0 = [1 / np.diag(self.delta_cov) for i in range(M)]
        
        # step2: update the posterior of thetas
        self.theta_num_post = [self.theta_num_post0[i] + self.post_theta_num_wo_prior[i] for i in range(M)]
        self.theta_den_post = [self.theta_den_post0[i] + self.post_theta_den_wo_prior[i] for i in range(M)]
        self.theta_mean_post = [self.theta_num_post[i]/self.theta_den_post[i] for i in range(M)]
        self.theta_cov_post = [1/self.theta_den_post[i] for i in range(M)]
        
    def update_campaign_i_post(self,i):
        # step1: update the prior of thetas given the newly sampled gamma
        self.theta_mean_prior[i] = self.Phi[i].dot(self.sampled_gamma)
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
    
    def _transfer_x(self, x):
        if self.transfer_x_type == 1:
            # the vars included in the original x: intercept, 'channel_Video Mobile App In-Stream', 'channel_Video Mobile App Out-Stream', 
            #                                      'channel_Video Web', 'channel_Web', 'channel_Web Billboard', 'Supply_Source_APS','log_total_cost_usd'
            # insert APS*channels
            for i in range(5):
                x = np.insert(x, i+7, x[:,i+1]*x[:,6], axis=1)
            # insert 'cost_channel_VMAIn','cost_channel_WB'
            x = np.insert(x, -2, x[:,1]*x[:,-1], axis=1)
            x = np.insert(x, -2, x[:,5]*x[:,-1], axis=1)
        elif self.transfer_x_type == 2:
            for i in range(5):
                x = np.insert(x, i+7, x[:,i+1]*x[:,6], axis=1)
            for i in range(5):
                x = np.insert(x, -2, x[:,i]*x[:,-1], axis=1)
        else:
            x = x
        return x