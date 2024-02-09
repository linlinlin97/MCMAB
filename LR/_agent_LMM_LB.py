from _util import *
from _optimizer import *

class LB_agent():
    @autoargs()
    def __init__(self, M, K, N, prior_gamma_mu = None, prior_gamma_cov = None, sigma = 1, d = 3, exp_episode = 2, order = None, log_log = False,
                true_gamma = None, approximate = True, real = False, transfer_x_type = None):
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        """
        self.items_tot = K * (N + 1)
        self._init_posterior()
        self.seed = 42
        #self.cnt_0 = 0
        #self.cnt_1 = 0

    def _init_posterior(self):
        self.Cov = self.prior_gamma_cov.copy()
        self.Cov_inv = inv(self.Cov)
        self.mu = self.prior_gamma_mu.copy()

    def take_action(self, i, t, X):
        """
        X = [items_tot in campaign i, p]
        """
        np.random.seed(self.seed)
        self.seed += 1
        if self.approximate:
            if self.order == "concurrent":
                if i == 0:
                    self.sampled_gamma = np.random.multivariate_normal(self.mu, self.Cov)
            elif self.order == "sequential":
                if t == 0:
                    self.sampled_gamma = np.random.multivariate_normal(self.mu, self.Cov)
        else:
            self.sampled_gamma = np.random.multivariate_normal(self.mu, self.Cov)
        
        if self.real:
            X = self._transfer_x(X)
            
        Rs = X.dot(self.sampled_gamma)
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)
        return A

    def receive_reward(self, i, t, A, obs_R, X):
        # update_data. update posteriors
        x = X[A]
        
        #self.cnt_0 += len(np.where(x[:, -1] == 0)[0])
        #self.cnt_1 += len(A)
        #if i == 0:
        #    print(self.cnt_0, self.cnt_1, self.cnt_0/self.cnt_1)
        rows_w_budget = np.where(x[:, -1] != 0)[0]
        x = x[rows_w_budget]
        obs_R = obs_R[rows_w_budget]
        
        if self.real:
            x = self._transfer_x(x)
        
        self.Cov_inv_last = self.Cov_inv.copy()
        self.Cov_inv += x.T.dot(x) / self.sigma ** 2
        self.Cov = inv(self.Cov_inv)
        self.mu = self.Cov.dot(self.Cov_inv_last.dot(self.mu) + x.T.dot(obs_R / self.sigma ** 2))
        
        if i == 0 and self.true_gamma:
            print(np.mean((self.mu - self.true_gamma)**2))

    def _optimize(self, Rs, t):
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
    
    

class LB_budget_ind():
    @autoargs()
    def __init__(self, M, K, N, prior_gamma_mu = None, prior_gamma_cov = None, sigma = 1, with_intercept = True, exp_episode = 2, log_log = False):
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        """
        self._init_posterior()
        self.seed = 42

    def _init_posterior(self):
        self.Cov_all = [self.prior_gamma_cov.copy() for _ in range(self.M*self.K)]
        self.Cov_inv_all = [inv(cov)  for cov in self.Cov_all]
        self.mu_all = [self.prior_gamma_mu.copy() for _ in range(self.M*self.K)]

    def take_action(self, i, t, X):
        """
        X = [items_tot, p]
        """
        np.random.seed(self.seed)
        self.seed += 1
        
        Rs = []
        sampled_gamma = np.random.multivariate_normal(self.prior_gamma_mu, self.prior_gamma_cov)
        for k in range(self.K):
            if t <= 10:
                sampled_gamma_k = sampled_gamma
            else:
                sampled_gamma_k = np.random.multivariate_normal(self.mu_all[i*self.K+k], self.Cov_all[i*self.K+k])
            if self.with_intercept:
                Rs += list(X[k*(self.N+1):(k+1)*(self.N+1),[0,-1]].dot(sampled_gamma_k))
            else:
                Rs += list(X[k*(self.N+1):(k+1)*(self.N+1),-1].dot(sampled_gamma_k))
        Rs = np.array(Rs)
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)
        return A

    def receive_reward(self, i, t, A, obs_R, X):
        # update_data. update posteriors
        x = X[A]
        self.Cov_inv_last_all = self.Cov_inv_all.copy()
        
        if self.with_intercept:
            x = x[:,[0,-1]]
        else:
            x = x[:,-1]
            
        for k in range(self.K):
            self.Cov_inv_all[i*self.K+k] += x[k].T.dot(x[k]) / self.sigma ** 2
            self.Cov_all[i*self.K+k] = inv(self.Cov_inv_all[i*self.K+k])
            self.mu_all[i*self.K+k] = self.Cov_all[i*self.K+k].dot(self.Cov_inv_last_all[i*self.K+k].dot(self.mu_all[i*self.K+k]) + x[k].T.dot(obs_R[k] / self.sigma ** 2))


    def _optimize(self, Rs, t):
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