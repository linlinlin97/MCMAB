from _util import *
from _optimizer import *

class TS_agent():
    '''
    oracle: u_prior_mean and u_prior_cov are the true values
    TS: u_prior_mean and u_prior_cov are the random values
    naive pooling, ignoring the difference between adlines across different campaigns
    '''
    @autoargs()
    def __init__(self, K, N, u_prior_mean = None, u_prior_cov_diag = None, sigma = 1, exp_episode = 2, log_log = False):
        ### R ~ N(mu, sigma)
        ### sigma as known
        ### prior over mu
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        """

        self.K, self.N = K, N
        self.items_tot = self.K * (self.N +1)
        self.cnts = np.zeros(self.items_tot)
        self._init_posterior()
        self.exp_episode = exp_episode
        
        self.seed = 42
        self.timer = 0

    def _init_posterior(self):
        self.posterior_u_num = self.u_prior_mean / self.u_prior_cov_diag
        self.posterior_u_den = 1 / self.u_prior_cov_diag
        self.posterior_u = self.posterior_u_num / self.posterior_u_den
        self.posterior_cov_diag = 1 / self.posterior_u_den

    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.normal(self.posterior_u, self.posterior_cov_diag)
        
        if self.log_log:
            Rs = np.exp(Rs)-1
            
        A = self._optimize(Rs, t)
        return A

    def receive_reward(self, i, t, A, obs_R, X = None):
        """
        A is the subset selected in round t
        obs_R is the corresponding reward for each selected (m,k)
        """
        current_time = now()
        # update_data. update posteriors
        self.posterior_u_num[A] += (obs_R / self.sigma ** 2)
        self.posterior_u_den[A] += (1 / self.sigma ** 2)
        self.posterior_u[A] = self.posterior_u_num[A] / self.posterior_u_den[A]
        self.posterior_cov_diag[A] = 1 / self.posterior_u_den[A]
        self.cnts[A] += 1

        self.timer += now() - current_time

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

    
    
    
class N_TS_agent():
    """ 
    Learning each ads campaign independently, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, M, K, N, u_prior_mean = None, u_prior_cov_diag = None, sigma = 1, exp_episode = 2, log_log = False):
        ### R ~ N(mu, sigma)
        ### len(u_prior_mean) = M * (K + 1)
        ### sigma as known
        ### prior over mu
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        """
        
        self.items_tot = K * (N + 1)
        self.cnts = np.zeros((M, self.items_tot))
        self._init_posterior(M)
        self.exp_episode = exp_episode
        self.seed = 42
        self.timer = 0
        
    def _init_posterior(self, M):
        self.posterior_u_num = [self.u_prior_mean / self.u_prior_cov_diag for _ in range(M)]
        self.posterior_u_den = [1 / self.u_prior_cov_diag for _ in range(M)]
        self.posterior_u = [self.posterior_u_num[i] / self.posterior_u_den[i] for i in range(M)] #zeros((N, items_tot))
        self.posterior_cov_diag = [1 / self.posterior_u_den[i] for i in range(M)]
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.multivariate_normal(self.posterior_u[i], np.diag(self.posterior_cov_diag[i]))
        
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)
        #if i ==0:
        #    print(i, (A%(self.N+1))/self.N)
        
        return A

    def receive_reward(self, i, t, A, obs_R, X = None):
        """
        A is the subset selected in round t
        obs_R is the corresponding reward for each selected (m,k)
        """
        current_time = now()
        # update_data. update posteriors
        self.posterior_u_num[i][A] += (obs_R / self.sigma ** 2)
        self.posterior_u_den[i][A] += (1 / self.sigma ** 2)
        self.posterior_u[i][A] = self.posterior_u_num[i][A] / self.posterior_u_den[i][A]
        self.posterior_cov_diag[i][A] = 1 / self.posterior_u_den[i][A]
        self.cnts[i, A] += 1

        self.timer += now() - current_time

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
    
    
class meta_oracle_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, M, K, N, u_prior_mean = None, u_prior_cov_diag = None, sigma = 1, exp_episode = 2, log_log = False):
        self.cnts = np.zeros((M, K*(N+1)))
        self._init_posterior(M, K*(N+1))
        self.exp_episode = exp_episode
        self.seed = 42
        
    def _init_posterior(self, M, L):
        self.posterior_u_num = [self.u_prior_mean[i] / self.u_prior_cov_diag[i] for i in range(M)]
        self.posterior_u_den = [1 / self.u_prior_cov_diag[i] for i in range(M)]
        self.posterior_u = [self.posterior_u_num[i] / self.posterior_u_den[i] for i in range(M)] 
        self.posterior_cov_diag = [1 / self.posterior_u_den[i] for i in range(M)] 
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.multivariate_normal(self.posterior_u[i], np.diag(self.posterior_cov_diag[i]))
        if self.log_log:
            Rs = np.exp(Rs)-1
        A = self._optimize(Rs, t)
        return A
        
    def receive_reward(self, i, t, A, obs_R, X = None):
        # update_data. update posteriors
        self.posterior_u_num[i][A] += (obs_R / self.sigma ** 2)
        self.posterior_u_den[i][A] += (1 / self.sigma ** 2)
        self.posterior_u[i][A] = self.posterior_u_num[i][A] / self.posterior_u_den[i][A]
        self.posterior_cov_diag[i][A] = 1 / self.posterior_u_den[i][A]
        self.cnts[i, A] += 1
        
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
    