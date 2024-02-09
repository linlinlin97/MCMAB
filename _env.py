from _util import *
from _optimizer import *
#setting with only the campaign-level random effect
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class Semi_env():
    @autoargs()
    def __init__(self, M, K, N, T, B_min, B_max, sigma_m, sigma_eps, mu_gamma, Sigma_gamma, with_intercept = True,
                 dm = None, Xm_mu = None, Xm_sigma = None, dk = None, Xk_mu = None, Xk_sigma = None, seed = 42, log_log = True,
                env_setting = "LMM"):
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        """
        self.setting = locals()
        self.setting['self'] = None
        self.seed = seed
        np.random.seed(self.seed)
        
        self.gamma = np.random.multivariate_normal(mu_gamma, Sigma_gamma, 1)[0]
        if self.log_log:
            self.gamma[-1] = np.clip(self.gamma[-1], .01, .99)
            #print(self.gamma[-1])
        self.get_Phi(M, K, N)
        self.get_theta(M, K, N, sigma_m)
        self._get_optimal_action()
        self.errors = randn(M, T, K*(N+1)) * sigma_eps


    def get_Phi(self, M, K, N):
        """ consider the simple case now
        [M*K*(N+1), d]
        """
        X_m = np.random.multivariate_normal(self.Xm_mu, self.Xm_sigma, (1, M))
        X_m = np.repeat(X_m,K*(N+1), axis = 1).reshape(M*K*(N+1),-1)
            
        # if we assume that all campaigns include entirely different set of adlines, then use the following code to generate X_k
        X_k = np.random.multivariate_normal(self.Xk_mu, self.Xk_sigma, (1, M * K))
        X_k = np.repeat(X_k,(N+1), axis = 1).reshape(M*K*(N+1),-1)
            
        self.B = np.random.uniform(self.B_min, self.B_max, M)
        X_n = np.array([np.repeat(self.B[i] * np.array(range(0, (N+1), 1)).reshape(1,-1)/N, K, axis = 0).reshape(-1,1) for i in range(M)])
        X_n = X_n.reshape(M*K*(N+1),-1)
        
        if self.log_log:
            X_m = abs(X_m)
            X_k = abs(X_k)
            X_n = np.log(X_n + 1)

        
        if self.with_intercept:
            intercept = np.ones((M*K*(N+1),1))
            self.Phi = np.concatenate([intercept, X_m, X_k, X_n], axis = 1)
            self.Phi = self.Phi.reshape((M,K*(N+1),-1)) #reshape to (M,K,N+1,1) if want to add adline-specific random error
            self.d = self.Phi.shape[2]
        else:
            self.Phi = np.concatenate([X_m, X_k, X_n], axis = 1)
            self.Phi = self.Phi.reshape((M,K*(N+1),-1))
            self.d = self.Phi.shape[2]

    def get_theta(self, M, K, N, sigma_m):
        """
        misspecifications can be added here. nonlinear as the true model to show the robustness w.r.t. LMM
        """
        if self.env_setting == "LMM":
            self.theta_mean = np.vstack([Phi_m[:,:-1].dot(self.gamma[:-1]) + Phi_m[:,-1].dot(self.gamma[-1]) for Phi_m in self.Phi])
            self.true_l1 = np.vstack([(Phi_m[:,:-1].dot(self.gamma[:-1]))[list(range(0,self.K*(self.N+1), self.N+1))] for Phi_m in self.Phi])
            self.true_l2 = [[self.gamma[-1]]*self.K for _ in range(M)]
            self.true_prior_mean = [[np.array([self.true_l1[i][k], self.true_l2[i][k]]) for k in range(self.K)] for i in range(M)]
        elif self.env_setting == "GP":
            self.theta_mean = np.vstack([np.log(abs((Phi_m**2).dot(self.gamma))) for Phi_m in self.Phi])
        elif self.env_setting == "NN":
            self.theta_mean = np.vstack([np.log(abs((Phi_m**2).dot(self.gamma))) for Phi_m in self.Phi])
        self.theta_mean[self.theta_mean < 0] = 0
        self.theta_mean[:,list(range(0,K*(N+1), N+1))] = 0
        
        self.delta = np.random.multivariate_normal(np.zeros(K*(N+1)), sigma_m ** 2 * np.identity(K*(N+1)), M)
        
        self.theta = [the_mean + delta for the_mean, delta in zip(self.theta_mean, self.delta)]
        self.theta = np.vstack(self.theta)
        
        #the reward should be non-negative
        self.theta[self.theta < 0] = 0
        self.theta[:,list(range(0,K*(N+1), N+1))] = 0
        print(self.theta)

    def get_reward(self, i, t, A):
        if self.log_log:
            obs_log_R = self.theta[i][A] + self.errors[i][t][A]
            
            #if log_log: log(y+1) > 0, Y = 0 if 0 budget
            obs_log_R[obs_log_R < 0] = 0
            #the reward for budget = 0 should be 0
            mask = np.isin(A, list(range(0,self.K*(self.N+1), self.N+1)))
            obs_log_R[mask] = 0
            
            #the LHS = log(Y+1)
            obs_R = np.exp(obs_log_R) - 1
            R = np.sum(obs_R)
            
            exp_R = np.sum(np.exp(self.theta[i][A]) - 1)

            return [obs_R, exp_R, R], obs_log_R
            
        else:
            obs_R = self.theta[i][A] + self.errors[i][t][A]
            obs_R[obs_R < 0] = 0
            #the reward for budget = 0 should be 0
            mask = np.isin(A, list(range(0,self.K*(self.N+1), self.N+1)))
            obs_R[mask] = 0
            
            R = np.sum(obs_R)
            exp_R = np.sum(self.theta[i][A])

            return [obs_R, exp_R, R]

    def get_optimal_reward(self, i, t):
        opt_A = self.opt_As[i]
        return self.get_reward(i, t, opt_A)
        
    def _get_optimal_action(self):
        #for each campaing m, run _optimizer to get the optimal action.
        # input is self.theta. [M, K*(N + 1)]
        if self.log_log:
            self.opt_As = [self._optimize(np.exp(self.theta[i])-1, 0) for i in range(self.M)]
        else:
            self.opt_As = [self._optimize(self.theta[i], 0) for i in range(self.M)]
        #print(self.gamma, np.mean([np.sum((self.opt_As[i]%(self.N+1))/self.N) for i in range(self.M)]))
        #print(self.theta[0],(np.array(self.opt_As[0])%(self.N+1))/self.N)
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
                if t < 2:
                    A[i] = i*(self.N+1) + self.N//self.K # evenly assign the budget
                else:
                    A[i] = i*(self.N+1) # assign 0 to all
        return A.astype(int)