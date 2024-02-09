#supply sources = [APS, 3PX] and channel = [Mobile App, Video Mobile App In-Stream, Video Mobile App Out-Stream, Web, Video Web, Web Billboard]
from _util import *
from _optimizer import *
import numpy as np
import pandas as pd
import random
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
#setting with only the campaign-level random effect

class Semi_env():
    @autoargs()
    def __init__(self, M, K, N, T, B_min, B_max, sigma_m, sigma_eps, with_intercept, seed = 42):
        """
        N: number of discretized budget for each ad line
        K: number of ad lines
        M: number of campaigns
        """
        self.setting = locals()
        self.setting['self'] = None
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.get_true_env(M, K, N, B_min, B_max, with_intercept)
        self.get_theta(M, K, N, sigma_m)
        self._get_optimal_action()
        self.errors = randn(M, T, K*(N+1)) * sigma_eps

    def get_true_env(self, M, K, N, B_min, B_max, with_intercept):
        
        XCM = pd.read_csv('Real_data/XCM_cleaned.csv')
        X = XCM[['channel', 'Supply_Source', 'dim_ad_id', 'log_total_cost_usd']]
        y = XCM[['log_clicks']]
        
        X_fit = pd.get_dummies(X[['channel', 'Supply_Source', 'log_total_cost_usd']], columns=['channel', 'Supply_Source'], drop_first=True)
        #mobile app, 3px are the reference group
        cb_model = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=12, l2_leaf_reg = 1, border_count = 254, bagging_temperature = 1, verbose=False)
        cb_model.fit(X_fit, y, verbose=False)
        
        ad_selected = random.sample(X.dim_ad_id.unique().tolist(), M*K)
        selected_X = X[X.dim_ad_id.isin(ad_selected)][['channel', 'Supply_Source', 'dim_ad_id']].drop_duplicates()
        selected_X = pd.get_dummies(selected_X[['channel', 'Supply_Source', 'dim_ad_id']], columns=['channel', 'Supply_Source'], drop_first=True)
        
        self.B = np.random.uniform(B_min, B_max, M)
        X_n = np.array([np.repeat(self.B[i] * np.array(range(0, (N+1), 1)).reshape(1,-1)/N, K, axis = 0).reshape(-1,1) for i in range(M)])
        X_n = X_n.reshape(M*K*(N+1),-1)
        X_n = np.log(X_n + 1)
    
        selected_X = selected_X.sample(frac=1, random_state=1).reset_index(drop=True)
        selected_X = selected_X.loc[selected_X.index.repeat(N+1)].reset_index(drop=True)
        selected_X['log_total_cost_usd'] = X_n
    
        Phi = selected_X[['channel_Video Mobile App In-Stream',
               'channel_Video Mobile App Out-Stream', 'channel_Video Web',
               'channel_Web', 'channel_Web Billboard', 'Supply_Source_APS',
               'log_total_cost_usd']]
        
        self.theta_mean = cb_model.predict(Phi)
        self.theta_mean = self.theta_mean.reshape(M,-1)       
        self.Phi = np.array(Phi, dtype = float)
    
        if self.with_intercept:
            intercept = np.ones((M*K*(N+1),1))
            self.Phi = np.concatenate([intercept, self.Phi], axis = 1).reshape((M,K*(N+1),-1))
        self.d = self.Phi.shape[2]

    def get_theta(self, M, K, N, sigma_m):
        """
        misspecifications can be added here. nonlinear as the true model to show the robustness w.r.t. LMM
        """
        self.theta_mean[self.theta_mean < 0] = 0
        self.theta_mean[:,list(range(0,K*(N+1), N+1))] = 0        
        self.delta = np.random.multivariate_normal(np.zeros(K*(N+1)), sigma_m ** 2 * np.identity(K*(N+1)), M)
        self.theta = [the_mean + delta for the_mean, delta in zip(self.theta_mean, self.delta)]
        self.theta = np.vstack(self.theta)
        
        #the reward should be non-negative
        self.theta[self.theta < 0] = 0
        self.theta[:,list(range(0,K*(N+1), N+1))] = 0

    def get_reward(self, i, t, A):
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
            
    def get_optimal_reward(self, i, t):
        opt_A = self.opt_As[i]
        return self.get_reward(i, t, opt_A)
        
    def _get_optimal_action(self):
        #for each campaing m, run _optimizer to get the optimal action.
        # input is self.theta. [M, K*(N + 1)]
        self.opt_As = [self._optimize(np.exp(self.theta[i])-1, 0) for i in range(self.M)]
        #print([(self.opt_As[i]%(self.N+1))/self.N for i in range(self.M)])
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
