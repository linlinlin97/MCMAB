from _util import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import _env
import LR._agent_LMM_TS as _agent_LMM_TS
import LR._agent_LMM_LB as _agent_LMM_LB
import LR._agent_LMM_MTB as _agent_LMM_MTB
import NN._agent_NeuralTS as _agent_NN_FD
import NN._agent_NeuralTS_MTB as _agent_NN_MTB

# concurrent +  memory check, thus faster
import GP._agent_GP_fast as _agent_GP_FD_fast
import GP._agent_GP_MTB_fast as _agent_GP_MTB_fast

import Contextual_Bandit as CB
import Hibou
import Real_data.real_env as _env_real
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class Experiment():
    """ 
    main module for running the experimnet
    """
    @autoargs()
    def __init__(self, M, K, N, T, B_min, B_max, sigma_m, sigma_eps, 
                 mu_gamma = None, Sigma_gamma = None, kernel_gamma = None, 
                 with_intercept = True, dm = None, Xm_mu = None, Xm_sigma = None, 
                 dk = None, Xk_mu = None, Xk_sigma = None, 
                 seed = 42, order = "concurrent", env_setting = "LMM", log_log = False, prior_f_mu = None):
        
        self.setting = locals()
        self.setting['self'] = None
        
        if env_setting == "LMM" or env_setting == "NN" or env_setting == "GP":
            self.env = _env.Semi_env(M = M, K = K, N = N, T = T, B_min = B_min, B_max = B_max, 
                                     sigma_m = sigma_m, sigma_eps = sigma_eps, mu_gamma = mu_gamma, Sigma_gamma = Sigma_gamma,
                                     with_intercept = with_intercept, dm = dm, Xm_mu = Xm_mu, Xm_sigma = Xm_sigma, 
                                     dk = dk, Xk_mu = Xk_mu, Xk_sigma = Xk_sigma, seed = seed, log_log = self.log_log, 
                                     env_setting = self.env_setting)
            self.gamma = self.env.gamma
        elif env_setting == "real":
            self.env = _env_real.Semi_env(M = M, K = K, N = N, T = T, B_min = B_min, B_max = B_max, 
                                     sigma_m = sigma_m, sigma_eps = sigma_eps, with_intercept = with_intercept, 
                                    seed = seed)
        self.d = self.env.d
        self.Xs = self.env.Phi # [M, K*(N+1), d]
        self.theta = self.env.theta
        self.get_task_sequence()
   
    def _init_agents(self, agents = None):
        # sigma, Sigma_delta
        self.agents = agents
        self.agent_names = agent_names = list(agents.keys())
        self.record = {}
        self.record['R'] = {name : [] for name in agent_names}
        self.record['exp_R'] = {name : [] for name in agent_names}
        self.record['R']['oracle'] = []
        self.record['exp_R']['oracle'] = []
        self.record['A'] = {name : [] for name in agent_names}
        self.record['regret'] = {name : [] for name in agent_names}
        self.record['meta_regret'] = {name : [] for name in agent_names}


    def get_task_sequence(self):
        self.task_sequence = []
        if self.order == "sequential":
            for i in range(self.M):
                for t in range(self.T):
                    self.task_sequence.append([i, t])
        if self.order == "concurrent":
            for t in range(self.T):
                for i in range(self.M):
                    self.task_sequence.append([i, t])
       
        
    def run(self):
        # sample one task, according to the order
        self.run_4_one_agent('oracle')
        for name in self.agent_names: 
            t0 = now()
            self.run_4_one_agent(name)
            t1 = now()
            print(name+' is done! It takes '+str(t1-t0))
        self.post_process()

    def run_4_one_agent(self, name):
        for i, t in self.task_sequence:
            self.run_one_time_point(i, t, name)

    def run_one_time_point(self, i, t, name):
        if name == "oracle":
            if self.log_log:
                R_origin_level, obs_log_R = self.env.get_optimal_reward(i, t)
                obs_R_opt, exp_R_opt, R_opt = R_origin_level
                self.record['R']["oracle"].append(R_opt)
                self.record['exp_R']["oracle"].append(exp_R_opt)
            else:
                obs_R_opt, exp_R_opt, R_opt = self.env.get_optimal_reward(i, t)
                self.record['R']["oracle"].append(R_opt)
                self.record['exp_R']["oracle"].append(exp_R_opt)
        else:
            X = self.Xs[i]
            # provide the task id and its feature to the agent, and then get the action from the agent
            A = self.agents[name].take_action(i, t, X)
            # provide the action to the env and then get reward from the env
            if self.log_log:
                R_origin_level, obs_log_R = self.env.get_reward(i, t, A)
                obs_R, exp_R, R = R_origin_level
                # provide the reward to the agent
                if name == "Hibou":
                    self.agents[name].receive_reward(i, t, A, obs_R, X)
                else:
                    self.agents[name].receive_reward(i, t, A, obs_log_R, X)
                
                # collect the reward
                self.record['R'][name].append(R)
                self.record['exp_R'][name].append(exp_R)
                self.record['A'][name].append(A)
            else:
                obs_R, exp_R, R = self.env.get_reward(i, t, A)
                # provide the reward to the agent
                self.agents[name].receive_reward(i, t, A, obs_R, X)
                
                # collect the reward
                self.record['R'][name].append(R)
                self.record['exp_R'][name].append(exp_R)
                self.record['A'][name].append(A)
  
    def post_process(self):
        for name in self.agent_names:
            self.record['regret'][name] = arr(self.record['R']["oracle"]) - arr(self.record['R'][name])

        self.record['cum_regret'] = {name : np.cumsum(self.record['regret'][name]) for name in self.agent_names}
        self.record['cum_R'] = {name : np.cumsum(self.record['R'][name]) for name in self.agent_names}
        self.record['cum_exp_R'] = {name : np.cumsum(self.record['exp_R'][name]) for name in self.agent_names}
        # x: time, y: cum_regret: group, name
        self.record['cum_regret_df'] = self.organize_Df(self.record['cum_regret'])
        self.record['cum_R_df'] = self.organize_Df(self.record['cum_R'])
        self.record['cum_exp_R_df'] = self.organize_Df(self.record['cum_exp_R'])
                
        if "oracle-TS" in self.agent_names:
            for name in self.agent_names:
                self.record['meta_regret'][name] = arr(self.record['R']['oracle-TS']) - arr(self.record['R'][name])
            self.record['cum_meta_regret'] = {name : np.cumsum(self.record['meta_regret'][name]) for name in self.agent_names}
            self.record['cum_meta_regret_df'] = self.organize_Df(self.record['cum_meta_regret'])


    def organize_Df(self, r_dict):
        T = len(r_dict[self.agent_names[0]])
        a = pd.DataFrame.from_dict(r_dict)
        # a.reset_index(inplace=True)
        a = pd.melt(a)
        a['time'] = np.tile(np.arange(T), len(self.agent_names))
        a = a.rename(columns = {'variable':'method', "value" : "regret", "time" : "time"})
        return a

            
    def plot_regret(self, skip_methods = ["TS"], plot_meta = True):
        # https://seaborn.pydata.org/generated/seaborn.lineplot.html
        #ax.legend(['label 1', 'label 2'])
        if plot_meta:
            data_plot =  self.record['cum_meta_regret_df'] 
            data_plot = data_plot[data_plot.method != "oracle-TS"]
        else:
            data_plot =  self.record['cum_regret_df'] 
        if skip_methods is not None:
            for met in skip_methods:
                data_plot = data_plot[data_plot.method != met]
        ax = sns.lineplot(data = data_plot, x="time", y="regret", 
                          n_boot = 100, hue="method" # group variable
                         )

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class run_experiment():
    @autoargs()
    def __init__(self, M, K, N, T, B_min, B_max, sigma_m, sigma_eps, 
                 dm = None, dk = None, with_intercept = None, 
                 mu_gamma_factor = None, Sigma_gamma_factor = None, 
                 Sigma_xm_factor = None, Sigma_xk_factor = None, cholesky_decomp = True,
                 GP_prior_f_mu = None, kernel = "RBF", kernel_gamma_factor = 1,
                 order = "concurrent", env_setting = "LMM1", save_prefix = None,
                 NN_m = None, NN_L = None, NN_reg = None, NN_nu = None, lr = None, 
                 exp_episode = None, refresh_memory = True, refresh_threshold = None,
                 log_log = False, augment_size1 = None, augment_size2 = None, 
                 CB_Sigma_gamma = None, CB_mu_gamma = None,
                prior_f_mu_env = None):          
        self.setting = locals()
        self.setting['self'] = None
        self.title_settting = " ".join([str(key) + "=" + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float]])
        printR(self.title_settting)
        self.date_time = get_date() + get_time()
        ################
        if self.env_setting != "real":
            if with_intercept:
                self.d = dm + dk + 1 + 1
            else:
                self.d = dm + dk + 1
            
            if self.log_log:
                self.mu_gamma_true = CB_mu_gamma
                self.Sigma_gamma_true = CB_Sigma_gamma
                self.prior_gamma_mu = np.zeros(int(self.d))
                self.prior_gamma_sigma = np.diag(np.ones(int(self.d))*20)
            else:
                self.mu_gamma_true = mu_gamma_factor * np.ones(self.d)
                self.prior_gamma_mu = self.mu_gamma_true
                if Sigma_gamma_factor == "identity":
                    self.Sigma_gamma_true = np.identity(self.d)
                else:
                    self.Sigma_gamma_true  = np.identity(self.d) / self.d  
                    self.Sigma_gamma_true *= Sigma_gamma_factor
                self.prior_gamma_mu = self.mu_gamma_true
                self.prior_gamma_sigma = self.Sigma_gamma_true
            
            self.Xm_mu = np.zeros(dm)
            self.Xm_sigma = identity(dm) * Sigma_xm_factor
            self.Xk_mu = np.zeros(dk)
            self.Xk_sigma = identity(dk) * Sigma_xk_factor

        #self.mu_f = mu_f_factor
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma_factor
        self.cholesky_decomp = cholesky_decomp
        self.refresh_memory = refresh_memory
        self.refresh_threshold = refresh_threshold
        if (GP_prior_f_mu == None):
            def prior_f_mu(x): # input the true prior mean here # change to input
                #return x.sum(1)
                return (abs(x)).sum(1)
                #return 10*np.ones(np.shape(x)[0])
            self.prior_f_mu = prior_f_mu
        else:
            self.prior_f_mu = GP_prior_f_mu
            
        if self.env_setting == "real":    
            self.names = ["oracle-TS", "FA-ind", "OSFA", "LMM-FD_2", "GP-FD", "Hibou", "LMM-FG_2", "GP-FG",
                          "CB_base_RF", "CB_base_LR","LMM-FD_1", "LMM-FG_1", "NN-FD", "NN-FG"]
        else:
            self.names = ["oracle-TS", "OSFA", "FA-ind", "FD", "FG", "CB_base_"+str(self.augment_size1), 
                          "CB_base_"+str(self.augment_size2), "Hibou",
                         "FD-approx", "FG-approx"]#, "FD_fast", "FG_fast","FD-budget-ind"
    
    #############################################################################################################################################
    #############################################################################################################################################
    def determine_some_priors_4_Gaussian(self, K, N, M, n_rep):
        print("start prior cal")
        #sample_gammas = np.random.multivariate_normal(self.mu_gamma_true, self.Sigma_gamma_true, n_rep)
        sample_gammas = np.random.multivariate_normal(self.prior_gamma_mu, self.prior_gamma_sigma, n_rep)

        X_m = np.random.multivariate_normal(self.Xm_mu, self.Xm_sigma, (n_rep,1,M))
        X_k = np.random.multivariate_normal(self.Xk_mu, self.Xk_sigma, (n_rep,1,M * K))
        X_m = [np.repeat(X_m[i,],K*(N+1), axis = 1).reshape(M*K*(N+1),-1) for i in range(n_rep)]
        X_k = [np.repeat(X_k[i,],(N+1), axis = 1).reshape(M*K*(N+1),-1) for i in range(n_rep)]
        B = np.random.uniform(self.B_min, self.B_max, (n_rep, M))
        X_n = [np.array([np.repeat(b[i] * np.array(range(0, (N+1), 1)).reshape(1,-1)/N, K, axis = 0).reshape(-1,1) for i in range(M)]) for b in B]
        X_n = [x_n.reshape(M*K*(N+1),-1) for x_n in X_n]
        
            
        if self.log_log:
            X_m = [abs(x_m) for x_m in X_m]
            X_k = [abs(x_k) for x_k in X_k]
            X_n = [np.log(x_n + 1) for x_n in X_n]
        #else:
        #    X_n = [(X_k[i][:,-1].reshape(M*K*(N+1),-1)) * X_n[i] for i in range(n_rep)]
                
        if self.with_intercept:
            intercept = np.ones((n_rep, M*K*(N+1),1))
            Phi = [np.concatenate([intercept[i], X_m[i], X_k[i], X_n[i]], axis = 1).reshape((M,K*(N+1),-1)) for i in range(n_rep)]
        else:
            Phi = [np.concatenate([X_m[i], X_k[i], X_n[i]], axis = 1).reshape((M,K*(N+1),-1)) for i in range(n_rep)]
            
        theta_mean = [np.vstack([Phi_m[:,:-1].dot(sample_gammas[i][:-1]) for Phi_m in Phi[i]]) for i in range(n_rep)]
        E_cov_r1_gamma = mean([np.var(theta_mean[i]) for i in range(n_rep)])
        COV_e_r1_gamma = np.var([mean(theta_mean[i]) for i in range(n_rep)])
        #print(mean(theta_mean))
        self.prior_mean_powerlaw_coef = np.array([mean(theta_mean), self.prior_gamma_mu[-1]])
        #self.prior_mean_powerlaw_coef = np.array([1, .5])
        #print(E_cov_r1_gamma+COV_e_r1_gamma)
        self.prior_cov_powerlaw_coef = np.diag([E_cov_r1_gamma+COV_e_r1_gamma, self.prior_gamma_sigma[-1][-1]])
        #self.prior_cov_powerlaw_coef = np.diag([20, .5])
        #print(self.prior_mean_powerlaw_coef, self.prior_cov_powerlaw_coef)
        theta_mean = [theta_mean[i] + np.vstack([Phi_m[:,-1].dot(sample_gammas[i][-1]) for Phi_m in Phi[i]]) for i in range(n_rep)]
        
        #if log_log: log(y+1) > 0, Y = 0 if 0 budget
        theta_mean = np.array(theta_mean)
        theta_mean[theta_mean < 0] = 0
        theta_mean[:,:,list(range(0,K*(N+1), N+1))] = 0
        
        self.u_TS = mean(np.array(theta_mean))#mean(mean(mean(mean(np.array(theta_mean),axis=0), axis = 0).reshape(K,-1), axis = 0)
        self.u_TS = np.append(0,np.repeat(self.u_TS,N, axis = 0).reshape(-1))
        self.u_TS = self.u_TS.reshape(1,-1)
        self.u_TS = np.repeat(self.u_TS,K, axis = 0).reshape(-1)
        
        E_cov_r_gamma = np.mean([np.cov(theta_mean[i]) for i in range(n_rep)])#np.mean([np.cov(np.vstack(theta_mean[i].reshape(M,K,-1)).T) for i in range(n_rep)], axis = 0 )
        COV_e_r_gamma = np.cov(np.array([np.mean(theta_mean[i]) for i in range(n_rep)])) #np.cov(np.array([np.mean(np.vstack(theta_mean[i].reshape(M,K,-1)), axis = 0) for i in range(n_rep)]).T)
        self.cov_TS_diag = E_cov_r_gamma + COV_e_r_gamma
        #self.cov_TS_diag = np.diag(self.cov_TS_diag)
        self.cov_TS_diag = np.append(0,np.repeat(self.cov_TS_diag,N, axis = 0).reshape(-1))
        self.cov_TS_diag = self.cov_TS_diag .reshape(1,-1)
        self.cov_TS_diag = np.repeat(self.cov_TS_diag.reshape(1,-1),K, axis = 0).reshape(-1) 
        print('prior cal ends')
        del sample_gammas, X_m, X_k, B, X_n, Phi, theta_mean, E_cov_r_gamma, COV_e_r_gamma   
 #############################################################################################################################################
    def run_one_seed_Gaussian(self, seed):
        self.exp = Experiment(M = self.M, K = self.K, N = self.N, T = self.T, B_min = self.B_min, B_max = self.B_max, 
                              sigma_m = self.sigma_m, sigma_eps = self.sigma_eps, mu_gamma = self.mu_gamma_true, Sigma_gamma = self.Sigma_gamma_true,
                              with_intercept = self.with_intercept, dm = self.dm, Xm_mu = self.Xm_mu, Xm_sigma = self.Xm_sigma, 
                              dk = self.dk, Xk_mu = self.Xk_mu, Xk_sigma = self.Xk_sigma, seed = seed, order = self.order, env_setting = self.env_setting, 
                              log_log = self.log_log)
        ###################################### Priors ##############################################################
        self.determine_some_priors_4_Gaussian(self.K, self.N, self.M, n_rep = 500)
        TS = _agent_LMM_TS.TS_agent(K = self.K, N = self.N, u_prior_mean = self.u_TS, u_prior_cov_diag = self.cov_TS_diag, sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log) 
        N_TS = _agent_LMM_TS.N_TS_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = self.u_TS, u_prior_cov_diag = self.cov_TS_diag, sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log)
        ###########
        LB_agent = _agent_LMM_LB.LB_agent(M = self.M, K = self.K, N = self.N, prior_gamma_mu = self.prior_gamma_mu, prior_gamma_cov = self.prior_gamma_sigma, sigma = np.sqrt(self.sigma_eps**2+self.sigma_m**2), d = self.d, order = self.order, exp_episode = self.exp_episode, log_log = self.log_log, approximate = False)
        LB_approx_agent = _agent_LMM_LB.LB_agent(M = self.M, K = self.K, N = self.N, prior_gamma_mu = self.prior_gamma_mu, prior_gamma_cov = self.prior_gamma_sigma, sigma = np.sqrt(self.sigma_eps**2+self.sigma_m**2), d = self.d, order = self.order, exp_episode = self.exp_episode, log_log = self.log_log, approximate = True)

        MTB_agent = _agent_LMM_MTB.MTB_agent(sigma_2 = self.sigma_eps, gamma_prior_mean = self.prior_gamma_mu, gamma_prior_cov = self.prior_gamma_sigma, sigma_1 = self.sigma_m, M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, order = self.order, exp_episode = self.exp_episode, log_log = self.log_log, approximate = False)
        MTB_approx_agent = _agent_LMM_MTB.MTB_agent(sigma_2 = self.sigma_eps, gamma_prior_mean = self.prior_gamma_mu, gamma_prior_cov = self.prior_gamma_sigma, sigma_1 = self.sigma_m, M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, order = self.order, exp_episode = self.exp_episode, log_log = self.log_log, approximate = True)
        
        meta_oracle = _agent_LMM_TS.meta_oracle_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = self.exp.env.theta_mean, u_prior_cov_diag = [self.sigma_m ** 2 * np.ones(self.K * (self.N + 1)) for _ in range(self.M)], sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log)
        
        global_model = LinearRegression()
        CB_agent_1 = CB.Contextual_Bandit_agent(global_model = global_model, sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N,
                                                prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef, prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef,
                                                Xs = self.exp.env.Phi, with_intercept = self.with_intercept, order = self.order, augment_size = self.augment_size1, 
                                                B = self.exp.env.B, true_gamma = self.exp.env.gamma, log_log = self.log_log, true_prior_mean_powerlaw_coef = self.exp.env.true_prior_mean)
        CB_agent_2 = CB.Contextual_Bandit_agent(global_model = global_model, sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N,
                                                prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef, prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef,
                                                Xs = self.exp.env.Phi, with_intercept = self.with_intercept, order = self.order, augment_size = self.augment_size2, 
                                                B = self.exp.env.B, true_gamma = self.exp.env.gamma,log_log = self.log_log, true_prior_mean_powerlaw_coef = self.exp.env.true_prior_mean)
            
        Hibou_agent = Hibou.Hibou_agent(M = self.M, K = self.K, N = self.N, B = self.exp.env.B)
                                                                                                
        agents = {"oracle-TS" : meta_oracle,
                  "OSFA" : TS,
                  "FA-ind" : N_TS, 
                  #"FD-budget-ind":LB_budget_ind,
                  "FD" : LB_agent,
                  "FD-approx" : LB_approx_agent,
                  "FG": MTB_agent,
                  "FG-approx": MTB_approx_agent,
                  "CB_base_"+str(self.augment_size1): CB_agent_1,
                  "CB_base_"+str(self.augment_size2): CB_agent_2,
                  "Hibou": Hibou_agent
                }
        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        return self.exp.record
    #############################################################################################################################################
    def run_one_seed_NN(self, seed):
        self.exp = Experiment(M = self.M, K = self.K, N = self.N, T = self.T, B_min = self.B_min, B_max = self.B_max, 
                              sigma_m = self.sigma_m, sigma_eps = self.sigma_eps, mu_gamma = self.mu_gamma_true, Sigma_gamma = self.Sigma_gamma_true,
                              with_intercept = self.with_intercept, dm = self.dm, Xm_mu = self.Xm_mu, Xm_sigma = self.Xm_sigma, 
                              dk = self.dk, Xk_mu = self.Xk_mu, Xk_sigma = self.Xk_sigma, seed = seed, order = self.order, env_setting = self.env_setting )
        ###################################### Priors ##############################################################
        meta_oracle = _agent_LMM_TS.meta_oracle_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = self.exp.env.theta_mean, u_prior_cov_diag = [self.sigma_m ** 2 * np.ones(self.K * (self.N + 1)) for _ in range(self.M)], sigma = self.sigma_eps)
        TS = _agent_LMM_TS.TS_agent(K = self.K, N = self.N,u_prior_mean = np.zeros(self.K*(self.N+1)), u_prior_cov_diag = np.ones(self.K*(self.N+1))*100, sigma = self.sigma_eps) 
        N_TS = _agent_LMM_TS.N_TS_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = np.zeros(self.K*(self.N+1)), u_prior_cov_diag = np.ones(self.K*(self.N+1))*100, sigma = self.sigma_eps)
        ###########
        NN_init_est_1 = _agent_NN_FD.MeanEstimatorWithBias(self.d, m = self.NN_m, L = self.NN_L)
        LB_agent = _agent_NN_FD.NeuralTS_agent(M = self.M, K = self.K, N = self.N, T = self.T, m = self.NN_m, nu = self.NN_nu*np.sqrt(self.sigma_eps**2+self.sigma_m**2), estimator = NN_init_est_1, reg = self.NN_reg, order = self.order, exp_episode = self.exp_episode, approximate = False, B_max = self.B_max, normalize = True, lr = self.lr)
        NN_init_est_2 = _agent_NN_FD.MeanEstimatorWithBias(self.d, m = self.NN_m, L = self.NN_L)
        LB_agent_approx = _agent_NN_FD.NeuralTS_agent(M = self.M, K = self.K, N = self.N, T = self.T, m = self.NN_m, nu = self.NN_nu*np.sqrt(self.sigma_eps**2+self.sigma_m**2), estimator = NN_init_est_2, reg = self.NN_reg, order = self.order, exp_episode = self.exp_episode, approximate = True, B_max = self.B_max, normalize = True, lr = self.lr)
        
        NN_init_est_3 = _agent_NN_MTB.MeanEstimatorWithBias(self.d, m = self.NN_m, L = self.NN_L)                                                                      
        MTB_agent =  _agent_NN_MTB.NeuralTS_agent(M = self.M, K = self.K, N = self.N, T = self.T, m = self.NN_m, nu = self.NN_nu, estimator = NN_init_est_3, reg = self.NN_reg, order = self.order, exp_episode = self.exp_episode, approximate = False, B_max = self.B_max, sigma_m = self.sigma_m, sigma_eps = self.sigma_eps, normalize = True, lr = self.lr)
        NN_init_est_4 = _agent_NN_MTB.MeanEstimatorWithBias(self.d, m = self.NN_m, L = self.NN_L)    
        MTB_agent_approx =  _agent_NN_MTB.NeuralTS_agent(M = self.M, K = self.K, N = self.N, T = self.T, m = self.NN_m, nu = self.NN_nu, estimator = NN_init_est_4, reg = self.NN_reg, order = self.order, exp_episode = self.exp_episode, approximate = True, B_max = self.B_max, sigma_m = self.sigma_m, sigma_eps = self.sigma_eps, normalize = True, lr = self.lr)
        
        self.prior_mean_powerlaw_coef = np.array([0, 0])
        self.prior_cov_powerlaw_coef = np.diag([100, 100])
        CB_agent_1 = CB.Contextual_Bandit_agent(global_model = RandomForestRegressor(), sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N, prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef,
                                                prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef, Xs = self.exp.env.Phi, 
                                                with_intercept = self.with_intercept, order = self.order, 
                                                augment_size = self.augment_size1, B = self.exp.env.B, log_log = self.log_log)
        ####################################################################################################
        Hibou_agent = Hibou.Hibou_agent(M = self.M, K = self.K, N = self.N, B = self.exp.env.B)
                                                                                                
        agents = {"oracle-TS" : meta_oracle,
                  "OSFA" : TS,
                  "FA-ind" : N_TS, 
                  #"FD-budget-ind":LB_budget_ind,
                  "FD" : LB_agent,
                  "FD-approx" : LB_agent_approx,
                  "FG": MTB_agent,
                  "FG-approx": MTB_agent_approx,
                  "CB_base_"+str(self.augment_size1): CB_agent_1,
                  #"CB_base_"+str(self.augment_size2): CB_agent_2,
                  "Hibou": Hibou_agent
                }
        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        return self.exp.record
#############################################################################################################################################
    #############################################################################################################################################
    def determine_some_priors_4_GP(self, d, K, N, sigma_m, seed):
        
        B = 200 # number of iterations in Bootstrap   

        # generate random sample with size B=1000 directly from the environment 
        # here using seed = 0 for simplicity and replicability
        def prior_f_mu(x):  # same as the environment setup
            #return x.sum(1)
            return 0*(abs(x)).sum(2)
        
        print("prior calculation starts...")
        self.env_GP_bootstrap = _env_GP.Semi_env_GP(M = max(B,int(1500/K)), K = K, N = N, T = self.T, 
                                                    B_min = self.B_min, B_max = self.B_max, sigma_m = sigma_m, 
                                                    sigma_eps = self.sigma_eps, kernel_gamma = self.kernel_gamma,
                                                    with_intercept = self.with_intercept, dm = self.dm, Xm_mu = self.Xm_mu,
                                                    Xm_sigma = self.Xm_sigma, dk = self.dk, 
                                                    Xk_mu = self.Xk_mu, Xk_sigma = self.Xk_sigma, seed = seed, prior_simulate = True, prior_f_mu = prior_f_mu)
        All_theta = self.env_GP_bootstrap.theta
        
        self.env_GP_bootstrap = None
        
        self.u_TS = np.mean(All_theta, 0)  
        self.cov_TS_diag = np.var(All_theta, 0)

        del All_theta
        print("prior calculation ends...")
            
 #############################################################################################################################################
    def run_one_seed_GP(self, seed):

        self.exp = Experiment(M = self.M, K = self.K, N = self.N, T = self.T, B_min = self.B_min, B_max = self.B_max, 
                              sigma_m = self.sigma_m, sigma_eps = self.sigma_eps, kernel_gamma = self.kernel_gamma,
                              with_intercept = self.with_intercept, dm = self.dm, Xm_mu = self.Xm_mu, Xm_sigma = self.Xm_sigma, 
                              mu_gamma = self.mu_gamma_true, Sigma_gamma = self.Sigma_gamma_true,
                              dk = self.dk, Xk_mu = self.Xk_mu, Xk_sigma = self.Xk_sigma, seed = seed, order = self.order,
                              env_setting = self.env_setting, prior_f_mu = self.prior_f_mu_env)
        
        
        ###################################### Priors ##############################################################
        #self.determine_some_priors_4_GP(self.d, self.K, self.N, self.sigma_m, seed)
        
        #TS = _agent_LMM_TS.TS_agent(K = self.K, N = self.N, u_prior_mean = self.u_TS, u_prior_cov_diag = self.cov_TS_diag, sigma = self.sigma_eps, exp_episode = self.exp_episode) 
        #N_TS = _agent_LMM_TS.N_TS_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = self.u_TS, u_prior_cov_diag = self.cov_TS_diag, sigma = self.sigma_eps, exp_episode = self.exp_episode)
        
        TS = _agent_LMM_TS.TS_agent(K = self.K, N = self.N, u_prior_mean = np.zeros(self.K*(self.N+1)), u_prior_cov_diag = np.ones(self.K*(self.N+1))*100, sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log) 
        N_TS = _agent_LMM_TS.N_TS_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = np.zeros(self.K*(self.N+1)), u_prior_cov_diag = np.ones(self.K*(self.N+1))*100, sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log)
                      
        meta_oracle = _agent_LMM_TS.meta_oracle_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = self.exp.env.theta_mean, u_prior_cov_diag = [self.sigma_m ** 2 * np.ones(self.K * (self.N + 1)) for _ in range(self.M)], sigma = self.sigma_eps, exp_episode = self.exp_episode)

        def GP_prior_f_mu(x):
            return np.zeros(np.shape(x)[0])
        LB_agent = _agent_GP_FD_fast.GP_agent(M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, prior_f_mu = GP_prior_f_mu, kernel = self.kernel, kernel_gamma = self.kernel_gamma, sigma = np.sqrt(self.sigma_eps**2+self.sigma_m**2), exp_episode = self.exp_episode, cholesky_decomp = self.cholesky_decomp, order = self.order, refresh_memory = self.refresh_memory, refresh_threshold = self.refresh_threshold, approximate = False, B_max = self.B_max, with_intercept = self.with_intercept)
        LB_agent_approx = _agent_GP_FD_fast.GP_agent(M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, prior_f_mu = GP_prior_f_mu, kernel = self.kernel, kernel_gamma = self.kernel_gamma, sigma = np.sqrt(self.sigma_eps**2+self.sigma_m**2), exp_episode = self.exp_episode, cholesky_decomp = self.cholesky_decomp, order = self.order, refresh_memory = self.refresh_memory, refresh_threshold = self.refresh_threshold, approximate = True, B_max = self.B_max, with_intercept = self.with_intercept)

        MTB_agent = _agent_GP_MTB_fast.GP_MTB_agent(sigma_2 = self.sigma_eps, prior_f_mu = GP_prior_f_mu, kernel = self.kernel, kernel_gamma = self.kernel_gamma, sigma_1 = self.sigma_m, M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, cholesky_decomp = self.cholesky_decomp, order = self.order, exp_episode = self.exp_episode, refresh_memory = self.refresh_memory, refresh_threshold = self.refresh_threshold, approximate = False, B_max = self.B_max, with_intercept = self.with_intercept)
        MTB_agent_approx = _agent_GP_MTB_fast.GP_MTB_agent(sigma_2 = self.sigma_eps, prior_f_mu = GP_prior_f_mu, kernel = self.kernel, kernel_gamma = self.kernel_gamma, sigma_1 = self.sigma_m, M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, cholesky_decomp = self.cholesky_decomp, order = self.order, exp_episode = self.exp_episode, refresh_memory = self.refresh_memory, refresh_threshold = self.refresh_threshold, approximate = True, B_max = self.B_max, with_intercept = self.with_intercept)

        #################################################################################################### ####################################################################################################
        self.prior_mean_powerlaw_coef = np.array([0, 0])
        self.prior_cov_powerlaw_coef = np.diag([100, 100])
        CB_agent_1 = CB.Contextual_Bandit_agent(global_model = RandomForestRegressor(), sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N, prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef,
                                                prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef, Xs = self.exp.env.Phi, 
                                                with_intercept = self.with_intercept, order = self.order, 
                                                augment_size = self.augment_size1, B = self.exp.env.B, log_log = self.log_log)
        CB_agent_2 = CB.Contextual_Bandit_agent(global_model = RandomForestRegressor(), sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N, prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef,
                                                prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef, Xs = self.exp.env.Phi, 
                                                with_intercept = self.with_intercept, order = self.order, 
                                                augment_size = self.augment_size2, B = self.exp.env.B, log_log = self.log_log)
            
        Hibou_agent = Hibou.Hibou_agent(M = self.M, K = self.K, N = self.N, B = self.exp.env.B)
                                                                                                
        agents = {"oracle-TS" : meta_oracle,
                  "OSFA" : TS,
                  "FA-ind" : N_TS, 
                  #"FD-budget-ind":LB_budget_ind,
                  #"FD" : LB_agent,
                  "FD-approx" : LB_agent_approx,
                  #"FG": MTB_agent,
                  "FG-approx": MTB_agent_approx,
                  "CB_base_"+str(self.augment_size1): CB_agent_1,
                  #"CB_base_"+str(self.augment_size2): CB_agent_2,
                  "Hibou": Hibou_agent
                }
        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        return self.exp.record    
    
############################################################################################################################3
    def run_one_seed_real(self, seed):
        self.exp = Experiment(M = self.M, K = self.K, N = self.N, T = self.T, B_min = self.B_min, B_max = self.B_max, 
                              sigma_m = self.sigma_m, sigma_eps = self.sigma_eps,
                              with_intercept = self.with_intercept, seed = seed, order = self.order, env_setting = self.env_setting, 
                              log_log = self.log_log)
        ###################################### Priors ##############################################################
        TS = _agent_LMM_TS.TS_agent(K = self.K, N = self.N, u_prior_mean = np.zeros(self.K*(self.N+1)), u_prior_cov_diag = np.ones(self.K*(self.N+1))*100, sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log) 
        N_TS = _agent_LMM_TS.N_TS_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = np.zeros(self.K*(self.N+1)), u_prior_cov_diag = np.ones(self.K*(self.N+1))*100, sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log)
        ###########
        LB_agent_1 = _agent_LMM_LB.LB_agent(M = self.M, K = self.K, N = self.N, prior_gamma_mu = np.zeros(int(self.exp.env.d)+7), prior_gamma_cov = np.diag(np.ones(int(self.exp.env.d)+7)*20), sigma = np.sqrt(self.sigma_eps**2+self.sigma_m**2), d = self.exp.env.d, order = self.order, exp_episode = self.exp_episode, log_log = self.log_log, real = True, transfer_x_type = 1, approximate = True)
        
        MTB_agent_1 = _agent_LMM_MTB.MTB_agent(sigma_2 = self.sigma_eps, gamma_prior_mean = np.zeros(int(self.exp.env.d)+7), gamma_prior_cov = np.diag(np.ones(int(self.exp.env.d)+7)*20), sigma_1 = self.sigma_m, M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, order = self.order, exp_episode = self.exp_episode, log_log = self.log_log, real = True, transfer_x_type = 1, approximate = True)
        meta_oracle = _agent_LMM_TS.meta_oracle_agent(M = self.M, K = self.K, N = self.N, u_prior_mean = self.exp.env.theta_mean, u_prior_cov_diag = [self.sigma_m ** 2 * np.ones(self.K * (self.N + 1)) for _ in range(self.M)], sigma = self.sigma_eps, exp_episode = self.exp_episode, log_log = self.log_log)
        
        self.prior_mean_powerlaw_coef = np.array([0, 0])
        self.prior_cov_powerlaw_coef = np.diag([20, 20])
        CB_agent_1 = CB.Contextual_Bandit_agent(global_model = RandomForestRegressor(), sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N, prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef,
                                                prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef, Xs = self.exp.env.Phi, 
                                                with_intercept = self.with_intercept, order = self.order, 
                                                augment_size = self.augment_size1, B = self.exp.env.B, log_log = self.log_log)
        CB_agent_2 = CB.Contextual_Bandit_agent(global_model = LinearRegression(), sigma_eps = np.sqrt(self.sigma_eps**2+self.sigma_m**2),
                                                M = self.M, K = self.K, N = self.N, prior_mean_powerlaw_coef = self.prior_mean_powerlaw_coef,
                                                prior_cov_powerlaw_coef = self.prior_cov_powerlaw_coef, Xs = self.exp.env.Phi, 
                                                with_intercept = self.with_intercept, order = self.order, augment_size = self.augment_size1, 
                                                B = self.exp.env.B, log_log = self.log_log)
            
        Hibou_agent = Hibou.Hibou_agent(M = self.M, K = self.K, N = self.N, B = self.exp.env.B)
        
        def GP_prior_f_mu(x):
            return np.zeros(np.shape(x)[0])

        MTB_agent_GP = _agent_GP_MTB_fast.GP_MTB_agent(sigma_2 = self.sigma_eps, prior_f_mu = GP_prior_f_mu, kernel = self.kernel, kernel_gamma = self.kernel_gamma, sigma_1 = self.sigma_m, M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, cholesky_decomp = self.cholesky_decomp, order = self.order, exp_episode = self.exp_episode, refresh_memory = self.refresh_memory, refresh_threshold = self.refresh_threshold, log_log = self.log_log, approximate = True, B_max = self.B_max, with_intercept = self.with_intercept)
        
        LB_agent_GP = _agent_GP_FD_fast.GP_agent(M = self.M, K = self.K, N = self.N, Xs = self.exp.Xs, prior_f_mu = GP_prior_f_mu, kernel = self.kernel, kernel_gamma = self.kernel_gamma, sigma = np.sqrt(self.sigma_eps**2+self.sigma_m**2), exp_episode = self.exp_episode, cholesky_decomp = self.cholesky_decomp, order = self.order, refresh_memory = self.refresh_memory, refresh_threshold = self.refresh_threshold, log_log = self.log_log, approximate = True, B_max = self.B_max, with_intercept = self.with_intercept)
        
        NN_init_est = _agent_NN_FD.MeanEstimatorWithBias(self.exp.env.d, m = self.NN_m, L = self.NN_L)
        LB_agent_NN = _agent_NN_FD.NeuralTS_agent(M = self.M, K = self.K, N = self.N, T = self.T, m = self.NN_m, nu = self.NN_nu*np.sqrt(self.sigma_eps**2+self.sigma_m**2), estimator = NN_init_est, reg = self.NN_reg, order = self.order, exp_episode = self.exp_episode, approximate = False, B_max = self.B_max, log_log = self.log_log, normalize = True)

        NN_init_est_2 = _agent_NN_MTB.MeanEstimatorWithBias(self.exp.env.d, m = self.NN_m, L = self.NN_L)
        MTB_agent_NN =  _agent_NN_MTB.NeuralTS_agent(M = self.M, K = self.K, N = self.N, T = self.T, m = self.NN_m, nu = self.NN_nu, estimator = NN_init_est_2, reg = self.NN_reg, order = self.order, exp_episode = self.exp_episode, approximate = False, B_max = self.B_max, sigma_m = self.sigma_m, sigma_eps = self.sigma_eps, log_log = self.log_log, normalize = True)
        agents = {
                 "oracle-TS" : meta_oracle,              
                  #"OSFA" : TS,
                  "FA-ind" : N_TS, 
                  "LMM-FD_1": LB_agent_1,
                  "LMM-FG_1": MTB_agent_1,  
                  "Hibou": Hibou_agent,
                  "GP-FG": MTB_agent_GP,
                  "GP-FD": LB_agent_GP,
                  "CB_base_RF": CB_agent_1,
                  "CB_base_LR": CB_agent_2,
                  "NN-FD":LB_agent_NN,
                  "NN-FG":MTB_agent_NN
                }

        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        return self.exp.record
       #############################################################################################################################################
    
    def run_multiple_parallel(self, reps, batch = 1):
        rep = reps // batch
        with open('log/{}.txt'.format(self.date_time), 'w') as f:
            print(self.title_settting, file=f)

        record = []
        for b in range(batch):
            print("batch = {}".format(b))
            if self.env_setting == "LMM":
                r = parmap(self.run_one_seed_Gaussian, range(rep * b, rep * b + rep))
            elif self.env_setting == "NN":
                r = parmap(self.run_one_seed_NN, range(rep * b, rep * b + rep))
            elif self.env_setting == "GP":
                r = parmap(self.run_one_seed_GP, range(rep * b, rep * b + rep))
            elif self.env_setting == "real":
                r = parmap(self.run_one_seed_real, range(rep * b, rep * b + rep))
            record += r
        self.record = record
        
    #############################################################################################################################################
    #############################################################################################################################################
    def plot_regret(self, skip_methods = []
                    , ci = None, freq = 20
                   , plot_mean = False, skip = 0
                   , y_min = None, y_max = None, log_R = False):

        reps = len(self.record)
        
        data = pd.concat([self.record[seed]['cum_regret_df'] for seed in range(reps)])
        data_meta = pd.concat([self.record[seed]['cum_meta_regret_df'] for seed in range(reps)])
        data_plot_meta = data_meta[data_meta.method != "oracle-TS"]
        
        data_R = pd.concat([self.record[seed]['cum_R_df'] for seed in range(reps)])
        data_exp_R = pd.concat([self.record[seed]['cum_exp_R_df'] for seed in range(reps)])


        if self.setting['order'] == "sequential":
            data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['T']) + self.setting['T'] - 1]   
            data_plot_BR.time = np.tile(np.arange(0, self.setting['M'])
                                        , len(data_plot_BR) // self.setting['M'])
            data_plot_meta = data_plot_meta.iloc[np.arange(0, len(data_plot_meta), step = self.setting['T']) + self.setting['T'] - 1] 
            data_plot_meta.time = np.tile(np.arange(0, self.setting['M'])
                                        , len(data_plot_meta) // self.setting['M'])
            data_plot_R = data_R.iloc[np.arange(0, len(data_R), step = self.setting['T']) + self.setting['T'] - 1]   
            data_plot_R.time = np.tile(np.arange(0, self.setting['M'])
                                        , len(data_plot_R) // self.setting['M'])
            data_plot_exp_R = data_exp_R.iloc[np.arange(0, len(data_exp_R), step = self.setting['T']) + self.setting['T'] - 1]   
            data_plot_exp_R.time = np.tile(np.arange(0, self.setting['M'])
                                        , len(data_plot_exp_R) // self.setting['M'])


        elif self.setting['order'] == "concurrent":
            data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['M']) + self.setting['M'] - 1]   
            data_plot_BR.time = np.tile(np.arange(0, self.setting['T'])
                                        , len(data_plot_BR) // self.setting['T'])
            data_plot_meta = data_plot_meta.iloc[np.arange(0, len(data_plot_meta), step = self.setting['M']) + self.setting['M'] - 1] 
            data_plot_meta.time = np.tile(np.arange(0, self.setting['T'])
                                        , len(data_plot_meta) // self.setting['T'])
            data_plot_R = data_R.iloc[np.arange(0, len(data_R), step = self.setting['M']) + self.setting['M'] - 1]   
            data_plot_R.time = np.tile(np.arange(0, self.setting['T'])
                                        , len(data_plot_R) // self.setting['T'])
            data_plot_exp_R = data_exp_R.iloc[np.arange(0, len(data_exp_R), step = self.setting['M']) + self.setting['M'] - 1]   
            data_plot_exp_R.time = np.tile(np.arange(0, self.setting['T'])
                                        , len(data_plot_exp_R) // self.setting['T'])

        self.data_plot_BR_original = data_plot_BR.copy()
        self.data_plot_meta_original = data_plot_meta.copy()
        self.data_plot_R_original = data_plot_R.copy()
        self.data_plot_exp_R_original = data_plot_exp_R.copy()

        if plot_mean:
            data_plot_BR.regret = data_plot_BR.regret / (data_plot_BR.time + 1)
            data_plot_meta.regret = data_plot_meta.regret / (data_plot_meta.time + 1)

        if skip_methods is not None:
            for met in skip_methods:
                data_plot_BR = data_plot_BR[data_plot_BR.method != met]
                data_plot_meta = data_plot_meta[data_plot_meta.method != met]

        data_plot_BR = data_plot_BR[data_plot_BR.time >= skip]
        data_plot_meta = data_plot_meta[data_plot_meta.time >= skip]
        data_plot_BR = data_plot_BR.reset_index()
        data_plot_meta = data_plot_meta.reset_index()
        
        if log_R:
            data_plot_BR['regret'] = np.log(data_plot_BR['regret'] + 1e-6)
            data_plot_meta['regret'] = np.log(data_plot_meta['regret'] + 1e-6)

    #############################################################################################################################################
        #############################################################################################################################################
    def save(self, main_path = "res/", fig_path = "fig/", sub_folder = [], no_care_keys = []
            , only_plot_matrix = 1):
        """
        Since all results together seems quite large
        a['record'][0].keys() = (['R', 'A', 'regret', 'meta_regret', 'cum_regret', 'cum_meta_regret'])

        regret / R can almost derive anything, except for A

        The only thing is that, we may need to re-read them. Probably we need a function, to convert a "skim recorder" to a "full recorder", when we do analysis.
        """
        ########################################################
        date = get_date()

        result_path = main_path + date
        fig_path = fig_path + date
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        ########################################################
        aa = self.env_setting
        fig_path += "/" + aa 
        result_path += "/" + aa 
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        ########################################################
        if len(sub_folder) > 0:
            fig_path += "/" 
            result_path += "/"
            for key in sub_folder:
                fig_path += ("_" + str(key) + str(self.setting[key]))
                result_path += ("_" + str(key) + str(self.setting[key]))
                no_care_keys.append(key)
        no_care_keys.append('save_prefix')
        if self.env_setting == "real":
            no_care_keys.append("sigma_m")
            no_care_keys.append("sigma_eps")
            no_care_keys.append("B_min")
            no_care_keys.append("B_max")
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        ############################
        if only_plot_matrix:
            record_R_only = {"data_plot_BR_original" : self.data_plot_BR_original,
                    "data_plot_meta_original" : self.data_plot_meta_original,
                    "data_plot_R_original" : self.data_plot_R_original,
                    "data_plot_exp_R_original" : self.data_plot_exp_R_original}
        else:
            record_R_only = {seed : self.record[seed]['R'] for seed in range(len(self.record))}

        r = {"setting" : self.setting
             , "record" : record_R_only
            , "name" : self.names}

        ############################
        path_settting = "_".join([str(key) + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float] and key not in no_care_keys])
        print(path_settting)
        if self.save_prefix:
            path_settting = path_settting + "-" + self.save_prefix

        ############################
        r_path = result_path + "/"  + path_settting
        fig_path = fig_path + "/"  + path_settting + ".pdf"
        print("save to {}".format(r_path))
        #self.fig.savefig(fig_path)
        dump(r,  r_path)