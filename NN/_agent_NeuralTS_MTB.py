from _util import *
from _optimizer import *
import numpy as np
import numpy.linalg as la
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Neural network for estimating the mean reward of each arm 
class MeanEstimator(nn.Module):
    def __init__(self,d,m,L):
        super().__init__()
        torch.manual_seed(42)
        self.d=d
        self.m=m
        self.L=L
        assert m%2==0,"m should be pair !"
        assert d%2==0,"d should be pair !"
        
        self.modules = [nn.Linear(d,m,bias=False),nn.ReLU()]
        
       
        for i in range (1,L-1):
            self.modules.append(nn.Linear(m,m,bias=False))
            self.modules.append(nn.ReLU())
       
        self.modules.append(nn.Linear(m,1,bias=False))
        self.modules.append(nn.ReLU())
        

        self.sequential = nn.ModuleList(self.modules)
        
    def init_weights(self):
        first_init=np.sqrt(4/self.m)*torch.randn((self.m,(self.d//2)-1)).to(device)
        first_init=torch.cat([first_init,torch.zeros(self.m,1).to(device),torch.zeros(self.m,1).to(device),first_init],axis=1)
        self.sequential[0].weight.data=first_init
         
        for i in range(2,self.L-1):
            if i%2==0:
                init=np.sqrt(4/self.m)*torch.randn((self.m,(self.m//2)-1)).to(device)
            self.sequential[i].weight.data=torch.cat([init,torch.zeros(self.m,1).to(device),torch.zeros(self.m,1).to(device),init],axis=1)
                 
        last_init=np.sqrt(2/self.m)*torch.randn(1,self.m//2).to(device)
        self.sequential[-2].weight.data=torch.cat([last_init,-last_init],axis=1)
        
    def forward(self,x):
        x=x
        # Pass the input tensor through each of our operations
        for layer in self.sequential:
            x = layer(x)
        return np.sqrt(self.m)*x
    
#Same architecture like before but using a bias
class MeanEstimatorWithBias(nn.Module):
    def __init__(self,d,m,L):
        super().__init__()
        torch.manual_seed(42)
        self.d=d
        self.m=m
        self.L=L
        assert m%2==0,"m should be pair !"
        assert d%2==0,"d should be pair !"
        
        self.modules = [nn.Linear(d,m),nn.ReLU()]
        
       
        for i in range (1,L-1):
            self.modules.append(nn.Linear(m,m))
            self.modules.append(nn.ReLU())
            
       
        self.modules.append(nn.Linear(m,1))
        self.modules.append(nn.ReLU())
        

        self.sequential = nn.ModuleList(self.modules)
      
        
    def init_weights(self):
         first_init=np.sqrt(4/self.m)*torch.randn((self.m,(self.d//2)-1)).to(device)
         first_init=torch.cat([first_init,torch.zeros(self.m,1).to(device),torch.zeros(self.m,1).to(device),first_init],axis=1)
         self.sequential[0].weight.data=first_init
         
         for i in range(2,self.L-1):
                if i%2==0:
                  init=np.sqrt(4/self.m)*torch.randn((self.m,(self.m//2)-1)).to(device)
                  self.sequential[i].weight.data=torch.cat([init,torch.zeros(self.m,1).to(device),torch.zeros(self.m,1).to(device),init],axis=1)
                 
         last_init=np.sqrt(2/self.m)*torch.randn(1,self.m//2).to(device)
         self.sequential[-2].weight.data=torch.cat([last_init,-last_init],axis=1)
        
        
    def forward(self,x):
        x=x
        # Pass the input tensor through each of our operations
        for layer in self.sequential:
            x = layer(x)
        return np.sqrt(self.m)*x
        
    
## flatten a large tuple containing tensors of different sizes
def flatten(tensor):
    T=torch.tensor([]).to(device)
    for element in tensor:
        T=torch.cat([T,element.to(device).flatten()])
    return T
    
#concatenation of all the parameters of a NN
def get_theta(model):
    return flatten(model.parameters())

#loss function of the neural TS
def criterion(estimated_reward,reward,m,reg,theta,theta_0):
    return 0.5*torch.sum(torch.square(estimated_reward-reward))+0.5*m*reg*torch.square(torch.norm(theta-theta_0))

#make the transformation of the context vectors so that we met the assumptions of the authors
def transform(x):
    return np.vstack([x/(np.sqrt(2)*la.norm(x)),x/(np.sqrt(2)*la.norm(x))]).reshape(-1)

#generation of context vectors
def generate_contexts(K,d):
    #Generation of normalized features - ||x|| = 1 and x symmetric
    X = torch.randn((K,d//2))
    X=torch.Tensor([transform(x) for x in X])
    return X

class NeuralTS_agent():
    
    '''
    oracle: u_prior_mean and u_prior_cov are the true values
    TS: u_prior_mean and u_prior_cov are the random values
    naive pooling, ignoring the difference between adlines across different campaigns
    '''
    def __init__(self,M,K,N,T,m,nu=1,estimator=None,criterion=criterion,reg=1,feature_transform=False,sigma=None,order="concurrent",exp_episode=2,sigma_eps=None,sigma_m=None,approximate=False, B_max = None, log_log = False, normalize = True, lr = 10**(-4)):
        # self.features=X.to(device)
        self.M, self.K, self.N, self.T = M, K, N, T
        self.seed = 100
        self.timer = 0
        self.reg=reg
        self.nu=nu
        self.m = m
        self.sigma=sigma
        self.sigma_eps = sigma_eps
        self.sigma_m = sigma_m
        self.estimator=estimator
        self.estimator.to(device)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr = lr)
        self.current_loss=0
        self.criterion=criterion
        self.feature_transform = feature_transform
        self.order=order
        self.exp_episode = exp_episode
        self.approximate = approximate
        self.B_max = B_max
        self.normalize = normalize
        self.log_log = log_log
        self.clear()
    
    def clear(self):
        # initialize the design matrix, its inverse, 
        # the vectors containing the arms chosen and the rewards obtained
        self.start=1
        self.estimator.init_weights()
        self.theta_zero=get_theta(self.estimator)
        self.p=(self.theta_zero).size(0)
        self.Design = torch.Tensor(self.reg*np.eye(self.p)).to(device)
        self.DesignInv = torch.Tensor((1/self.reg)*np.eye(self.p)).to(device)
        self.NumRecords = np.zeros((self.M, self.K*(self.N+1)))
        self.RewardRecords = np.zeros((self.M, self.K*(self.N+1)))
        self.ActionRecords=[]
        self.ChosenArms=[]
        self.rewards=[]
        
    def take_action(self, i, t, X):
        torch.manual_seed(self.seed)
        self.seed += 1
        Rs=[]
        if self.normalize:
            X = X.copy()
            if self.log_log:
                X[:,-1] /= np.log(self.B_max+1)
            else:
                X[:,-1] /= self.B_max
        if self.feature_transform:
          X = np.apply_along_axis(transform, axis=1, arr=X)
        features = torch.from_numpy(X).to(device)
        features = features.to(torch.float)
        #print(features)
        
        for k in range(self.K*(self.N+1)):
            f=self.estimator(features[k])
            g=torch.autograd.grad(outputs=f,inputs=self.estimator.parameters())
            g=flatten(g).detach()
            start=time.time()
            sigma_squared=(self.reg*(1/self.m)*torch.matmul(torch.matmul(g.T,self.DesignInv),g)).to(device)
            sigma=torch.sqrt(sigma_squared)
            r=(self.nu)*(sigma)*torch.randn(1).to(device)+f.detach()
            n_t = self.NumRecords[i][k]
            sum_r_t = self.RewardRecords[i][k]
            scale_t = 1/np.square(self.sigma_m)+ n_t/np.square(self.sigma_eps)
            sigma_t = np.sqrt(1 / scale_t)
            r_tilda = (r/np.square(self.sigma_m) + sum_r_t/np.square(self.sigma_eps))/scale_t+(sigma_t)*torch.randn(1).to(device)
            Rs.append(r_tilda.detach().item())

        Rs = np.array(Rs)
        if self.log_log:
            Rs = np.exp(Rs)-1
        #print(Rs)
        arm_to_pull=self._optimize(Rs, t)
        #print(arm_to_pull)
        self.NumRecords[i][arm_to_pull] += 1
        self.ActionRecords.extend(arm_to_pull)
        #print(self.ActionRecords)
        #print(self.NumRecords)
        self.ChosenArms.append(features[arm_to_pull].tolist())
        #print((np.array(arm_to_pull)%(self.N+1))/self.N)
        return arm_to_pull
        
    def receive_reward(self, i, t, A, obs_R, X):
        """
        arm is the subset selected in round t
        reward is the corresponding reward for each selected (m,k)
        """
        if self.normalize:
            X = X.copy()
            if self.log_log:
                X[:,-1] /= np.log(self.B_max+1)
            else:
                X[:,-1] /= self.B_max
        current_time = now()
        self.rewards.append(obs_R)
        rewards = np.concatenate(self.rewards).ravel()
        self.RewardRecords[i][A] += obs_R
        #torch.autograd.set_detect_anomaly(True)
        
        if self.order == "concurrent" and self.approximate == True and i == self.M-1:
          estimated_rewards=self.estimator(torch.Tensor(self.ChosenArms).to(device)).view(-1)
          if self.feature_transform:
            X = np.apply_along_axis(transform, axis=1, arr=X)
          features = torch.from_numpy(X).to(device)
          features = features.to(torch.float)
          self.current_loss=self.criterion(estimated_rewards.to(device),torch.Tensor(torch.Tensor(rewards)).to(device),self.m,self.reg,get_theta(self.estimator),self.theta_zero)
          
          #gradient descent
          if self.start==1:
              self.current_loss.backward(retain_graph=True)    
          else:
              self.current_loss.backward()
              
          self.optimizer.step() 
          self.optimizer.zero_grad() 
  
          f_t = self.estimator(torch.Tensor(self.ChosenArms).to(device)).reshape(-1)
          f_t_batch = f_t[(t*self.M*self.K):((t+1)*self.M*self.K)]
          
          g = [torch.zeros_like(param) for param in self.estimator.parameters()]
          for l in range(0, self.M):
            for k in range(0, self.K):
              # Compute gradients for f_t[k][0] with respect to the model parameters
              act = self.ActionRecords[(l+t*self.M)*self.K+k]
              n_t = self.NumRecords[l][act]
              w = self.sigma_eps/np.sqrt(np.square(self.sigma_eps)+np.square(self.sigma_m)*n_t)/np.sqrt(np.square(self.sigma_eps)+np.square(self.sigma_m)*(n_t-1))
              #print(n_t,w,k+l*self.K)
              gradients = torch.autograd.grad(outputs=f_t_batch[k+l*self.K], inputs=self.estimator.parameters(), retain_graph=True)
              gradients_w = [w * grad for grad in gradients]
              # Accumulate the gradients
              g = [sum_grad + grad for sum_grad, grad in zip(g, gradients_w)]

          g=flatten(g)
          g=g/(np.sqrt(self.m))
          
          self.Design+=torch.matmul(g,g.T).to(device)
          #self.DesignInv=torch.inverse(torch.diag(torch.diag(self.Design))) #approximation proposed by the authors
          self.DesignInv=torch.diag(1.0 / torch.diag(self.Design))
          self.start+=1

        if self.order == "sequential" and self.approximate == True and t == self.T-1:
          estimated_rewards=self.estimator(torch.Tensor(self.ChosenArms).to(device)).view(-1)
          if self.feature_transform:
            X = np.apply_along_axis(transform, axis=1, arr=X)
          features = torch.from_numpy(X).to(device)
          features = features.to(torch.float)
          self.current_loss=self.criterion(estimated_rewards.to(device),torch.Tensor(torch.Tensor(rewards)).to(device),self.m,self.reg,get_theta(self.estimator),self.theta_zero)
          
          #gradient descent
          if self.start==1:
              self.current_loss.backward(retain_graph=True)    
          else:
              self.current_loss.backward()
              
          self.optimizer.step() 
          self.optimizer.zero_grad() 

          f_t = self.estimator(torch.Tensor(self.ChosenArms).to(device)).reshape(-1)
          f_t_batch = f_t[(i*self.T*self.K):((i+1)*self.T*self.K)]
          
          g = [torch.zeros_like(param) for param in self.estimator.parameters()]
          for l in range(0, self.T):
            for k in range(0, self.K):
              # Compute gradients for f_t[k][0] with respect to the model parameters
              act = self.ActionRecords[(l+i*self.T)*self.K+k]
              n_t = self.NumRecords[i][act]
              w = self.sigma_eps/np.sqrt(np.square(self.sigma_eps)+np.square(self.sigma_m)*n_t)/np.sqrt(np.square(self.sigma_eps)+np.square(self.sigma_m)*(n_t-1))
              #print(n_t,w,k+l*self.K)
              gradients = torch.autograd.grad(outputs=f_t_batch[k+l*self.K], inputs=self.estimator.parameters(), retain_graph=True)
              gradients_w = [w * grad for grad in gradients]
              # Accumulate the gradients
              g = [sum_grad + grad for sum_grad, grad in zip(g, gradients_w)]
        
          g=flatten(g)
          g=g/(np.sqrt(self.m))
          
          self.Design+=torch.matmul(g,g.T).to(device)
          #self.DesignInv=torch.inverse(torch.diag(torch.diag(self.Design))) #approximation proposed by the authors
          self.DesignInv=torch.diag(1.0 / torch.diag(self.Design))
          self.start+=1
        
        if self.approximate == False:
          estimated_rewards=self.estimator(torch.Tensor(self.ChosenArms).to(device)).view(-1)
          if self.feature_transform:
            X = np.apply_along_axis(transform, axis=1, arr=X)
          features = torch.from_numpy(X).to(device)
          features = features.to(torch.float)
          self.current_loss=self.criterion(estimated_rewards.to(device),torch.Tensor(torch.Tensor(rewards)).to(device),self.m,self.reg,get_theta(self.estimator),self.theta_zero)
          
          #gradient descent
          if self.start==1:
              self.current_loss.backward(retain_graph=True)    
          else:
              self.current_loss.backward()
              
          self.optimizer.step() 
          self.optimizer.zero_grad() 
            
          f_t_batch=self.estimator(features[A])
          
          g = [torch.zeros_like(param) for param in self.estimator.parameters()]
          for k in range(0, self.K):
            # Compute gradients for f_t[k][0] with respect to the model parameters
            #act = self.ActionRecords[(i+t*self.M)*self.K+k]
            act = A[k]
            n_t = self.NumRecords[i][act]
            w = self.sigma_eps/np.sqrt(np.square(self.sigma_eps)+np.square(self.sigma_m)*n_t)/np.sqrt(np.square(self.sigma_eps)+np.square(self.sigma_m)*(n_t-1))
            gradients = torch.autograd.grad(outputs=f_t_batch[k], inputs=self.estimator.parameters(), retain_graph=True)
            gradients_w = [w * grad for grad in gradients]
            # Accumulate the gradients
            g = [sum_grad + grad for sum_grad, grad in zip(g, gradients_w)]

          g=flatten(g)
          g=g/(np.sqrt(self.m))
          
          self.Design+=torch.matmul(g,g.T).to(device)
          #self.DesignInv=torch.inverse(torch.diag(torch.diag(self.Design))) #approximation proposed by the authors
          self.DesignInv=torch.diag(1.0 / torch.diag(self.Design))
          self.start+=1

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