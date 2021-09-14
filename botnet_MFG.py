import numpy as np
from numpy import random as rnd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sys import exit

T = 100
N_s = 1000
delta_t = T/N_s

beta_UU = .3
beta_UD = .4
beta_DU = .3
beta_DD = .4

q_rec_D = .5
q_rec_U = .4
q_inf_D = .4
q_inf_U = .3

k_D = .3
k_I = .5

v_H = .6

lambd = .8

# number of episodes
episodes = 10000

# initial distributions and Q-table
mu = np.zeros((episodes,N_s,4))
nu = np.zeros((episodes,N_s,4,2))
mu_0 = np.array([.25,.25,.25,.25])
nu_0 = np.zeros((4,2)) + .125
Q = np.zeros((episodes,N_s,4,2))

# counter for each time state-action pair is visited at time t_n for use in calculating learning rate for Q
sa_counter = np.zeros((4,2,N_s))

# exploration rate for e-greedy
epsilon = .1
gamma = .5
omega_Q = .55
assert omega_Q >.5
assert omega_Q <= 1
omega_nu = .85
tol_nu = .0005
tol_Q = .0005

# will be used for analysis of learning rates
rho_Q_plot = np.zeros((episodes, N_s))
rho_nu_plot = np.zeros((episodes, N_s))


# states:
# 0 = DI
# 1 = DS
# 2 = UI
# 3 = US
def transition_rate(state, action, mu):
    
    Q_matrix = np.empty((4,4))
    
    if action == 0:
        Q_matrix[0] = np.array([-q_rec_D, q_rec_D, 0, 0])
        Q_matrix[1] = np.array([v_H*q_inf_D + beta_DD*mu[0] + beta_UD*mu[2], -v_H*q_inf_D - beta_DD*mu[0] - beta_UD*mu[2], 0, 0])
        Q_matrix[2] = np.array([0, 0, -q_rec_U, q_rec_U])
        Q_matrix[3] = np.array([0, 0, v_H*q_inf_U + beta_UU*mu[2] + beta_DU*mu[0], -v_H*q_inf_U - beta_UU*mu[2] - beta_DU*mu[0]])
    elif action == 1:
        Q_matrix[0] = np.array([-q_rec_D - lambd, q_rec_D, lambd, 0])
        Q_matrix[1] = np.array([v_H*q_inf_D + beta_DD*mu[0] + beta_UD*mu[2], -v_H*q_inf_D - beta_DD*mu[0] - beta_UD*mu[2] - lambd, 0, lambd])
        Q_matrix[2] = np.array([lambd, 0, -q_rec_U - lambd, q_rec_U])
        Q_matrix[3] = np.array([0, lambd, v_H*q_inf_U + beta_UU*mu[2] + beta_DU*mu[0], -v_H*q_inf_U - beta_UU*mu[2] - beta_DU*mu[0] - lambd])
    
    return Q_matrix[state]


for k in tqdm(range(episodes)):
    for tau in range(N_s):
        if tau == 0:
            # choose a random state if is the first time step of the episode
            X = rnd.choice(np.arange(4), p = mu_0)
            state_trans = np.zeros(4)
            state_trans[X] = 1.
        if k == 0:
            # if it is the first episode, choose action at random and use initial state-action distribution
            A = rnd.choice([0,1])
            v = nu_0
        else:
            # give optimal action for given Q table
            min_A = np.argmin(Q[k-1][tau][X])
            # use epsilon-greedy criterion for action selection
            if rnd.rand() > epsilon:
                A = min_A
            else:
                A = rnd.choice([0,1])
            # set state action distribution for update
            v = nu[k-1][tau]
        # increment counter for state-action-t_n visits by 1 for calculating learning rate for Q
        sa_counter[X][A][tau] += 1
        
        # calculate learning rates. if the MFG condition (rho_Q > rho_nu) does not hold, exit and prompt user to change
        # rho_Q = 1/(1 + N_s * sa_counter[X][A][tau])**omega_Q
        rho_Q = 1/(1 + sa_counter[X][A][tau])**omega_Q
        rho_nu = 1/(1 + (k+1))**omega_nu
        if rho_Q < rho_nu:
            print('\nMFG condition does not hold, adjust omegas such that rho_Q > rho_nu:')
            print('rho_Q: ', rho_Q)
            print('rho_nu: ', rho_nu)
            while len(tqdm._instances) > 0:
                tqdm._instances.pop().close()
            exit()
        rho_Q_plot[k][tau] = rho_Q
        rho_nu_plot[k][tau] = rho_nu
        
        # update state-action distribution
        dirac = np.zeros((4,2))
        dirac[X][A] = 1
        nu[k][tau] = v + rho_nu * (dirac - v)
        # update state distribution
        mu[k][tau] = nu[k][tau].sum(axis=1)
        
        cost = delta_t * (bool(X < 2) * k_D + bool(X==0 | X==2) * k_I)
        
        state_trans += transition_rate(X, A, mu[k][tau]) * delta_t
        state_trans = np.where(state_trans < 0, 0, state_trans)
        # print('\ntau: {} X: {} A: {}'.format(tau, X, A))
        # print(state_trans)
        
        full_trans = np.count_nonzero(state_trans >= 1)
        if full_trans == 0:
            X_prime = X
            
        elif full_trans == 1:
            X_prime = np.argmax(state_trans)
            state_trans[X] = 0
            
            # if a new state is reached, we preserve infection/recovery/progress towards D/S switching
            if X == 0 and X_prime == 1:
                state_trans[3] = state_trans[2]
                state_trans[2] = 0
            elif X == 0 and X_prime == 2:
                state_trans[3] = state_trans[1]
                state_trans[1] = 0
            elif X == 1 and X_prime == 0:
                state_trans[2] = state_trans[3]
                state_trans[3] = 0
            elif X == 1 and X_prime == 3:
                state_trans[2] = state_trans[0]
                state_trans[0] = 0
            elif X == 2 and X_prime == 0:
                state_trans[1] = state_trans[3]
                state_trans[3] = 0
            elif X == 2 and X_prime == 3:
                state_trans[1] = state_trans[0]
                state_trans[0] = 0
            elif X == 3 and X_prime == 1:
                state_trans[0] = state_trans[2]
                state_trans[2] = 0
            elif X == 3 and X_prime == 2:
                state_trans[0] = state_trans[1]
                state_trans[1] = 0
            
            state_trans[X_prime] = 1.
            
        elif full_trans == 2:
            # if full transition is reached for both states (due to lack of granularity of timestep),
            # it means the agent has completed U/D switch and also has become either infected or susceptible
            # so set the next state according to that
            state_trans = np.zeros(4)
            if X == 0:
                state_trans[3] = 1.
                X_prime = 3
            elif X == 1:
                state_trans[2] = 1.
                X_prime = 2
            elif X == 2:
                state_trans[1] = 1.
                X_prime = 1
            elif X == 3:
                state_trans[0] = 1.
                X_prime = 0
                
        elif full_trans > 2:
            print('\nWhat should we do here?')
            print('tau: ', tau)
            print('X:', X)
            print('A:', A)
            print(state_trans)
            while len(tqdm._instances) > 0:
                tqdm._instances.pop().close()
            exit()
        else:
            print('\nstate transition vector is weird:')
            print(state_trans)
            while len(tqdm._instances) > 0:
                tqdm._instances.pop().close()
            exit()
        
        # update Q
        if k == 0:
            q = np.zeros((4,2))
            q_prime = 0
        else:
            q = Q[k-1][tau]
            if tau < N_s - 1:
                q_prime = np.min(Q[k-1][tau+1][X_prime])
            else:
                q_prime = 0
        B = cost + gamma * q_prime
        Q[k][tau] = q
        Q[k][tau][X][A] = q[X][A] + rho_Q * (B - q[X][A])
        
        # set next state
        X = X_prime
    
    # after episode has finished, check for convergence, else continue
    if k > 0:
        if abs(nu[k][tau] - nu[k-1][tau]).sum() <= tol_nu and abs(Q[k][tau] - Q[k-1][tau]).sum() <= tol_Q:
            print('\nSuccess: Algorithm has converged after ', k+1, ' episodes.')
            print('Final average state distribution: ', np.mean(mu[k,:,:], axis=0))
            plt.figure(0)
            plt.plot(np.mean(mu[:k+1,:,:], axis=1))
            plt.title('Average state distribution mu for each episode\nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
            plt.xlabel('Episode')
            plt.ylabel('mu')
            plt.legend(['DI','DS','UI','US'])
            plt.show()
            
            plt.figure(1)
            plt.plot(np.mean(Q[:k+1,:,0,:], axis=1))
            plt.title('Average Q values for state DI \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
            plt.xlabel('Episode')
            plt.ylabel('Q value')
            plt.legend(['Action: 0','Action: 1'])
            plt.show()
            
            plt.figure(2)
            plt.plot(np.mean(Q[:k+1,:,1,:], axis=1))
            plt.title('Average Q values for state DS \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
            plt.xlabel('Episode')
            plt.ylabel('Q value')
            plt.legend(['Action: 0','Action: 1'])
            plt.show()
            
            plt.figure(3)
            plt.plot(np.mean(Q[:k+1,:,2,:], axis=1))
            plt.title('Average Q values for state UI \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
            plt.xlabel('Episode')
            plt.ylabel('Q value')
            plt.legend(['Action: 0','Action: 1'])
            plt.show()
            
            plt.figure(4)
            plt.plot(np.mean(Q[:k+1,:,3,:], axis=1))
            plt.title('Average Q values for state US \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
            plt.xlabel('Episode')
            plt.ylabel('Q value')
            plt.legend(['Action: 0','Action: 1'])
            plt.show()
            
            plt.figure(5)
            plt.plot(np.mean(rho_nu_plot[:k+1], axis=1))
            plt.plot(np.mean(rho_Q_plot[:k+1], axis=1))
            plt.title('Average Learning Rates\nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
            plt.xlabel('Episode')
            plt.ylabel('Learning Rate')
            plt.legend(['rho_nu','rho_Q'])
            plt.show()
            while len(tqdm._instances) > 0:
                tqdm._instances.pop().close()
            exit()
        elif k == episodes - 1:
            print('\nFailure: Algorithm failed to converge. Adjust parameters or increase number of episodes.')

# plot the average state distributions for each episodes
plt.figure(0)
plt.plot(np.mean(mu[:k+1,:,:], axis=1))
plt.title('Average state distribution mu for each episode\nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
plt.xlabel('Episode')
plt.ylabel('mu')
plt.legend(['DI','DS','UI','US'])
plt.show()

plt.figure(1)
plt.plot(np.mean(Q[:k+1,:,0,:], axis=1))
plt.title('Average Q values for state DI \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
plt.xlabel('Episode')
plt.ylabel('Q value')
plt.legend(['Action: 0','Action: 1'])
plt.show()

plt.figure(2)
plt.plot(np.mean(Q[:k+1,:,1,:], axis=1))
plt.title('Average Q values for state DS \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
plt.xlabel('Episode')
plt.ylabel('Q value')
plt.legend(['Action: 0','Action: 1'])
plt.show()

plt.figure(3)
plt.plot(np.mean(Q[:k+1,:,2,:], axis=1))
plt.title('Average Q values for state UI \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
plt.xlabel('Episode')
plt.ylabel('Q value')
plt.legend(['Action: 0','Action: 1'])
plt.show()

plt.figure(4)
plt.plot(np.mean(Q[:k+1,:,3,:], axis=1))
plt.title('Average Q values for state US \nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
plt.xlabel('Episode')
plt.ylabel('Q value')
plt.legend(['Action: 0','Action: 1'])
plt.show()

plt.figure(5)
plt.plot(np.mean(rho_nu_plot[:k+1], axis=1))
plt.plot(np.mean(rho_Q_plot[:k+1], axis=1))
plt.title('Average Learning Rates\nepsilon = {}, gamma = {}\nT = {}, N = {}'.format(epsilon, gamma, T, N_s))
plt.xlabel('Episode')
plt.ylabel('Learning Rate')
plt.legend(['rho_nu','rho_Q'])
plt.show()

# plot the evolution of the state distribution in the final episode
# plt.plot(mu[k,:,:])
# plt.title('State distribution mu over final episode')
# plt.xlabel('Time')
# plt.ylabel('mu')
# plt.legend(['DI','DS','UI','US'])
# plt.show()

while len(tqdm._instances) > 0:
    tqdm._instances.pop().close()
