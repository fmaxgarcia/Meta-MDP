
import numpy as np 
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
import time

from sys import stdout
# from Features import fourier_basis
import copy
import random
from tqdm import tqdm



ACTION_DISCRETE = 0
ACTION_CONTINUOUS = 1



class Actor(torch.nn.Module):
    def __init__(self, num_features, num_actions, action_type, distribution):
        super(Actor, self).__init__()
        self.action_type = action_type
        self.distribution = distribution
        self.linear = torch.nn.Linear(num_features, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 32, bias=True)

        if action_type == ACTION_DISCRETE:
            self.linear3 = torch.nn.Linear(32, num_actions, bias=True)
        else:
            self.linear_param1 = torch.nn.Linear(32, num_actions, bias=True)
            self.linear_param2 = torch.nn.Linear(32, num_actions, bias=True)
            
        self.num_actions = num_actions


    def forward(self, x):
        x = self.linear( x )
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)

        if self.action_type == ACTION_DISCRETE:
            x = self.linear3(x)
            x = x - torch.max( x )
            x = F.softmax( x )
            return x
        else:
            x1 = self.linear_param1( x )
            x2 = self.linear_param2( x )
            if self.distribution == "normal":
                return F.sigmoid( x1 ), F.sigmoid( x2 )
            
            elif self.distribution == "beta":
                return F.softplus( x1 ), F.softplus( x2 )
            else:
                assert(False)


    def get_distribution(self, params1, params2):
        if self.distribution == "normal":
            dist = Normal( params1, params2)
        elif self.distribution == "beta":
            dist = Beta(params1, params2)

        return dist

    def select_action(self, phi):

        x = torch.from_numpy(phi).float()

        # print(softmax)
        if self.action_type == ACTION_CONTINUOUS:
            param1, param2 = self.forward( x )
            param1 = param1.view(1,-1)
            param2 = param2.view(1,-1)

            dist = self.get_distribution(param1, param2)
            action = dist.sample()[0]
            action = action.detach().numpy()
        elif self.action_type == ACTION_DISCRETE:
            action = self.forward( x )

            action = action.detach().numpy()
            action = np.random.choice(range(action.shape[0]), p=action)
        else:
            assert(False)
            
        return action



class Critic(torch.nn.Module):
    def __init__(self, num_features):
        super(Critic, self).__init__()
        self.linear = torch.nn.Linear(num_features, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 32, bias=True)
        self.linear3 = torch.nn.Linear(32, 1, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)

        return x

def get_phi(env, state):
        st = state.reshape( (state.shape[0],) )
        return st #/ 100.0
        st = (st - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) 
        return st

        return phi



   
        
class PPO():
    
    GPU = False
    
    
    def __init__(self, env, alpha, beta, gamma, t_length, update_steps, buff_size=500, action_type=ACTION_DISCRETE, distribution="beta", verbose=True):
        # super(ActorCriticPytorch, self).__init__()
        # print(env.env.action_space.__dict__)
        # print(env.env.observation_space.__dict__)


        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.lamba_ = 0.95
        self.env = env.env

        self.action_type = ACTION_DISCRETE if hasattr(env.env.action_space, 'n') else ACTION_CONTINUOUS
        self.num_actions = env.env.action_space.n if hasattr(env.env.action_space, 'n') else env.env.action_space.low.shape[0]
        self.distribution = distribution

        if self.action_type == ACTION_CONTINUOUS:
            self.torch_high = torch.from_numpy(env.env.action_space.high)
            self.torch_low = torch.from_numpy(env.env.action_space.low)
        
        self.num_features = get_phi(env, env.reset()).shape[0]
        self.epochs = 4

        self.max_grad_norm = 2.0
        

        self.buffer_size = buff_size
        self.batch_size = 8
        self.buffer = []
        self.clear_history()
        self.t_length = t_length
        self.update_steps = update_steps


        self.critic = Critic(self.num_features).cuda() if self.GPU == True else Critic(self.num_features)
            
        self.actor_act = Actor(self.num_features, self.num_actions, self.action_type, distribution)
        self.actor_update = self.actor_act
        if self.GPU:
            self.actor_update = Actor(self.num_features, self.num_actions, self.action_type, distribution)
            self.actor_update.load_state_dict(self.actor_act.state_dict())
            self.actor_update = self.actor_update.cuda()

        self.actor_old = Actor(self.num_features, self.num_actions, self.action_type, distribution).cuda() if self.GPU == True else Actor(self.num_features, self.num_actions, self.action_type, distribution)

        self.optimizer_actor = torch.optim.Adam(self.actor_update.parameters(), lr=self.alpha)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.beta)

        self.reset()



    def clear_history(self):
        self.hist_s, self.hist_sn, self.hist_a, self.hist_r, self.hist_term = None, None, None, None, None


    def reset(self):
        self.clear_history()
        self.step = 0
    
    def add_hist_to_buffer(self):
        if self.hist_s is None or self.hist_sn is None or self.hist_r is None or self.hist_a is None or self.hist_term is None:
            return

        try:
            x = (self.hist_s.copy(), self.hist_sn.copy(), self.hist_r.copy(), self.hist_a.copy(), self.hist_term.copy())
        except:
            print(x[0])
            print(x[1])
            print(x[2])
            print(x[3])
            print(x[4])

        sample_num = x[0].shape[0]
        if x[1].shape[0] != sample_num or x[2].shape[0] != sample_num or x[3].shape[0] != sample_num or x[4].shape[0] != sample_num:
            return
        
        x = (torch.from_numpy(x[0]).float(), torch.from_numpy(x[1]).float(), torch.from_numpy(x[2]).float(),
            torch.from_numpy(x[3]), torch.from_numpy(x[4]).float())

        if self.GPU:
            x = (x[0].cuda(), x[1].cuda(), x[2].cuda(), x[3].cuda(), x[4].cuda())

        if len(self.buffer) == self.buffer_size:
            self.buffer.pop( 0 )

        self.buffer.append( x )
        


    def add_to_history(self, tup):
        s, sn, r, a, term = tup
        if self.hist_s is None:
            self.hist_s = s.reshape(1,-1)
            self.hist_sn = sn.reshape(1,-1)
            self.hist_a = a
            self.hist_r = r
            self.hist_term = term
        else:
            try:
                self.hist_s = np.vstack( (self.hist_s, s))
                self.hist_sn = np.vstack( (self.hist_sn, sn))
                self.hist_a = np.vstack( (self.hist_a, a))
                self.hist_r = np.vstack( (self.hist_r, r))
                self.hist_term = np.vstack( (self.hist_term, term))
            except:
                print(self.hist_s.shape)
                print(self.hist_sn.shape)
                print(self.hist_a.shape)
                print(self.hist_r.shape)
                print(self.hist_term.shape)


    def update(self, state, action, reward, next_state, next_action, done, optimize=True):
        self.step += 1

        phi = get_phi(self.env, state.reshape(-1,1)).reshape(1,-1)
        phi_n = get_phi(self.env, next_state.reshape(-1,1)).reshape(1,-1)
        r = np.array([[reward]])
        a = np.array([[action]]) if self.action_type == ACTION_DISCRETE else np.array([action])
        nonterm = np.array([[0]]) if done else np.array([[1]])
        self.add_to_history( (phi, phi_n, r, a, nonterm) )
        

        if self.hist_s is not None and (self.hist_s.shape[0] == self.t_length or done):
            self.add_hist_to_buffer( )
            self.clear_history()

        if optimize:
            if self.step % self.update_steps == 0 or done:
                self.optimize( verbose=False )


    def select_action(self, state):
        phi = get_phi(self.env, state.reshape(-1,1)).T
        action = self.actor_update.select_action( phi )
        return action




    def compute_loss_terms(self, samples):
        Ls = torch.zeros( (self.t_length, len(samples))) 
        vfs = torch.zeros( (self.t_length, len(samples))) 
        ents = torch.zeros( (self.t_length, len(samples))) 

        for s_index, hist in enumerate(samples):
            s, sn, r, a, nonterm = hist

            vn = self.critic.forward( sn )
            v = self.critic.forward( s )


            deltas = (r + (nonterm * self.gamma * vn)) - v

            A = torch.zeros((s.shape[0], 1))

            for i in range(s.shape[0]):
                A[i] = sum([ (self.gamma*self.lamba_)**(k-i) * deltas[k] for k in range(i, s.shape[0])])
                 
            if self.action_type == ACTION_DISCRETE: 
                p = self.actor_update.forward( s )
                p_o = self.actor_old.forward( s )

                R = torch.exp( torch.log(p.gather(1, a)) - torch.log(p_o.gather(1, a)) )
                H = -torch.sum( p * torch.log(p), dim=1).view(-1,1)
            else:
                p1, p2 = self.actor_update.forward( s )
                p_o1, p_o2 = self.actor_old.forward( s )

                dist = self.actor_update.get_distribution(p1, p2)
                dist_old = self.actor_update.get_distribution(p_o1, p_o2)
                

                R = torch.exp(dist.log_prob(a) - dist_old.log_prob(a))
                H = torch.sum( dist.entropy(), dim=1 ).view(-1,1)

            e = 0.2
            l1 = R*A 
            l2 = torch.clamp(R, 1-e, 1+e) * A

            L = torch.mean( torch.min(l1, l2), dim=1 )
            
            for t_index in range(L.shape[0]):
                Ls[t_index, s_index] = L[t_index]

            for t_index in range(deltas.shape[0]):
                vfs[t_index, s_index] = deltas[t_index]**2

            for t_index in range(H.shape[0]):
                ents[t_index, s_index] = torch.mean(H[t_index])


        L = torch.sum( torch.mean(Ls, dim=1) )
        vf = torch.sum( torch.mean(vfs, dim=1) )
        ent = 0.1 * torch.sum( torch.mean(ents, dim=1) )
        

        return L, vf, ent


    
    def optimize(self, verbose):
        k = self.batch_size if self.batch_size < len(self.buffer) else len(self.buffer)
        if k < 1:
            return

        # self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        self.actor_old.load_state_dict(self.actor_update.state_dict())
        for _ in range(self.epochs):
            samples = random.sample( self.buffer, k=k-1)
            samples.append( self.buffer[-1] )
            L, vf, ent = self.compute_loss_terms(samples)

            
            loss = L - vf + ent
            loss = -loss
            if verbose:
                print("Total loss: %f - L: %f - delta: %f - entropy: %f"
                    %(loss, L, vf, ent))
            


            self.optimizer_actor.zero_grad()
            actor_loss = -(L+ent)
            actor_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm(self.actor_update.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            vf.backward()
            # torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

            #loss.backward()


            # self.optimizer_actor_critic.step()
        
        if self.GPU == True:
            self.actor_act.load_state_dict(self.actor_update.state_dict())



    def save_model(self, path, i):
        torch.save(self.actor_update, path+"actor_%f_%f_%f_%d.pt" %(self.alpha, self.beta, self.gamma, i))
        critic = self.critic.cpu() if self.GPU == True else self.critic
        
        torch.save(critic, path+"critic_%f_%f_%f_%d.pt" %(self.alpha, self.beta, self.gamma, i))

    
    def load_model(self, actor, critic):
        actor_dict = torch.load(actor)
        critic_dict = torch.load(critic)
        print(actor_dict)
        print(critic_dict)

        for k, p in enumerate(list(self.actor_update.parameters())):
            p.data = list(actor_dict.parameters())[k].data 

        for k, p in enumerate(list(self.critic.parameters())):
            p.data = list(critic_dict.parameters())[k].data 
