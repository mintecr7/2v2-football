"""
Main training script for the class-project 2v2 football setup.

This file couples together:
- Unity environment bootstrap
- observation stitching
- model creation and checkpoint loading
- PPO-style training
- a rough post-training evaluation loop

Naming note:
- `g_brain_name` and `b_brain_name` are legacy names for the two Unity
  behaviors exposed by the environment.
- In practice, this script treats those behaviors as the two teams.
- Goalie versus striker is decided by row index within each behavior.
"""

from mlagents_envs.environment import UnityEnvironment
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import tensorflow as tf

from model import ActorModel, CriticModel
from memory import Memory

from agent import Agent
from optimizer import Optimizer

from time import perf_counter 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Environment bootstrap and behavior discovery
# ---------------------------------------------------------------------------

env = UnityEnvironment(file_name="CoE202")

# The Unity build exposes two behaviors. The original variable names refer to
# "brains", but operationally this script uses them as the two teams.
env_info = env.reset()
g_brain_name =list(env.behavior_specs)[0]
g_brain =list(env.behavior_specs)[0]
print(list(env.behavior_specs)[1])
b_brain_name = list(env.behavior_specs)[1]

b_brain =list(env.behavior_specs)[1]


# Each team behavior is assumed to contain two controlled players.
n_goalie_agents = 2
#len(env_info[g_brain_name].agents)
print('Number of goalie agents:', n_goalie_agents)
n_striker_agents =2
# len(env_info[s_brain_name].agents)
print('Number of striker agents:', n_striker_agents)

# number of actions
goalie_action_size = 3
# g_brain.vector_action_space_size
print('Number of goalie actions:', goalie_action_size)
striker_action_size = 3
# s_brain.vector_action_space_size
print('Number of striker actions:', striker_action_size)

# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------
#
# Each Unity behavior returns two observation tensors per player. The script
# concatenates those two tensors to form one flat state per player, then
# stacks players by role: both goalies together and both strikers together.

decision_steps_p, terminal_steps_p = env.get_steps(g_brain_name)
decision_steps_b, terminal_steps_b = env.get_steps(b_brain_name)

# Team 0 goalie / team 1 goalie
cur_obs_g1 = np.transpose(np.concatenate((decision_steps_p.obs[0][0,:], decision_steps_p.obs[1][0,:])))
cur_obs_g2 = np.transpose(np.concatenate((decision_steps_b.obs[0][0,:], decision_steps_b.obs[1][0,:])))
cur_obs_g= np.vstack((cur_obs_g1, cur_obs_g2))
# Team 0 striker / team 1 striker
cur_obs_s1 =  np.transpose(np.concatenate((decision_steps_p.obs[0][1,:], decision_steps_p.obs[1][1,:]) ))
cur_obs_s2 =  np.transpose(np.concatenate((decision_steps_b.obs[0][1,:], decision_steps_b.obs[1][1,:]) ))
cur_obs_s=np.vstack((cur_obs_s1, cur_obs_s2))

goalie_states = cur_obs_g
print(decision_steps_p.obs[0][0,:].shape)
#env_info[g_brain_name].vector_observations
goalie_state_size = goalie_states.shape[1]
##print(goalie_state_size)
print('There are {} goalie agents. Each receives a state with length: {}'.format(goalie_states.shape[0], goalie_state_size))
striker_states = cur_obs_s
#env_info[s_brain_name].vector_observations
striker_state_size = striker_states.shape[1]
print('There are {} striker agents. Each receives a state with length: {}'.format(striker_states.shape[0], striker_state_size))


# ---------------------------------------------------------------------------
# Hyperparameters and checkpoint paths
# ---------------------------------------------------------------------------

N_STEP = 8
BATCH_SIZE = 32
GAMMA = 0.995
EPSILON = 0.1
ENTROPY_WEIGHT = 0.001
GRADIENT_CLIP = 0.5
GOALIE_LR = 8e-5
STRIKER_LR = 1e-4


CHECKPOINT_GOALIE_ACTOR = './checkpoint_goalie_actor.pth'
CHECKPOINT_GOALIE_CRITIC = './checkpoint_goalie_critic.pth'
CHECKPOINT_STRIKER_ACTOR = './checkpoint_striker_actor.pth'
CHECKPOINT_STRIKER_CRITIC = './checkpoint_striker_critic.pth'

# Actors and Critics
GOALIE_0_KEY = 0
STRIKER_0_KEY = 0
GOALIE_1_KEY = 1
STRIKER_1_KEY = 1

# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------
#
# Actors consume role-local state only. Critics consume the concatenated
# global state of all four players.

goalie_actor_model = ActorModel( goalie_state_size, goalie_action_size ).to(DEVICE)
goalie_critic_model = CriticModel( goalie_state_size + striker_state_size + goalie_state_size + striker_state_size ).to(DEVICE)
goalie_optim = optim.Adam( list( goalie_actor_model.parameters() ) + list( goalie_critic_model.parameters() ), lr=GOALIE_LR )
# self.optim = optim.RMSprop( list( self.actor_model.parameters() ) + list( self.critic_model.parameters() ), lr=lr, alpha=0.99, eps=1e-5 )


striker_actor_model = ActorModel( striker_state_size, striker_action_size ).to(DEVICE)
striker_critic_model = CriticModel( striker_state_size + goalie_state_size + striker_state_size + goalie_state_size ).to(DEVICE)
striker_optim = optim.Adam( list( striker_actor_model.parameters() ) + list( striker_critic_model.parameters() ), lr=STRIKER_LR )
# self.optim = optim.RMSprop( list( self.actor_model.parameters() ) + list( self.critic_model.parameters() ), lr=lr, alpha=0.99, eps=1e-5 )

goalie_actor_model.load( CHECKPOINT_GOALIE_ACTOR )
goalie_critic_model.load( CHECKPOINT_GOALIE_CRITIC )
striker_actor_model.load( CHECKPOINT_STRIKER_ACTOR )
striker_critic_model.load( CHECKPOINT_STRIKER_CRITIC )


# ---------------------------------------------------------------------------
# Agent and optimizer wrappers
# ---------------------------------------------------------------------------

goalie_0 = Agent( DEVICE, GOALIE_0_KEY, goalie_actor_model, N_STEP )
goalie_optimizer = Optimizer( DEVICE, goalie_actor_model, goalie_critic_model, goalie_optim,  
    N_STEP, BATCH_SIZE, GAMMA, EPSILON, ENTROPY_WEIGHT, GRADIENT_CLIP)

striker_0 = Agent( DEVICE, STRIKER_0_KEY, striker_actor_model, N_STEP )
striker_optimizer = Optimizer( DEVICE, striker_actor_model, striker_critic_model, striker_optim,  
    N_STEP, BATCH_SIZE, GAMMA, EPSILON, ENTROPY_WEIGHT, GRADIENT_CLIP)

def ppo_train():
    """Train the shared goalie and striker policies against a random opponent."""
    n_episodes = 5000
    team_0_window_score = deque(maxlen=100)
    team_0_window_score_wins = deque(maxlen=100)

    team_1_window_score = deque(maxlen=100)
    team_1_window_score_wins = deque(maxlen=100)

    draws = deque(maxlen=100)

    for episode in range(n_episodes):
        # Reset the environment and rebuild the four player states.
        env_info = env.reset()                        # reset the environment    
        decision_steps_p, terminal_steps_p = env.get_steps(g_brain_name)
        decision_steps_b, terminal_steps_b = env.get_steps(b_brain_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent
        cur_obs_g1 = np.transpose(np.concatenate((decision_steps_p.obs[0][0,:], decision_steps_p.obs[1][0,:])))
        cur_obs_g2 = np.transpose(np.concatenate((decision_steps_b.obs[0][0,:], decision_steps_b.obs[1][0,:])))
        cur_obs_g= np.vstack((cur_obs_g1, cur_obs_g2))
        cur_obs_s1 =  np.transpose(np.concatenate((decision_steps_p.obs[0][1,:], decision_steps_p.obs[1][1,:]) ))
        cur_obs_s2 =  np.transpose(np.concatenate((decision_steps_b.obs[0][1,:], decision_steps_b.obs[1][1,:]) ))
        cur_obs_s= np.vstack((cur_obs_s1, cur_obs_s2))
        goalies_states = cur_obs_g  # get initial state (goalies)
        strikers_states = cur_obs_s # get initial state (strikers)

        goalies_scores = np.zeros(n_goalie_agents)                   # initialize the score (goalies)
        strikers_scores = np.zeros(n_striker_agents)                 # initialize the score (strikers)         

        steps = 0
        t0 = perf_counter()
        print("t0, for loop = ", t0)
        while True:       
            # Team 0 is controlled by the learned goalie/striker policies.
            t1 = perf_counter() - t0
            print("t1 = ", t1)
            
            action_goalie_0, log_prob_goalie_0 = goalie_0.act( goalies_states[0] )
            action_striker_0, log_prob_striker_0= striker_0.act( strikers_states[0] )

            log_prob_goalie_1= goalie_0.act( goalies_states[1] )
            log_prob_striker_1 = striker_0.act( strikers_states[1] )
            if tracked_agent == -1 and len(decision_steps_p) >= 1:
                tracked_agent = decision_steps_p.agent_id[0]

            print(decision_steps_p.agent_id[0])
            decision_steps_p.agent_id[1]
            if tracked_agent == -1 and  len(decision_steps_b) >= 1:
                tracked_agent = decision_steps_b.agent_id[0]

            # Team 1 is driven by random actions. TensorFlow is only used for
            # this random-opponent action generation path.
            action_goalie_1 = np.asarray( tf.random.uniform(shape=[3], minval=-1, maxval=2, dtype=tf.int64) )
            action_striker_1 = np.asarray(  tf.random.uniform(shape=[3], minval=-1, maxval=2, dtype=tf.int64) )
            print(action_goalie_0)
            print(action_goalie_1)
            #print(cur_obs_s2)
           # actions_goalies = np.array( (action_goalie_0, action_goalie_1) )                                    
            #actions_strikers = np.array( (action_striker_0, action_striker_1) )
            env.set_actions(g_brain_name, np.vstack((action_goalie_0, action_striker_0)))
            env.set_actions(b_brain_name, np.vstack((action_goalie_1, action_striker_1)))
    


            env.step()
            #env_info = env.step(actions)

            # Pull the next observations and rewards out of Unity.
            decision_steps_p, terminal_steps_p = env.get_steps(g_brain_name)
            decision_steps_b, terminal_steps_b = env.get_steps(b_brain_name)
            #print(tracked_agent)
            # if tracked_agent in terminal_steps_p: # The agent terminated its episode
            #     goalie_0_reward += terminal_steps_p[tracked_agent].rewards
            #     done = True
            # if tracked_agent in terminal_steps_b: # The agent terminated its episode
            #     goalie_1_reward += terminal_steps_b[tracked_agent].reward
            #     done = True

            
            
            # Reward bookkeeping is split back into per-role arrays so the
            # learned goalie and striker can store their own trajectories.
            goalie_0_reward=decision_steps_p.reward[0]
            goalie_1_reward=decision_steps_b.reward[0]
            striker_0_reward=decision_steps_p.reward[1]
            striker_1_reward=decision_steps_b.reward[1]

            
            goalies_rewards = np.array([goalie_0_reward, goalie_1_reward])  
            strikers_rewards =np.array([striker_0_reward, striker_1_reward])
            #print("normal reward: ", goalies_rewards[0])
            #print("normal reward: ", strikers_rewards[0]) 
            if (t1 < 60): 
                goalies_scores += goalies_rewards
                strikers_scores += strikers_rewards
            else:
                # Legacy anti-stalling reward shaping: if observations appear
                # unchanged after a long span, apply a small penalty.
                cur_obs_g1_new = np.transpose(np.concatenate((decision_steps_p.obs[0][0,:], decision_steps_p.obs[1][0,:])))
                cur_obs_g2_new= np.transpose(np.concatenate((decision_steps_b.obs[0][0,:], decision_steps_b.obs[1][0,:])))
                
                cur_obs_s1_new =  np.transpose(np.concatenate((decision_steps_p.obs[0][1,:], decision_steps_p.obs[1][1,:]) ))
                cur_obs_s2_new =  np.transpose(np.concatenate((decision_steps_b.obs[0][1,:], decision_steps_b.obs[1][1,:]) ))
                
                t0=t1;
                if (cur_obs_g1_new == cur_obs_g1).all() or (cur_obs_s1 ==cur_obs_s1).all():
                    goalies_rewards = np.array([-0.1, -0.1])  
                    strikers_rewards =np.array([-0.1, -0.1])
                elif (cur_obs_g2_new == cur_obs_g2).all() or (cur_obs_s2 ==cur_obs_s2).all():
                    goalies_rewards = np.array([-0.1, -0.1])  
                    strikers_rewards =np.array([-0.1, -0.1])
                cur_obs_g1=cur_obs_g1_new
                cur_obs_s1=cur_obs_s1_new
                cur_obs_g2=cur_obs_g2_new
                cur_obs_s2=cur_obs_s2_new
                goalies_scores += goalies_rewards
                strikers_scores += strikers_rewards  
            # check if episode finished
            if (abs(goalies_rewards[0]) >=1 or abs(goalies_rewards[1])>=1 or abs(strikers_rewards[0])>1 or abs(strikers_rewards[1])>1):
                done = True
            #print("purple goalie score = ", goalies_scores[0], "and striker score = ", strikers_scores[0])
            #print("blue goalie score = ", goalies_scores[1], "and striker score = ", strikers_scores[1])

            #print(terminal_steps_p)

            # exit loop if episode finished
             
            # Store one experience for the learned goalie. The critic gets the
            # concatenated global state of all four players.
            goalie_0_reward = goalies_rewards[goalie_0.KEY]
            goalie_0.step( 
                goalies_states[goalie_0.KEY],
                np.concatenate( 
                    (
                        goalies_states[goalie_0.KEY],
                        strikers_states[striker_0.KEY],
                        goalies_states[GOALIE_1_KEY],

                        strikers_states[STRIKER_1_KEY],
                    ), axis=0 ),
                action_goalie_0,
                log_prob_goalie_0,
                goalie_0_reward 
            )


            # Store the matching striker experience with its own critic state.
            striker_0_reward = strikers_rewards[striker_0.KEY]
            striker_0.step(                 
                strikers_states[striker_0.KEY],
                np.concatenate( 
                    (
                        strikers_states[striker_0.KEY],
                        goalies_states[goalie_0.KEY],                        
                        strikers_states[STRIKER_1_KEY],                 
                        goalies_states[GOALIE_1_KEY]                        
                    ), axis=0 ),               
                action_striker_0,
                log_prob_striker_0,
                striker_0_reward
            )
            if (goalie_0_reward > 0 or goalie_1_reward>0): #or t1 > 180:

                done=True
                

            if done:
               break 
            cur_obs_g1 = np.transpose(np.concatenate((decision_steps_p.obs[0][0,:], decision_steps_p.obs[1][0,:]), axis=0))
            cur_obs_g2 = np.transpose(np.concatenate((decision_steps_b.obs[0][0,:], decision_steps_b.obs[1][0,:]), axis=0))
            cur_obs_g= np.vstack((cur_obs_g1, cur_obs_g2))
            cur_obs_s1 =  np.transpose(np.concatenate((decision_steps_p.obs[0][1,:], decision_steps_p.obs[1][1,:]), axis=0 ))
            cur_obs_s2 =  np.transpose(np.concatenate((decision_steps_b.obs[0][1,:], decision_steps_b.obs[1][1,:]), axis=0 ))
            cur_obs_s=np.vstack((cur_obs_s1, cur_obs_s2))
            goalies_next_states = cur_obs_g        
            strikers_next_states = cur_obs_s
            

            # roll over states to next time step
            goalies_states = goalies_next_states
            strikers_states = strikers_next_states

            steps += 1

        # PPO-style role-specific updates, followed by checkpoint writes.
        goalie_loss = goalie_optimizer.learn(goalie_0.memory)
        striker_loss = striker_optimizer.learn(striker_0.memory)        

        goalie_actor_model.checkpoint( CHECKPOINT_GOALIE_ACTOR )   
        goalie_critic_model.checkpoint( CHECKPOINT_GOALIE_CRITIC )    
        striker_actor_model.checkpoint( CHECKPOINT_STRIKER_ACTOR )    
        striker_critic_model.checkpoint( CHECKPOINT_STRIKER_CRITIC )

        team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
        team_0_window_score.append( team_0_score )
        team_0_window_score_wins.append( 1 if team_0_score > 0 else 0)        

        team_1_score = goalies_scores[GOALIE_1_KEY] + strikers_scores[STRIKER_1_KEY]
        team_1_window_score.append( team_1_score )
        team_1_window_score_wins.append( 1 if team_1_score > 0 else 0 )

        draws.append( team_0_score == team_1_score )
        
        print('Episode: {} \tSteps: \t{} \tGoalie Loss: \t {:.10f} \tStriker Loss: \t {:.10f}'.format( episode + 1, steps, goalie_loss, striker_loss ))
        print('\tPurple Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_0_window_score_wins), team_0_score, np.sum(team_0_window_score) ))
        print('\tBlue Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_1_window_score_wins), team_1_score, np.sum(team_1_window_score) ))
        print('\tDraws: \t{}'.format( np.count_nonzero(draws) ))

        if np.count_nonzero( team_0_window_score_wins ) >= 95:
            break
    

# train the agent
ppo_train()

# ---------------------------------------------------------------------------
# Post-training evaluation loop
# ---------------------------------------------------------------------------
#
# The original project kept a second loop here for ad hoc evaluation. It
# mirrors the state extraction/action path from training, but does not update
# memory or weights.

team_0_window_score = deque(maxlen=100)
team_0_window_score_wins = deque(maxlen=100)

team_1_window_score = deque(maxlen=100)
team_1_window_score_wins = deque(maxlen=100)

draws = deque(maxlen=100)

for episode in range(50):                                               # play game for n episodes
    env_info = env.reset()                        # reset the environment    
    decision_steps_p, terminal_steps_p = env.get_steps(g_brain_name)
    decision_steps_b, terminal_steps_b = env.get_steps(b_brain_name)

    # Rebuild the four role-specific states exactly as in training.
    cur_obs_g1 = np.transpose(np.concatenate((decision_steps_p.obs[0][0,:], decision_steps_p.obs[1][0,:])))
    cur_obs_g2 = np.transpose(np.concatenate((decision_steps_b.obs[0][0,:], decision_steps_b.obs[1][0,:])))
    cur_obs_g= np.vstack((cur_obs_g1, cur_obs_g2))
    cur_obs_s1 =  np.transpose(np.concatenate((decision_steps_p.obs[0][1,:], decision_steps_p.obs[1][1,:]) ))
    cur_obs_s2 =  np.transpose(np.concatenate((decision_steps_b.obs[0][1,:], decision_steps_b.obs[1][1,:]) ))
    cur_obs_s=np.vstack((cur_obs_s1, cur_obs_s2))
    goalies_states = cur_obs_g  # get initial state (goalies)
    strikers_states = cur_obs_s # get initial state (strikers)
    goalies_scores = np.zeros(n_goalie_agents)                          # initialize the score (goalies)
    strikers_scores = np.zeros(n_striker_agents)                        # initialize the score (strikers)

    steps = 0

    while True:
        # During evaluation, both teams use the learned role-shared policies.
        action_goalie_0 = goalie_0.act( goalies_states[0] )
        action_striker_0 = striker_0.act( strikers_states[0] )

        action_goalie_1= goalie_0.act( goalies_states[1] )
        action_striker_1 = striker_0.act( strikers_states[1] )

        #print(action_goalie_0.shape)
        # if steps==0:
        #     action_goalie_0 = np.asarray( [np.random.randint(goalie_action_size)] )
        #     action_striker_0 = np.asarray( [np.random.randint(striker_action_size)] )         
        #     action_goalie_1 = np.asarray( [np.random.randint(goalie_action_size)] )
        #     action_striker_1 = np.asarray( [np.random.randint(striker_action_size)] )
        #print(action_goalie_0,action_goalie_1)
    
        # env_info_goalie_0 = goalie_0.step(action_goalie_0)
        # env_info_striker_0= striker_0.step(action_striker_0)
        # env_info_goalie_1= goalie_0.step(action_goalie_1)
        # env_info_striker_1= striker_0.step(action_striker_1)     
       # actions_goalies = np.array( (action_goalie_0, action_goalie_1) )                                    
       # actions_strikers = np.array( (action_striker_0, action_striker_1) )

        env.set_actions(g_brain_name, np.vstack((action_goalie_0, action_striker_0)))
        env.set_actions(b_brain_name, np.vstack((action_goalie_1, action_striker_1)))
    

        # actions_goalies = np.array( (action_goalie_0, action_goalie_1) )                                    
        # actions_strikers = np.array( (action_striker_0, action_striker_1) )

        #actions = dict( zip( [g_brain_name, b_brain_name], [actions_goalies, actions_strikers] ) )



        env.step()
      
        # Read next observations and rewards from Unity.
        decision_steps_p, terminal_steps_p = env.get_steps(g_brain_name)
        decision_steps_b, terminal_steps_b = env.get_steps(b_brain_name)

        cur_obs_g1 = np.transpose(np.concatenate((decision_steps_p.obs[0][0,:], decision_steps_p.obs[1][0,:])))
        cur_obs_g2 = np.transpose(np.concatenate((decision_steps_b.obs[0][0,:], decision_steps_b.obs[1][0,:])))
        cur_obs_g= np.vstack((cur_obs_g1, cur_obs_g2))
        cur_obs_s1 =  np.transpose(np.concatenate((decision_steps_p.obs[0][1,:], decision_steps_p.obs[1][1,:]) ))
        cur_obs_s2 =  np.transpose(np.concatenate((decision_steps_b.obs[0][1,:], decision_steps_b.obs[1][1,:]) ))
        cur_obs_s=np.vstack((cur_obs_s1, cur_obs_s2))
        goalies_next_states = cur_obs_g        
        strikers_next_states = cur_obs_s
        
        # get reward and update scores
        goalie_0_reward=decision_steps_p.reward[0]
        goalie_1_reward=decision_steps_b.reward[0]
        striker_0_reward=decision_steps_p.reward[1]
        striker_1_reward=decision_steps_b.reward[1]
        goalies_rewards = np.array(goalie_0_reward, goalie_1_reward)  
        strikers_rewards =np.array(striker_0_reward, striker_1_reward)
        #print(goalies_rewards)
        #print(strikers_rewards)
        goalies_scores += goalies_rewards
        strikers_scores += strikers_rewards
                    
        # Roll over states to the next step. This section appears experimental
        # and may need cleanup before being used as a reliable evaluation loop.
        goalies_states = goalies_next_states
        strikers_states = strikers_next_states

        steps += 1
        
    team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
    team_0_window_score.append( team_0_score )
    team_0_window_score_wins.append( 1 if team_0_score > 0 else 0)        

    team_1_score = goalies_scores[GOALIE_1_KEY] + strikers_scores[STRIKER_1_KEY]
    team_1_window_score.append( team_1_score )
    team_1_window_score_wins.append( 1 if team_1_score > 0 else 0 )

    draws.append( team_0_score == team_1_score )
    
    print('Episode {}'.format( episode + 1 ))
    print('\tRed Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_0_window_score_wins), team_0_score, np.sum(team_0_window_score) ))
    print('\tBlue Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_1_window_score_wins), team_1_score, np.sum(team_1_window_score) ))
    print('\tDraws: \t{}'.format( np.count_nonzero( draws ) ))

env.close()
