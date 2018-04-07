import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from reinforce import Reinforce

STATE_SPACE = 8
ACTION_SPACE = 4
GAMMA = 1.00


# NOTE: Ignoring inheritance so I don't have to look at two files
class A2C(object):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n

        # TODO: Use this if loading weights
        # self.model.load_weights('reinforce_model_no_bias_2.h5')


        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

        self.custom_actor_adam = keras.optimizers.Adam(lr=lr)
        self.custom_critic_adam = keras.optimizers.Adam(lr=critic_lr)

        self.model.compile(optimizer=self.custom_actor_adam, loss=keras.losses.categorical_crossentropy) 

        self.critic_model.compile(optimizer=self.custom_critic_adam, loss=keras.losses.mean_squared_error)


    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        return

    def generate_episode(self, env, bias=0.0, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        # NOTE: Used exact method as imitation.py
        e_states = []
        e_actions = []
        e_rewards = []

        done = False
        state = env.reset()  # Restart the environment
        T = 0
        # return_t = np.zeros(1)  # To pass in to predict
        # T_tensor = np.zeros(1)
        while not done:
            T += 1
            e_states.append(state)  # TODO: Should this be done before or after reshape?
            state = np.array([state])
            # model_output = self.model.predict(x = [state, return_t, T_tensor], verbose = 0)  # Get action from model
            model_output = self.model.predict(x=state, verbose=0)
            # action = np.argmax(model_output)  # Equivalent to greedy policy
            probabilities = model_output/np.sum(model_output)
            action = np.random.choice(a=range(ACTION_SPACE), size=1, p=probabilities.flatten())
            action = action[0]
            action_vec = np.zeros(ACTION_SPACE)
            action_vec[action] = 1
            e_actions.append(action_vec)
            state, reward, done, info = env.step(action)
            
            # As stated in writeup
            # reward /= 100

            e_rewards.append(reward)
            if render:
                env.render()
        
        e_returns = np.zeros(T)
        e_return_vec = np.zeros((T, ACTION_SPACE))
        T_vector = np.zeros(T)
        for t in reversed(range(T)):
            T_vector[t] = T
            if (t == T-1):
                e_returns[t] = e_rewards[t] - bias
                e_return_vec[t, :] = e_returns[t]
            else:
                e_returns[t] = e_rewards[t] + GAMMA*e_returns[t+1] - bias
                e_return_vec[t, :] = e_returns[t]
            e_return_vec[t, :] = np.multiply(e_return_vec[t,:], e_actions[t])

        # print(np.array(e_states))
        e_return_vec /= 100
        # print(e_return_vec)
        
        # TODO: Delete these once done troubleshooting
        # print(e_rewards)
        # print(e_returns)

        # return np.array(e_states), np.array(model_output), np.array(e_actions), np.array(e_rewards), e_return_vec, T_vector
        return np.array(e_states), e_return_vec, np.array(e_rewards)


class Critic_Model():
    # Model for the critic
    def __init__(self, critic_config_path):
        # Define model parameters and save to critic_config_path
        from keras.models import Model
        from keras.layers import Input, Dense

        in_layer = Input(shape=(STATE_SPACE,))
        dense_1 = Dense(32)(in_layer)
        out_layer = Dense(1)(dense_1)

        model = Model(inputs=in_layer, outputs=out_layer)

        model_json = model.to_json()
        with open("critic-config.json", "w") as json_file:
            json_file.write(model_json)



def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--critic-config-path', dest='critic_config_path',
                        type=str, default='critic-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    critic_config_path = args.critic_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    
    # Note: for some reason, cannot read both in a single open() call

    # TODO: This line only needs to be called if saving a new model, but doesn't hurt
    Critic_Model(critic_config_path)  # In order to save the model structure before opening
    
    with open(critic_config_path, 'r') as f:
        # Start with critic model same as actor model
        critic_model = keras.models.model_from_json(f.read())

    actor_critic = A2C(model, lr, critic_model, critic_lr, )

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
