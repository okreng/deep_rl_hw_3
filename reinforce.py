import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATE_SPACE = 8
ACTION_SPACE = 4


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        # self.model = model
        self.lr = lr # In case needed later

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

        # TODO: If statement for loading trained weights
        self.custom_adam = keras.optimizers.Adam(lr=lr)  # So that lr can be specified

        from keras import Input, Model, backend

        # TODO: Do I need to define Keras lambda layers?
        # def power_t(gamma_tensor, T_minus_t_tensor):
        #     return keras.backend.pow(gamma_tensor, T_minus_t_tensor)

        T_tensor = Input(shape=(1,), name='T')  # Add T=num_episodes as a tensor 
        gamma_tensor = Input(shape=(1,), name='gamma')  # Add gamma = discount rate as tensor
        # TODO: T needed?
        time_step_tensor = Input(shape=(1,), name='time_step')  # Add time step as a tensor
        reward_tensor = Input(shape=(1,), name='reward')  # Add reward as a tensor

        # Sum of the first n terms of geometric series: https://en.wikipedia.org/wiki/Geometric_series
        # Define the reward as r_t * ((1-gamma^(T-t))/(1-gamma))
        one_tensor = keras.backend.constant(1,shape=(1,))
        negative_one_tensor = keras.backend.constant(-1, shape=(1,))
        T_minus_t_tensor = keras.layers.subtract([T_tensor, time_step_tensor])
        # return_gamma_numerator_tensor = keras.layers.Lambda(power_t(gamma_tensor, T_minus_t_tensor))   #
        return_gamma_numerator_tensor = keras.backend.pow(gamma_tensor, T_minus_t_tensor)
        return_numerator_tensor = keras.layers.subtract([one_tensor, return_gamma_numerator_tensor])
        return_denominator_tensor = keras.layers.subtract([one_tensor, gamma_tensor])
        return_denominator_inverse_tensor = keras.backend.pow(return_denominator_tensor, negative_one_tensor)
        return_wo_reward_tensor = keras.layers.multiply([return_numerator_tensor, return_denominator_inverse_tensor])
        return_t_tensor = keras.layers.multiply([reward_tensor, return_wo_reward_tensor])

        self.model = Model(inputs=[model.inputs[0], reward_tensor, time_step_tensor, gamma_tensor, T_tensor], \
                           outputs=[model.outputs[0], return_t_tensor])
        
        # TODO: Delete once no longer in use for troubleshooting
        for inputs in self.model.inputs:
            print(inputs)
        for outputs in self.model.outputs:
            print(outputs)

        # TODO: Implement loss function

        # self.loss_function = self.reinforce_loss
        # TODO: Downscale the rewards by a factor of 100
        self.model.compile(optimizer=self.custom_adam, loss='categorical_crossentropy', \
            metrics=['categorical_accuracy']) 


    def reinforce_loss(self, y_true, y_pred):
        #  Defines the REINFORCE loss function
        pass



    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        return

    def generate_episode(self, env, render=False):
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
        while not done:  
            e_states.append(state)  # TODO: Should this be done before or after reshape?
            state = np.array([state])
            model_output = self.model.predict(x = state, verbose = 0)  # Get action from model
            action = np.argmax(model_output)  # Equivalent to greedy policy
            action_vec = np.zeros(ACTION_SPACE)
            action_vec[action] = 1
            e_actions.append(action_vec)
            state, reward, done, info = env.step(action)
            e_rewards.append(reward)
            if render:
                env.render()


        return np.array(e_states), np.array(e_actions), np.array(e_rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

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
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    reinforce = Reinforce(model, lr)  # Default learning rate is 0.0005

    # print(reinforce.generate_episode(env))


if __name__ == '__main__':
    main(sys.argv)
