import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import Input, Model, backend
import gym

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


STATE_SPACE = 8
ACTION_SPACE = 4
GAMMA = 1.00


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr # In case needed later

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

        # TODO: If statement for loading trained weights
        self.custom_adam = keras.optimizers.Adam(lr=lr)  # So that lr can be specified

        # from keras import Input, Model, backend

        # TODO: Do I need to define Keras lambda layers?
        # def power_t(gamma_tensor, T_minus_t_tensor):
        #     return keras.backend.pow(gamma_tensor, T_minus_t_tensor)

        # T_tensor = Input(shape=(1,), name='T')  # Add T=num_episodes as a tensor 
        # gamma_tensor = Input(shape=(1,), name='gamma')  # Add gamma = discount rate as tensor
        # # TODO: T needed?
        # time_step_tensor = Input(shape=(1,), name='time_step')  # Add time step as a tensor
        # reward_tensor = Input(shape=(1,), name='reward')  # Add reward as a tensor

        # TODO: Delete this once done troubleshooting
        # G_t function was corrected, r_k in argument, not r_t

        # Sum of the first n terms of geometric series: https://en.wikipedia.org/wiki/Geometric_series
        # Define the reward as r_t * ((1-gamma^(T-t))/(1-gamma))
        # one_tensor = keras.backend.constant(1,shape=(1,))
        # negative_one_tensor = keras.backend.constant(-1, shape=(1,))
        # T_minus_t_tensor = keras.layers.subtract([T_tensor, time_step_tensor])
        # # return_gamma_numerator_tensor = keras.layers.Lambda(power_t(gamma_tensor, T_minus_t_tensor))   #
        # return_gamma_numerator_tensor = keras.backend.pow(gamma_tensor, T_minus_t_tensor)
        # return_numerator_tensor = keras.layers.subtract([one_tensor, return_gamma_numerator_tensor])
        # return_denominator_tensor = keras.layers.subtract([one_tensor, gamma_tensor])
        # return_denominator_inverse_tensor = keras.backend.pow(return_denominator_tensor, negative_one_tensor)
        # return_wo_reward_tensor = keras.layers.multiply([return_numerator_tensor, return_denominator_inverse_tensor])
        # return_t_tensor = keras.layers.multiply([reward_tensor, return_wo_reward_tensor])

        # self.model = Model(inputs=[model.inputs[0], reward_tensor, time_step_tensor, gamma_tensor, T_tensor], \
        #                    outputs=[model.outputs[0], return_t_tensor])

        # self.return_tensor = Input(shape=(1,), name='return')
        # self.T_tensor = Input(shape=(1,), name='T')

        # self.output_choice = keras.backend.argmax(model.outputs[0])
        # # self.choices_tensor = keras.backend.constant([0, 1, 2, 3])
        # self.output_one_hot = keras.utils.to_categorical(self.output_choice[0])

        # self.model = Model(inputs=[model.inputs[0]], outputs=[model.outputs[0], self.output_one_hot])
        
        # TODO: Delete once no longer in use for troubleshooting
        # for inputs in self.model.inputs:
        #     print(inputs)
        # for outputs in self.model.outputs:
        #     print(outputs)

        # TODO: Implement loss function

        # TODO: Downscale the rewards by a factor of 100
        # self.model.compile(optimizer=self.custom_adam, loss=self.reinforce_loss)
        self.model.compile(optimizer=self.custom_adam, loss=keras.losses.categorical_crossentropy) 


    # def reinforce_loss(self, y_true, y_pred):
    #     #  Defines the REINFORCE loss function
    #     action_log = keras.backend.log(y_pred)
    #     # action_mask_action = keras.backend.argmax(y_pred)
    #     # loss_product_unmasked = keras.layers.multiply([action_log, y_true])
    #     # action_mask = keras.utils.to_categorical(keras.backend.argmax(y_pred), num_classes = ACTION_SPACE)
    #     # loss_product_masked = keras.layers.multiply([loss_product_unmasked, action_mask])
    #     # Should have dimension (1,), which keedims=True ensures -> https://keras.io/backend/
    #     # loss_sum = keras.backend.sum(loss_product, keepdims=True)  
    #     # negative_one_tensor = keras.backend.constant(-1, shape=(1,))
    #     # T_inverse_tensor = keras.backend.pow(self.T_tensor, negative_one_tensor)
    #     # reinforce_loss_tensor = keras.layers.multiply([loss_sum, T_inverse_tensor])
    #     # return reinforce_loss_tensor
    #     loss_product = keras.layers.multiply([action_log, y_true])
    #     return loss_product

    def train(self, env, gamma=1.0):  # Note: 
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        # states, actions, actions_one_hot, rewards, returns, T = self.generate_episode(env)
        states, returns, _ = self.generate_episode(env)

        # print(states.size)
        # print(rewards.size)
        # As stated in writeup - now in generate_episode
        # rewards /= 100
        # returns /= 100

        # print(len(states))
        # print(len(returns))


        # Fit method requires labels, but our loss function doesn't use labels
        # junk_labels = np.zeros(actions.shape)
        # self.model.fit(x=states, y=returns, batch_size=T.size, verbose=0, class_weight=actions_one_hot)
        # self.model.fit(x=states, y=returns, batch_size=int(np.floor(rewards.size/2)), verbose=0)
        self.model.fit(x=states, y=returns, batch_size=len(states), verbose=0)

        # self.model.fit(x=[states, returns, T], y=junk_labels, batch_size=T.size, verbose=0)
        # self.model.fit(x=[states, returns, T], y=junk_labels, batch_size=1, verbose=0)

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
                e_returns[t] = e_rewards[t]
                e_return_vec[t, :] = e_returns[t]
            else:
                e_returns[t] = e_rewards[t] + GAMMA*e_returns[t+1]
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
    # reinforce.generate_episode(env)

    # plt.ion()
    for episode in range(num_episodes):
        reinforce.train(env)
        if episode % 1000 == 0:
            print("Episode: {}".format(episode))
            cum_reward = []
            for test_episode in range(100):  # Fixed by handout
                # states, _, actions, rewards, _, _ = reinforce.generate_episode(env)
                states, returns, rewards = reinforce.generate_episode(env)
                cum_reward.append(np.sum(rewards))
            mean = np.mean(cum_reward) # * 100
            std = np.std(cum_reward) # * 100
            print("Mean cumulative reward is: {}".format(mean))
            print("Reward standard deviation is: {}".format(std))
            plt.errorbar(episode, mean, yerr=std, fmt='--o')
            plt.title("Mean reward over training episodes")
            plt.xlabel('Training episodes')
            plt.ylabel('Mean cumulative reward for 100 test episodes')
            plt.draw()
            reinforce.model.save_weights("reinforce_model.h5")

            # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
            # serialize model to JSON
            model_json = model.to_json()
            with open("reinforce_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("reinforce_model.h5")
    plt.show()






if __name__ == '__main__':
    main(sys.argv)
