#!/usr/bin/env python3

# b57daf89-cd2f-11e8-a4be-00505601122b
# 68ec476c-c305-11e8-a4be-00505601122b
# 3f076765-806c-11eb-a1a9-005056ad4f31


import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=16, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.05, type=float, help="Learning rate.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
class Agent:
    def __init__(self, env, args):
        # TODO: Create a suitable model. The predict method assumes
        # it is stored as `self._model`.
        #
        # Using Adam optimizer with given `args.learning_rate` is a good default.
        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.Input(shape=(4,)))
        self._model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"))
        self._model.add(tf.keras.layers.Dense(2, activation="softmax"))

        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self._model.compile(loss=loss, optimizer=opt)

    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        # TODO: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to use the `sample_weight` argument of
        # tf.losses.Loss.__call__, but you can also construct the Loss object
        # with tf.losses.Reduction.NONE and perform the weighting manually.

        with tf.GradientTape() as tape:
            pred = self._model(states, training=True)
            loss = self._model.compiled_loss(actions, pred, sample_weight=returns)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the agent
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                prob = agent.predict([state])
                action = np.random.choice(2, 1, p=prob[0]).item()

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns from the received rewards
            returns = []
            T = len(rewards)
            for t in np.arange(T): # t=0,...,T-1
                G=0
                for k in np.arange(t+1,T+1): # k=t+1,...,T
                    G += np.power(args.gamma, k-t-1)*rewards[k-1]
                returns.append(G)

            # TODO: Add states, actions and returns to the training batch

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)


        # TODO: Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose greedy action
            action = np.argmax(agent.predict([state])).item()
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
