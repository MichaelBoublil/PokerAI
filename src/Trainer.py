"""
Credits: The training algorithm contained in the start() method has been inspired by Deep Q Network Tutorial article
written by Arthur Juliani that I found on Medium.
links:
- Medium Article: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
- Github Notebook: https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb?source=post_page-----8438a3e2b8df----------------------

The traning algorithm is inspired by this code that shows how to train a deep-Q Network that has nothing to do with a
poker game but there are many aspects that are different in my code and I show many versions of that algorithm for the
reward part.
"""

from copy import deepcopy

import tensorflow as tf
import numpy as np
import random as rand
from pypokerengine.api.game import setup_config, start_poker

from DQNPlayer import DQNPlayer, DQNPlayerV1, DQNPlayerV2, DQNPlayerV3And4, DQNPlayerV5, DQNPlayerV6
from random_player import RandomPlayer
from honest_player import HonestPlayer
from fish_player import FishPlayer
from console_player import ConsolePlayer
from my_emulator import MyEmulator


class experience_buffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(rand.sample(self.buffer, size)), [size, 5])


class Trainer:
    @staticmethod
    def updateTargetGraph(tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder

    @staticmethod
    def updateTarget(op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def __init__(self, batch_size=128, update_freq=50, discount=0.99, path=None, nb_players=5, max_rounds=10, start_stack=1500, load=False):
        self.batch_size = batch_size
        self.update_freq = update_freq
        # self.learning_rate = 0.001
        self.learning_rate = 0.0001
        self.y = discount
        self.start_E = 1 # starting chance of random action
        self.end_E = 0.1 # final chance of random action
        self.annealings_steps = 10000 # how many steps to reduce start_E to end_E
        self.num_episodes = 20000
        self.pre_train_steps = 1500 # how many steps of random action before training begin
        self.path = path
        self.tau = 0.01 # rate to update target network toward primary network
        self.latest_reward = 0

        self.nb_players = nb_players
        self.max_rounds = max_rounds
        self.start_stack = start_stack

        self.saver = None
        self.load = load
        self.emulator = MyEmulator()

    def start(self, file=None):
        tf.reset_default_graph()
        main_qn = DQNPlayerV6(learning_rate=self.learning_rate, discount=self.y, nb_players=self.nb_players,
                            start_stack=self.start_stack, max_round=self.max_rounds, custom_uuid="1")
        target_qn = DQNPlayerV6(learning_rate=self.learning_rate, discount=self.y, nb_players=self.nb_players,
                              start_stack=self.start_stack, max_round=self.max_rounds)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        trainables = tf.trainable_variables()
        target_ops = self.updateTargetGraph(trainables, self.tau)
        buffer = experience_buffer()

        e = self.start_E
        stepDrop = (self.start_E - self.end_E) / self.annealings_steps

        action_list = []
        total_steps = 0

        self.emulator.set_game_rule(player_num=self.nb_players, max_round=self.max_rounds, small_blind_amount=5, ante_amount=0)
        self.emulator.register_player(uuid="1", player=main_qn)
        self.emulator.register_player(uuid="2", player=FishPlayer())
        self.emulator.register_player(uuid="3", player=FishPlayer())
        self.emulator.register_player(uuid="4", player=HonestPlayer(nb_players=self.nb_players))
        self.emulator.register_player(uuid="5", player=RandomPlayer())

        with tf.Session() as sess:
            sess.run(init)
            main_qn.set_session(sess)
            if self.load:
                print('restoring model')
                if not file:
                    ckpt = tf.train.get_checkpoint_state(self.path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    self.saver.restore(sess, self.path + file)

            for i in range(0, self.num_episodes):
                episode_buffer = experience_buffer()
                init_state = self.emulator.generate_initial_game_state({
                    "1": {"name": "DQNPlayer", "stack": self.start_stack},
                    "2": {"name": "FishPlayer1", "stack": self.start_stack},
                    "3": {"name": "FishPlayer2", "stack": self.start_stack},
                    "4": {"name": "HonestPlayer", "stack": self.start_stack},
                    "5": {"name": "RandomPlayer", "stack": self.start_stack},
                })
                game_state, events = self.emulator.start_new_round(init_state)
                main_qn.set_begin_round_stack(game_state['table'].seats.players)
                main_qn.receive_round_start_message(None, None, game_state['table'].seats.players)
                rAll = 0
                msgs = []

                prev_inputs = None
                prev_action = None
                last_round = False
                j = 0
                nb_rounds = 0

                while not last_round:
                    params = self.emulator.run_until_my_next_action(game_state, "1", msgs)

                    if len(params) == 4:
                        game_state, valid_actions, hole_card, round_state = params
                        action_idx, action, amount = main_qn.declare_action_emul(valid_actions, hole_card, round_state)

                        if np.random.rand(1) < e or total_steps < self.pre_train_steps:
                            action_idx, action, amount = main_qn.select_action(valid_actions, np.random.randint(0, main_qn.nb_outputs))

                        game_state, msgs = self.emulator.apply_my_action(game_state, action, amount)
                        total_steps += 1
                        print('total number of actions:', total_steps)

                        #  I have to wait to have the next state which I won't know before the next hand before saving
                        #  my experience
                        if prev_inputs:
                            episode_buffer.add(np.reshape(
                                np.array([prev_inputs, prev_action, 0, main_qn.inputs, False]), [1, 5])
                            )

                        prev_inputs = main_qn.inputs
                        prev_action = action_idx
                        action_list.append(action_idx)
                        if total_steps > self.pre_train_steps:
                            if e > self.end_E:
                                e -= stepDrop
                            if total_steps % self.update_freq == 0:
                                train_batch = buffer.sample(self.batch_size)
                                Q1 = sess.run(main_qn.predict,
                                              feed_dict={main_qn.input_layer: np.vstack(train_batch[:, 3])})
                                Q2 = sess.run(target_qn.output_layer,
                                              feed_dict={target_qn.input_layer: np.vstack(train_batch[:, 3])})
                                end_multiplier = -(train_batch[:, 4] - 1)
                                double_q = Q2[range(self.batch_size), Q1]
                                target_q = train_batch[:, 2] + (self.y * double_q * end_multiplier)
                                _, loss = sess.run([main_qn.update, main_qn.loss],
                                                   feed_dict={
                                                       main_qn.input_layer: np.vstack(train_batch[:, 0]),
                                                       main_qn.target_output: target_q,
                                                       main_qn.actions: train_batch[:, 1]
                                                   })
                                self.updateTarget(target_ops, sess)
                    else:
                        j += 1
                        game_state, reward = params
                        print('reward before process:', reward)
                        last_round = self.emulator._is_last_round(game_state, self.emulator.game_rule)
                        nb_rounds, reward = self.set_reward_v6(reward, game_state, main_qn, nb_rounds, j, last_round)
                        print('reward for round after process:', reward)
                        rAll += reward
                        if reward != 0:
                            episode_buffer.add(np.reshape(
                                np.array([prev_inputs, prev_action, reward, main_qn.inputs, True]), [1, 5])
                            )
                        if last_round:
                            prev_inputs = None
                            prev_action = None
                        game_state, events = self.emulator.start_new_round(game_state)
                        main_qn.set_begin_round_stack(game_state['table'].seats.players)
                        main_qn.receive_round_start_message(None, None, game_state['table'].seats.players)

                buffer.add(episode_buffer.buffer)
                print(" -------- finished episode number: ---------------- ", i)
                if i % 200 == 0:
                    self.saver.save(sess, self.path+'/model_v6-'+str(i)+'.ckpt')
                    print("Saved Model")
            self.saver.save(sess, self.path+'/model_v6-'+str(i)+'.ckpt')

    def set_reward_v6(self, reward, game_state, main_qn, nb_rounds, j, last_round):
        if reward != 0:
            try:
                new_reward = reward / self.start_stack
            except Exception:
                new_reward = 0.5
            reward = new_reward * (self.max_rounds + 1 - j) if reward < 0 else new_reward * j
        else:
            if main_qn.stack_begin_of_round > 0:
                if main_qn.latest_ehs < 1.0 / main_qn.nb_participating_players:
                    reward = 5.0
                else:
                    reward = -5.0
            elif nb_rounds == 0:
                nb_rounds = j
        main_qn.update_inputs(game_state)
        if last_round:
            reward += 20 if main_qn.is_winner(game_state) else -20
        return nb_rounds, reward

    def set_reward_v4_and_v5(self, reward, game_state, main_qn, nb_rounds, j, last_round):
        if reward != 0:
            try:
                new_reward = reward / self.start_stack
            except Exception:
                new_reward = 0.5
            reward = new_reward * (self.max_rounds + 1 - j) if reward < 0 else new_reward * j
        else:
            if main_qn.stack_begin_of_round > 0:
                if main_qn.latest_ehs < 1.0 / self.nb_players:
                    reward = 5.0
                else:
                    reward = -5.0
            elif nb_rounds == 0:
                nb_rounds = j
        return nb_rounds, reward

    def set_reward_v3(self, reward, game_state, main_qn, nb_rounds, j, last_round):
        if reward != 0:
            new_reward = 0
            try:
                # This reward formula didn't work out for v3, but keeping it for the record
                # reward = reward / main_qn.stack_begin_of_round
                new_reward = reward / self.start_stack
            except Exception:
                new_reward = 0.5
            reward = new_reward * (self.max_rounds + 1 - j) if reward < 0 else new_reward * j
        else:
            if main_qn.stack_begin_of_round > 0:
                if main_qn.latest_ehs < 1.0 / main_qn.nb_participating_players:
                    reward = 1.0
                else:
                    reward = -1.0
            elif nb_rounds == 0:
                nb_rounds = j
        return nb_rounds, reward

    def set_reward_v2(self, reward, game_state, main_qn, nb_rounds, j, last_round):
        reward = reward / (self.start_stack * self.nb_players)
        return nb_rounds, reward

    def start_real_game(self, players, ai_version='5'):
        ai_params = {
            '3': {
                'class': DQNPlayerV3And4,
                'model': '3200'
            },
            '4': {
                'class': DQNPlayerV3And4,
                'model': '9999'
            },
            '5': {
                'class': DQNPlayerV5,
                'model': '19999'
            },
        }
        path = './models/v' + ai_version + '/model_v' + ai_version + '-' + ai_params[ai_version]['model'] + '.ckpt'
        tf.reset_default_graph()
        main_qn = ai_params[ai_version]['class'](learning_rate=self.learning_rate, discount=self.y,
                                                 nb_players=self.nb_players, start_stack=self.start_stack,
                                                 max_round=self.max_rounds)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            main_qn.set_session(sess)
            self.saver.restore(sess, path)
            config = setup_config(max_round=self.max_rounds, initial_stack=self.start_stack, small_blind_amount=5)
            i = 1
            for player in players:
                config.register_player(name='p'+str(i), algorithm=player['class'](**player['kwargs']))
                i += 1
            config.register_player(name='p' + str(i), algorithm=main_qn)
            game_result = start_poker(config, verbose=0)
            return game_result
