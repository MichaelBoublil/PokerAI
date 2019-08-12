from copy import deepcopy

import tensorflow as tf
import numpy as np
import random as rand
from pypokerengine.api.game import setup_config, start_poker

from DQNPlayer import DQNPlayer
from random_player import RandomPlayer
from fold_player import FoldPlayer
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

    def __init__(self, batch_size=128, update_freq=50, discount=0.99, path=None, nb_players=5, load=False):
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.learning_rate = 0.0001
        self.y = discount
        self.start_E = 1 # starting chance of random action
        self.end_E = 0.1 # final chance of random action
        self.annealings_steps = 10000 # how many steps to reduce start_E to end_E
        self.num_episodes = 10000
        self.pre_train_steps = 5000 # how many steps of random action before training begin
        self.path = path
        self.tau = 0.01 # rate to update target network toward primary network
        self.nb_players = nb_players

        self.saver = None
        self.load = load
        self.emulator = MyEmulator()

    def start(self):
        tf.reset_default_graph()
        main_qn = DQNPlayer(learning_rate=self.learning_rate, discount=self.y, nb_players=self.nb_players,
                            start_stack=1500, custom_uuid="1")
        target_qn = DQNPlayer(learning_rate=self.learning_rate, discount=self.y, nb_players=self.nb_players,
                              start_stack=1500)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        trainables = tf.trainable_variables()
        target_ops = self.updateTargetGraph(trainables, self.tau)
        buffer = experience_buffer()

        e = self.start_E
        stepDrop = (self.start_E - self.end_E) / self.annealings_steps

        jList = []
        rList = []
        action_list = []
        total_steps = 0

        self.emulator.set_game_rule(player_num=self.nb_players, max_round=15, small_blind_amount=5, ante_amount=0)
        self.emulator.register_player(uuid="1", player=main_qn)
        self.emulator.register_player(uuid="2", player=FoldPlayer())
        self.emulator.register_player(uuid="3", player=FishPlayer())
        self.emulator.register_player(uuid="4", player=HonestPlayer(nb_players=self.nb_players))
        self.emulator.register_player(uuid="5", player=RandomPlayer())

        with tf.Session() as sess:
            sess.run(init)
            main_qn.set_session(sess)
            if self.load:
                self.saver.restore(sess, self.path + self.load)
            for i in range(self.num_episodes):
                episode_buffer = experience_buffer()
                init_state = self.emulator.generate_initial_game_state({
                    "1": {"name": "DQNPlayer", "stack": 1500},
                    "2": {"name": "FoldPlayer", "stack": 1500},
                    "3": {"name": "FishPlayer", "stack": 1500},
                    "4": {"name": "HonestPlayer", "stack": 1500},
                    "5": {"name": "RandomPlayer", "stack": 1500},
                })
                game_state, events = self.emulator.start_new_round(init_state)
                rAll = 0
                msgs = []

                prev_inputs = None
                prev_action = None
                last_round = False
                j = 0

                while not last_round:
                    j += 1
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

                                r = np.mean(rList[-2:])
                                j = np.mean(jList[-2:])
                                q2 = double_q[0]
                                print(action_list)
                                print(action_list[-10:])
                                al = np.mean(action_list[-10:])

                                summary = tf.Summary()
                                summary.value.add(tag='Perf/Reward', simple_value=float(r))
                                summary.value.add(tag='Perf/Length', simple_value=float(j))
                                summary.value.add(tag='Perf/Action_list', simple_value=al)
                                summary.value.add(tag='Perf/E', simple_value=e)
                                summary.value.add(tag='Q/Q2', simple_value=float(q2))
                                summary.value.add(tag='Q/Target', simple_value=target_q[0])
                                summary.value.add(tag='Q/Action', simple_value=Q1[0])
                                summary.value.add(tag='Loss/Error', simple_value=loss)
                                main_qn.summary_writer.add_summary(summary, total_steps)
                                if total_steps % (self.update_freq * 2) == 0:
                                    main_qn.summary_writer.flush()
                    else:
                        game_state, reward = params
                        # reward = np.log(1 + reward) if reward >= 0 else -np.log(1 - reward)
                        reward = reward / (self.nb_players * 1500)
                        rAll += reward
                        if prev_inputs:
                            episode_buffer.add(np.reshape(
                                np.array([prev_inputs, prev_action, reward, main_qn.inputs, True]), [1, 5])
                            )
                        last_round = self.emulator._is_last_round(game_state, self.emulator.game_rule)
                        game_state, events = self.emulator.start_new_round(game_state)
                        prev_inputs = None
                        prev_action = None

                buffer.add(episode_buffer.buffer)
                rList.append(rAll)
                jList.append(j)
                print(" -------- finished episode number: ---------------- ", i)
                if i % 200 == 0:
                    self.saver.save(sess, self.path+'/model-'+str(i)+'.ckpt')
                    print("Saved Model")
                if len(rList) % 10 == 0:
                    print(total_steps,np.mean(rList[-10:]), e)
            self.saver.save(sess, self.path+'/model-'+str(i)+'.ckpt')

    def start_real_game(self):
        tf.reset_default_graph()
        main_qn = DQNPlayer(learning_rate=self.learning_rate, discount=self.y, nb_players=self.nb_players, custom_uuid="1")

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            main_qn.set_session(sess)
            ckpt = tf.train.get_checkpoint_state(self.path)
            print(ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
            config.register_player(name="p1", algorithm=FishPlayer())
            config.register_player(name="p2", algorithm=RandomPlayer())
            config.register_player(name="p3", algorithm=RandomPlayer())
            config.register_player(name="p4", algorithm=RandomPlayer())
            config.register_player(name="p5", algorithm=main_qn)
            game_result = start_poker(config, verbose=1)
            print(game_result)


def setup_ai():
    tf.reset_default_graph()
    main_qn = DQNPlayer(learning_rate=0.001, discount=0.99, nb_players=3, load=True)
    target_qn = DQNPlayer(learning_rate=0.001, discount=0.99, nb_players=3)
    target_qn.set_session(main_qn)

    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    Trainer.updateTargetGraph(trainables, 0.01)

    saver = tf.train.Saver()
    main_qn.session.run(init)
    saver.restore(main_qn.session, './logs/model-1000.ckpt')
    return main_qn


if __name__ == '__main__':
    setup_ai()
