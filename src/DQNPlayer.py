from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import tensorflow as tf


class DQNPlayer(BasePokerPlayer):

    h_size = 9  # TODO: probably change the value, but for now we put this just as a reference
    output_size = 4

    def __init__(self, learning_rate, discount, nb_players, custom_uuid=None):
        super().__init__()
        self.nb_players = nb_players

        self.learning_rate = learning_rate
        self.discount = discount
        self.nb_inputs = 11 + (nb_players - 1)
        self.nb_outputs = 4

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.elu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h1, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_outputs])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

        self.summary_writer = tf.summary.FileWriter('./stats/')
        self.session = None
        self.inputs = None
        if custom_uuid:
            self.uuid = custom_uuid

    def set_session(self, session):
        self.session = session

    def gather_informations(self, hole_card, round_state):
        hand_strength = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=self.nb_players, hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card']))
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        dealer_btn = round_state['dealer_btn']
        next_player = round_state['next_player']
        small_blind = round_state['small_blind_pos']
        big_blind = round_state['big_blind_pos']
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0]
        other_stacks = [player['stack'] for player in round_state['seats'] if player['uuid'] != self.uuid]

        # TODO: add player type (aggressive, chill, ...)

        return [hand_strength, *list(street.values()), pots, dealer_btn, next_player, small_blind, big_blind, player_stack, *other_stacks]

    @staticmethod
    def select_action(valid_actions, action_idx):
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
        }
        # TODO: add a 5th action that would be the middle between min and max ?
        #  or maybe a second neural network that would evaluate the best value depending on the situation ?

        action = actions[action_idx]
        if action[1] == -1:
            action = actions[1]

        return action[0], action[1]

    def declare_action(self, valid_actions, hole_card, round_state):
        self.inputs = self.gather_informations(hole_card, round_state)

        action = self.session.run(self.predict, feed_dict={self.input_layer: [self.inputs]})[0]
        action, amount = self.select_action(valid_actions, action_idx=action)

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
