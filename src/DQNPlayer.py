from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import tensorflow as tf


class DQNPlayer(BasePokerPlayer):

    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, version, nb_inputs, nb_outputs,
                 custom_uuid=None, load=False):
        super().__init__()

        self.nb_players = nb_players
        self.start_stack = start_stack
        self.stack_begin_of_round = start_stack
        self.max_round = max_round
        self.agressivity = 0
        self.latest_ehs = 0
        self.overall_agressivity = 0
        self.nb_actions_history = 0
        self.seat_position = 0

        self.call_amount = 10
        self.pot_odds = 10
        self.load = load
        if custom_uuid:
            self.uuid = custom_uuid
        self.learning_rate = learning_rate
        self.discount = discount

        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.summary_writer = tf.summary.FileWriter('./stats/v' + str(version))
        if load:
            self.session = tf.Session()
        self.inputs = None

    def set_session(self, session):
        self.session = session

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        raise NotImplementedError()

    @staticmethod
    def select_action(valid_actions, action_idx):
        raise NotImplementedError()

    def declare_action(self, valid_actions, hole_card, round_state):
        self.inputs = self.gather_informations(hole_card, round_state, valid_actions)

        action = self.session.run(self.predict, feed_dict={self.input_layer: [self.inputs]})[0]
        _, action, amount = self.select_action(valid_actions, action_idx=action)

        return action, amount

    def declare_action_emul(self, valid_actions, hole_card, round_state):
        self.inputs = self.gather_informations(hole_card, round_state, valid_actions)

        action_idx = self.session.run(self.predict, feed_dict={self.input_layer: [self.inputs]})[0]
        _, action, amount = self.select_action(valid_actions, action_idx=action_idx)

        return action_idx, action, amount

    def set_begin_round_stack(self, players):
        for player in players:
            if player.uuid == self.uuid:
                self.stack_begin_of_round = player.stack

    def receive_game_start_message(self, game_info):
        self.agressivity = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        seat_pos = 0
        for seat in seats:
            uuid = seat.uuid if type(seat) != dict else seat['uuid']
            if uuid == self.uuid:
                break
            seat_pos += 1
        self.seat_position = seat_pos

    def receive_street_start_message(self, street, round_state):
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        if street != 'preflop':
            self.update_agressivity(round_state, street_map[street] - 1)

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def is_winner(self, game_result):
        winner = game_result['table'].seats.players[0]
        for player in game_result['table'].seats.players:
            if player.stack > winner.stack:
                winner = player
        return winner.uuid == self.uuid

    def update_inputs(self, game_result):
        raise NotImplementedError()

    def update_agressivity(self, round_state, old_street):
        street_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}

        old_street = street_map[old_street] if type(old_street) is int else old_street

        try:
            actions_history = round_state['action_histories'][old_street]
        except KeyError:
            return
        for action in actions_history:
            if action['uuid'] == self.uuid:
                continue

            if action['action'] == 'FOLD' or action['action'] == 'CALL':
                self.agressivity += 0 / self.call_amount if action['action'] == 'FOLD' \
                    else action['amount'] / self.call_amount
                self.nb_actions_history += 1
            elif action['action'] == 'RAISE':
                self.agressivity += action['amount'] / self.call_amount
                self.nb_actions_history += 1
                self.call_amount = action['amount']
        self.overall_agressivity = self.agressivity / self.nb_actions_history

class DQNPlayerV6(DQNPlayer):
    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, custom_uuid=None, load=False):
        super().__init__(learning_rate, discount, nb_players, start_stack, max_round, version=6,
                         nb_inputs=12+(nb_players-1), nb_outputs=7, custom_uuid=custom_uuid, load=load)

        self.nb_participating_players = self.nb_players

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h1, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        participating_players = \
            len([player for player in round_state['seats'] if player['state'] == 'participating'])

        if participating_players == 1:
            participating_players += 1
        self.nb_participating_players = participating_players
        hand_strength = estimate_hole_card_win_rate(nb_simulation=2000, nb_player=participating_players,
                                                    hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card'])) / participating_players
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        dealer_btn = self.seat_position - round_state['dealer_btn']
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] / self.start_stack
        other_stacks = [player['stack'] / self.start_stack if player['state'] == 'participating' else 0
                        for player in round_state['seats'] if player['uuid'] != self.uuid]

        call_amount_in_live = valid_actions[1]['amount'] if valid_actions[1]['amount'] > 0 else valid_actions[2]['amount']['min']
        self.pot_odds = pots / call_amount_in_live
        self.latest_ehs = hand_strength
        round_ratio = round_state['round_count'] / self.max_round

        return [dealer_btn, hand_strength, call_amount_in_live, pots, self.overall_agressivity, round_ratio,
                *list(street.values()), player_stack, *other_stacks, participating_players]

    @staticmethod
    def select_action(valid_actions, action_idx):
        gap = (valid_actions[2]['amount']['max'] - valid_actions[2]['amount']['min']) / 4
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            4: (valid_actions[2]['action'], int(valid_actions[2]['amount']['min'] + gap)),
            5: (valid_actions[2]['action'], int(valid_actions[2]['amount']['min'] + (gap * 2))),
            6: (valid_actions[2]['action'], int(valid_actions[2]['amount']['min'] + (gap * 3)))
        }

        action = actions[action_idx]

        if action[1] == -1:
            action = actions[1]
        elif action_idx == 0 and actions[1][1] == 0:
            action = actions[1]

        return action_idx, action[0], action[1]

    def update_inputs(self, game_result):
        table = game_result['table']
        player_stack = [player.stack for player in table.seats.players if player.uuid == self.uuid][0] / self.start_stack
        other_stacks = [player.stack / self.start_stack for player in table.seats.players if player.uuid != self.uuid]
        return [table.dealer_btn, self.latest_ehs, 0, 0, self.overall_agressivity, 1,
                0, 0, 0, 0, player_stack, *other_stacks, 5]


class DQNPlayerV5(DQNPlayer):

    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, custom_uuid=None, load=False):
        super().__init__(learning_rate, discount, nb_players, start_stack, max_round, version=5, nb_inputs=10+(nb_players-1),
                         nb_outputs=7, custom_uuid=custom_uuid, load=load)

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h1, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        hand_strength = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=self.nb_players, hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card'])) / self.nb_players
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] / self.start_stack
        other_stacks = [player['stack'] / self.start_stack for player in round_state['seats'] if player['uuid'] != self.uuid]

        call_amount_in_live = valid_actions[1]['amount'] if valid_actions[1]['amount'] > 0 else valid_actions[2]['amount']['min']
        self.pot_odds = pots / call_amount_in_live
        self.latest_ehs = hand_strength
        round_ratio = round_state['round_count'] / self.max_round

        return [hand_strength, call_amount_in_live, pots, self.overall_agressivity, round_ratio, *list(street.values()), player_stack, *other_stacks]

    @staticmethod
    def select_action(valid_actions, action_idx):
        gap = (valid_actions[2]['amount']['max'] - valid_actions[2]['amount']['min']) / 4
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            4: (valid_actions[2]['action'], int(valid_actions[2]['amount']['min'] + gap)),
            5: (valid_actions[2]['action'], int(valid_actions[2]['amount']['min'] + (gap * 2))),
            6: (valid_actions[2]['action'], int(valid_actions[2]['amount']['min'] + (gap * 3)))
        }

        action = actions[action_idx]

        if action[1] == -1:
            action = actions[1]
        elif action_idx == 0 and actions[1][1] == 0:
            action = actions[1]

        return action_idx, action[0], action[1]

    def update_inputs(self, game_result):
        table = game_result['table']
        player_stack = [player.stack for player in table.seats.players if player.uuid == self.uuid][0] / self.start_stack
        other_stacks = [player.stack / self.start_stack for player in table.seats.players if player.uuid != self.uuid]
        return [self.latest_ehs, 0, 0, self.overall_agressivity, 1, 0, 0, 0, 0, player_stack, *other_stacks]


class DQNPlayerV4(DQNPlayer):

    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, custom_uuid=None, load=False):
        super().__init__(learning_rate, discount, nb_players, start_stack, max_round, version=5, nb_inputs=9+(nb_players-1),
                         nb_outputs=5, custom_uuid=custom_uuid, load=load)

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h1, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        hand_strength = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=self.nb_players, hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card'])) / self.nb_players
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] / self.start_stack
        other_stacks = [player['stack'] / self.start_stack for player in round_state['seats'] if player['uuid'] != self.uuid]

        call_amount_in_live = valid_actions[1]['amount'] if valid_actions[1]['amount'] > 0 else valid_actions[2]['amount']['min']
        self.pot_odds = pots / call_amount_in_live
        self.latest_ehs = hand_strength
        round_ratio = round_state['round_count'] / self.max_round

        return [hand_strength, pots, self.overall_agressivity, round_ratio, *list(street.values()), player_stack, *other_stacks]

    @staticmethod
    def select_action(valid_actions, action_idx):
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            4: (valid_actions[2]['action'], valid_actions[2]['amount']['min'] + valid_actions[2]['amount']['max'] // 2),
        }

        action = actions[action_idx]

        if action[1] == -1:
            action = actions[1]
        elif action_idx == 0 and actions[1][1] == 0:
            action = actions[1]

        return action_idx, action[0], action[1]

    def update_inputs(self, game_result):
        table = game_result['table']
        player_stack = [player.stack for player in table.seats.players if player.uuid == self.uuid][0] / self.start_stack
        other_stacks = [player.stack / self.start_stack for player in table.seats.players if player.uuid != self.uuid]
        return [self.latest_ehs, 0, self.overall_agressivity, 1, 0, 0, 0, 0, player_stack, *other_stacks]


class DQNPlayerV3(DQNPlayer):

    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, custom_uuid=None, load=False):
        super().__init__(learning_rate, discount, nb_players, start_stack, max_round, version=5, nb_inputs=9+(nb_players-1),
                         nb_outputs=5, custom_uuid=custom_uuid, load=load)

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        # h2 = tf.layers.dense(h1, self.h_size * 2, activation=tf.nn.relu,
        #                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        # h3 = tf.layers.dense(h2, self.h_size / 2, activation=tf.nn.relu,
        #                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h1, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        hand_strength = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=self.nb_players, hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card'])) / self.nb_players
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] / self.start_stack
        other_stacks = [player['stack'] / self.start_stack for player in round_state['seats'] if player['uuid'] != self.uuid]

        call_amount_in_live = valid_actions[1]['amount'] if valid_actions[1]['amount'] > 0 else valid_actions[2]['amount']['min']
        self.pot_odds = pots / call_amount_in_live
        self.latest_ehs = hand_strength
        round_ratio = round_state['round_count'] / self.max_round

        return [hand_strength, pots, self.overall_agressivity, round_ratio, *list(street.values()), player_stack, *other_stacks]

    @staticmethod
    def select_action(valid_actions, action_idx):
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            4: (valid_actions[2]['action'], valid_actions[2]['amount']['min'] + valid_actions[2]['amount']['max'] // 2),
        }

        action = actions[action_idx]

        if action[1] == -1:
            action = actions[1]
        elif action_idx == 0 and actions[1][1] == 0:
            action = actions[1]

        return action_idx, action[0], action[1]

    def update_inputs(self, game_result):
        table = game_result['table']
        player_stack = [player.stack for player in table.seats.players if player.uuid == self.uuid][0] / self.start_stack
        other_stacks = [player.stack / self.start_stack for player in table.seats.players if player.uuid != self.uuid]
        return [self.latest_ehs, 0, self.overall_agressivity, 1, 0, 0, 0, 0, player_stack, *other_stacks]


class DQNPlayerV2(DQNPlayer):

    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, custom_uuid=None, load=False):
        super().__init__(learning_rate, discount, nb_players, start_stack, max_round, version=5, nb_inputs=9+(nb_players-1),
                         nb_outputs=5, custom_uuid=custom_uuid, load=load)

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        h2 = tf.layers.dense(h1, self.h_size * 2, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        h3 = tf.layers.dense(h2, self.h_size / 2, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h3, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        hand_strength = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=self.nb_players, hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card'])) / self.nb_players
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] / self.start_stack
        other_stacks = [player['stack'] / self.start_stack for player in round_state['seats'] if player['uuid'] != self.uuid]

        call_amount_in_live = valid_actions[1]['amount'] if valid_actions[1]['amount'] > 0 else valid_actions[2]['amount']['min']
        self.pot_odds = pots / call_amount_in_live
        self.latest_ehs = hand_strength

        return [hand_strength, pots, *list(street.values()), player_stack, *other_stacks]

    @staticmethod
    def select_action(valid_actions, action_idx):
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            4: (valid_actions[2]['action'], valid_actions[2]['amount']['min'] + valid_actions[2]['amount']['max'] // 2),
        }

        action = actions[action_idx]

        if action[1] == -1:
            action = actions[1]
        elif action_idx == 0 and actions[1][1] == 0:
            action = actions[1]

        return action_idx, action[0], action[1]

    def update_inputs(self, game_result):
        table = game_result['table']
        player_stack = [player.stack for player in table.seats.players if player.uuid == self.uuid][0] / self.start_stack
        other_stacks = [player.stack / self.start_stack for player in table.seats.players if player.uuid != self.uuid]
        return [self.latest_ehs, 0, 0, 0, 0, 0, player_stack, *other_stacks]


class DQNPlayerV1(DQNPlayer):

    h_size = 32

    def __init__(self, learning_rate, discount, nb_players, start_stack, max_round, custom_uuid=None, load=False):
        super().__init__(learning_rate, discount, nb_players, start_stack, max_round, version=5, nb_inputs=9+(nb_players-1),
                         nb_outputs=5, custom_uuid=custom_uuid, load=load)

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_inputs])

        h1 = tf.layers.dense(self.input_layer, self.h_size, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_layer = tf.layers.dense(h1, self.nb_outputs)
        self.predict = tf.argmax(self.output_layer, 1)

        self.target_output = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(self.actions, self.nb_outputs, dtype=tf.float32)
        self.QOut = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
        self.error = tf.square(self.target_output - self.QOut)

        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def gather_informations(self, hole_card, round_state, valid_actions=None):
        hand_strength = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=self.nb_players, hole_card=gen_cards(hole_card),
                                                    community_card=gen_cards(round_state['community_card'])) / self.nb_players
        street = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0, round_state['street']: 1}

        pots = sum([round_state['pot']['main']['amount']] + [pot['amount'] for pot in round_state['pot']['side']])
        player_stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] / self.start_stack
        other_stacks = [player['stack'] / self.start_stack for player in round_state['seats'] if player['uuid'] != self.uuid]

        call_amount_in_live = valid_actions[1]['amount'] if valid_actions[1]['amount'] > 0 else valid_actions[2]['amount']['min']
        self.pot_odds = pots / call_amount_in_live
        self.latest_ehs = hand_strength

        return [hand_strength, pots, *list(street.values()), player_stack, *other_stacks]

    @staticmethod
    def select_action(valid_actions, action_idx):
        actions = {
            0: (valid_actions[0]['action'], valid_actions[0]['amount']),
            1: (valid_actions[1]['action'], valid_actions[1]['amount']),
            2: (valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            3: (valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            4: (valid_actions[2]['action'], valid_actions[2]['amount']['min'] + valid_actions[2]['amount']['max'] // 2),
        }

        action = actions[action_idx]

        if action[1] == -1:
            action = actions[1]
        elif action_idx == 0 and actions[1][1] == 0:
            action = actions[1]

        return action_idx, action[0], action[1]

    def update_inputs(self, game_result):
        table = game_result['table']
        player_stack = [player.stack for player in table.seats.players if player.uuid == self.uuid][0] / self.start_stack
        other_stacks = [player.stack / self.start_stack for player in table.seats.players if player.uuid != self.uuid]
        return [self.latest_ehs, 0, 0, 0, 0, 0, player_stack, *other_stacks]

