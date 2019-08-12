import sys
# import pickle
sys.path.insert(0, 'src/')
#
# from pypokerengine.api.game import setup_config, start_poker
# from fish_player import FishPlayer
# from DQNPlayer import DQNPlayer
#
# config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
# config.register_player(name="p1", algorithm=FishPlayer())
# config.register_player(name="p2", algorithm=DQNPlayer(0.001, 0.99, 2))
# game_result = start_poker(config, verbose=1)
#
# print(game_result)

from Trainer import Trainer

trainer = Trainer(path='./logs')

trainer.start()
# trainer.start_real_game()
