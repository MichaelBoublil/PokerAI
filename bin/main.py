import sys
import configparser
sys.path.insert(0, 'src/')
from Trainer import Trainer

from honest_player import HonestPlayer
from fish_player import FishPlayer
from console_player import ConsolePlayer
from random_player import RandomPlayer
from fold_player import FoldPlayer


fullCallSet = [
    {'class': FishPlayer, 'kwargs': {}},
    {'class': FishPlayer, 'kwargs': {}},
    {'class': FishPlayer, 'kwargs': {}},
    {'class': FishPlayer, 'kwargs': {}},
]

fullHonestSet = [
    {'class': HonestPlayer, 'kwargs': {'nb_players': 5}},
    {'class': HonestPlayer, 'kwargs': {'nb_players': 5}},
    {'class': HonestPlayer, 'kwargs': {'nb_players': 5}},
    {'class': HonestPlayer, 'kwargs': {'nb_players': 5}},
]

fullRandomSet = [
    {'class': RandomPlayer, 'kwargs': {}},
    {'class': RandomPlayer, 'kwargs': {}},
    {'class': RandomPlayer, 'kwargs': {}},
    {'class': RandomPlayer, 'kwargs': {}},
]

PolySetWithPlayer = [
    {'class': FishPlayer, 'kwargs': {}},
    {'class': RandomPlayer, 'kwargs': {}},
    {'class': ConsolePlayer, 'kwargs': {}},
    {'class': HonestPlayer, 'kwargs': {'nb_players': 5}},
]

PolySet = [
    {'class': FishPlayer, 'kwargs': {}},
    {'class': RandomPlayer, 'kwargs': {}},
    {'class': FishPlayer, 'kwargs': {}},
    {'class': HonestPlayer, 'kwargs': {'nb_players': 5}},
]


def test_ai(trainer: Trainer, set, nb_games=10):
    nb_wins = {
        'p1': 0,
        'p2': 0,
        'p3': 0,
        'p4': 0,
        'p5': 0
    }
    for i in range(0, nb_games):
        game_result = trainer.start_real_game(players=set, ai_version=ai_version)
        winner = game_result['players'][0]
        for player in game_result['players']:
            if player['stack'] > winner['stack']:
                winner = player
        print('winner for game number {0} is: {1}'.format(i, winner))
        nb_wins[winner['name']] += 1
    for key in nb_wins.keys():
        print(' ---- win rate of {0} is {1} ----'.format(key, nb_wins[key] / nb_games))


try:
    cfg = configparser.ConfigParser()
    cfg.read('ai_config.cfg')

    ai_version = str(cfg['DEFAULT']['ai_version'])
    if ai_version not in ['3', '4', '5', '6']:
        raise Exception('Error: only versions from 3 to 6 are supported for now')
except Exception:
    print('Error: Something happened, make sure the version inserted is supported and that you have an ai_config.cfg')
    exit(0)

trainer = Trainer(path='./logs', nb_players=5, max_rounds=15)

# trainer.start()

test_ai(trainer, fullCallSet, 100)
test_ai(trainer, fullRandomSet, 100)
test_ai(trainer, fullHonestSet, 100)
test_ai(trainer, PolySet, 100)

# If you want to play, uncomment the line below
# test_ai(trainer, PolySetWithPlayer, 100)
