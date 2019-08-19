import sys
sys.path.insert(0, 'src/')
from Trainer import Trainer


def test_ai(trainer: Trainer, nb_games=10):
    nb_wins = {
        'p1': 0,
        'p2': 0,
        'p3': 0,
        'p4': 0,
        'p5': 0
    }
    for i in range(0, nb_games):
        game_result = trainer.start_real_game(file='/model_v3-3200.ckpt')
        winner = game_result['players'][0]
        for player in game_result['players']:
            if player['stack'] > winner['stack']:
                winner = player
        print('winner for game number {0} is: {1}'.format(i, winner))
        nb_wins[winner['name']] += 1
    for key in nb_wins.keys():
        print(' ---- win rate of {0} is {1} ----'.format(key, nb_wins[key] / nb_games))


trainer = Trainer(path='./logs', nb_players=5, max_rounds=15, load=True)

trainer.start(file='/model_v5-9999.ckpt')

# test_ai(trainer, 100)
