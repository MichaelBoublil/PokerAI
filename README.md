# Presentation

Hi, I'm Michael Halfon, a student at University of Kent, and here's my Dissertation project for the Msc Advanced Computer Science.

The objective was to create an Artificial Intelligence (AI) that could win at poker against other players. The details of this AI will be of course revealed on the dissertation. Here you'll find the entire code of this project.

To achieve my objectives, I've decided to use a Deep Q-Network for the model of my AI.

bin/main.py contains the demo launcher. /!\ THE DEMO HAS TO BE LAUNCHED FROM THE ROOT OF THE REPOSITORY, like this: `./bin/main.py`

src/ contains the entire code for this project.

models/ contains the most updated model for each version of my Network

logs/ contains some outdated models for each version of my network

stats/ contains tensorboard informations that were useful for the monitoring of the training.

## Credits

### The Game

The game engine that I use is PyPokerEngine

link: https://github.com/ishikota/PyPokerEngine

#### MyEmulator (src/my_emulator.py)

The code contained in this file doesn't belong to me. I modified it so that it would work out well with my algorithm, but most of the code has been taken from the repository TensorPoker, maintained by EvgenyKashin.
link: https://www.github.com/EvgenyKashin/TensorPoker

This code is not relevant for my AI. The main utility is to make the game run until my player has an action to do.
It should be included in the native Emulator of PyPokerEngine but it's not the case, so someone fixed it.

### Trainer (src/Trainer.py)

The training algorithm contained in the start() method has been inspired by Deep Q Network Tutorial article written by Arthur Juliani that I found on Medium.
links:
- Medium Article: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
- Github Notebook: https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb?source=post_page-----8438a3e2b8df----------------------

The traning algorithm is inspired by this code that shows how to train a deep-Q Network that has nothing to do with a poker game but there are many aspects that are different in my code and I show many versions of that algorithm for the reward part.

### Players

Only my AI models belong to me (So everything in DQNPlayer.py), all the other players xxxx_player.py belong to the
PyPokerEngine repository.

link: https://github.com/ishikota/PyPokerEngine/tree/master/examples/players

# Demo

## Configuration
To configure the version of the AI that you'd like to try, please insert a number between 3 and 6 inside the ai_config.cfg. The rest will be taken care of automatically.

/!\ The versions 1 and 2 of the model are available to look at, but there aren't any models saved to test, so only versions from 3 to 6 are available for testing. Performances of 1 and 2 will be ready to analyze for the dissertation /!\

## Automatic demonstration

To launch the demo just execute this at the root of the repository: `./bin/main.py`

The demonstration of my work is 100 games against 4 different set of players. Which means that a total of 400 games will be simulated. At the end of each set (100 games) the winrate of each player will be displayed. And the winner of the game is displayed after each game

Our agent is ALWAYS p5 (player 5)

The automatic demo should take about 3 hours to complete the 400 games.


## Demo with you as a player

The game is available only in console, because to play in GUI with the current AI system some modifications should be required in the PyPokerGUI library, which I can't control on your computer.

If you want to try a game with you playing too, go in `./bin/main.py`, comment all the `test_ai(...)` funtion calls, and uncomment the last line of the file.
