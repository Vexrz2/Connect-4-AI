# Connect-4-AI
 Connect 4 Game implemented using PyGame (for basic interface and game mechanics).
 Features: Human Agent, Random Agent, MinMax Agent, AlphaBeta Agent, DQN Agent (need to train network to use).

 ### Play
 Play on the "Game.py" file, and choose agents to play with.

 ### Train
 Use the "Trainer*.py" files to train a DQN Agent using DQN. Configure ANN on "DQN.py" file, and epsilon decay on "Constant.py" file.

 ### Play with DQN
 To play with the DQN agent, you need to train a network first with custom parameters (network size, batch size, etc.) and store the parameters after training in files (for parameters_path parameter in DQN agent constructor).

 ### Requirements
 Install python libraries: pygame, torch (for DQN), numpy

