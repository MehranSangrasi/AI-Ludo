# Ludo Game with AI Players
This Ludo game, implemented using Pygame and PyTorch, features AI players that learn and improve using reinforcement learning.

## Installation
Before running the game, ensure you have the required libraries installed:
``` bash
pip install pygame torch
```
## Game Setup
- Pygame Library: Used for game development including rendering the game window and managing events.
- Assets: Game assets (images and sounds) must be placed in the same directory as the game script.
- Neural Network for AI Players
- QNetwork Class: A neural network with two fully connected layers, using ReLU activation for the hidden layer.
- AI Players: Four separate instances for each AI player, each with its own neural network.

## Running the Game
To start the game, run the main Python script:
``` bash
python ludo_game.py
```
## Gameplay
Players can choose between AI-only and mixed human-AI gameplay (from terminal for now, Until We make some GUI for that 🙏)
The game follows standard Ludo rules with AI players making decisions based on reinforcement learning.

## AI and Reinforcement Learning
- State Representation: Game state is represented as flattened token positions.
- Action Selection: An ε-greedy strategy is used for action selection.
- Reward System: Rewards are given based on game events like reaching the winner zone, capturing opponent tokens, etc.
- Training: AI players' neural networks are trained based on their actions and received rewards.

## Conclusion
This project demonstrates the application of reinforcement learning in a traditional board game setting, showcasing how AI can adapt and improve through gameplay.


## Team
- Mehran Wahid
- Abdullah Saim - [Github](https://github.com/ASa1m)
- Areeba Tanveer - [Github](https://github.com/areeba-tanveer).
- Onkar - [Github](https://github.com/onkarrai06) 

