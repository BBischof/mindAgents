# The Mind - AI Card Game

A Python implementation of The Mind card game where AI agents collaborate to play cards in ascending order without communication.

## Game Overview

The Mind is a cooperative card game where players must play their cards in ascending order without any verbal communication. In this implementation, AI agents take on the role of players, using sophisticated language models to make decisions about when to play their cards.

### Core Features
- Multiple AI players working together
- Lives system for mistakes
- Bonus lives awarded at specific rounds
- Star power for revealing cards
- Auto-play system for maintaining game flow

## Game Simulator

The simulator allows you to test different scenarios and analyze AI behavior in specific game situations.

### Features
- Test specific card combinations
- Analyze AI decision-making
- Track statistics for each player
- Compare different model behaviors

### Parameters
- `-p/--player-cards`: Number of cards each player has
- `-o/--other-cards`: Number of cards held by other players
- `-l/--played-cards`: Number of cards already played
- `-m/--model`: Model to use for the test player

## Model Selection

The game supports multiple AI models that can be assigned to different players. Each model may have different strategies and behaviors.

### Available Models
- CLAUDE3_SONNET: Claude 3 Sonnet model (default)
- GPT4: GPT-4 model
- GPT35: GPT-3.5 model
- (Other models as defined in `types.py`)

### Using Different Models

You can specify which model each player should use when running the game. To see all available models and their descriptions:

```bash
# List all available models
python -m src.mind_agents.play_the_mind --list-models
# or for the simulator
python -m src.mind_agents.mind_simulator --list-models
```

Then use specific models in your games:

```bash
# Run game with specific models for each player
python -m src.mind_agents.play_the_mind --models CLAUDE3_SONNET GPT4 GPT35

# Run game with the same model for all players
python -m src.mind_agents.play_the_mind --models CLAUDE3_SONNET

# Run simulator with a specific model
python -m src.mind_agents.mind_simulator -p 1 -o 1 -l 0 -m GPT4
```

The game will display which model is controlling each player and track their individual performance statistics.

## Installation and Requirements

1. Clone the repository
2. Create a Python virtual environment
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API keys in `~/.config/llm_keys/config.json`

## Output Format

The game provides detailed output including:
- Current game state
- Player actions and decisions
- Card plays and their outcomes
- Star power usage
- End-game statistics for each player/model

### Player Statistics
For each player, the game tracks:
- Number of cards successfully played
- Number of times they attempted to use a star
- Number of lives they cost the team

This allows for analysis of different models' performance and strategies.

```bash
# Run the standard game
uv run python -m src.mind_agents.play_the_mind

# Run with verbose output
uv run python -m src.mind_agents.play_the_mind -v
```

## Game Simulator

The simulator allows testing specific game scenarios to analyze the AI's decision-making. This simple benchmark can be used to evaluate the performance of different AI models.

### Simulator Features

- Test specific card combinations
- Control number of cards held by other players
- Specify previously played cards
- Generate comprehensive test data

### Running Simulations

```bash
# Basic simulation with default parameters
uv run python -m src.mind_agents.mind_simulator

# Specify parameters:
# -p: Number of cards the player has
# -o: Number of cards held by other players
# -l: Number of played cards to consider
# -v: Verbose output
uv run python -m src.mind_agents.mind_simulator -p 2 -o 3 -l 1
```

### Simulation Parameters

- `player_cards (-p)`: Number of cards the test player has
  - Must be ≥ 1
  - Will test all possible combinations of cards
- `other_cards (-o)`: Number of cards held by other players
  - Must be ≥ 1 (game logic requires at least one other card in play)
- `played_cards (-l)`: Number of cards already played
  - Must be ≥ 0
  - Will test all valid combinations of played cards
- `resolution (-r)`: Space between consecutive card values
  - Must be ≥ 1
  - Higher values reduce the number of combinations tested
  - Example: resolution=3 will use cards 1,4,7,10,...
- `model (-m)`: Model to use for decision making
  - Defaults to GPT35
  - Use `--list-models` to see available options

### Example Usage

```bash
# Test scenarios where:
# - Player has 2 cards
# - Other players have 3 cards total
# - 1 card has been played
# - Using resolution of 3 (cards spaced by 3)
# - Using GPT-4 model
python -m src.mind_agents.mind_simulator -p 2 -o 3 -l 1 -r 3 -m GPT4

# Results will be saved to: simulation_p2_h3_played1_r3_gpt4.csv

# List available models
python -m src.mind_agents.mind_simulator --list-models

# Run with default parameters (GPT3.5)
python -m src.mind_agents.mind_simulator
```

### Output Format

The simulator generates CSV files with the naming format:
```
simulation_p{player_cards}_h{other_cards}_played{played_cards}_r{resolution}_{model}.csv
```

Each CSV contains:
- `player_cards`: Tuple of cards the player has
- `other_cards`: Number of cards held by others
- `played_cards`: Tuple of cards already played
- `wait_time`: The AI's decided wait time before playing
- `model`: The model used for decision making

## Installation

1. Clone the repository
2. Create a Python virtual environment
3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Requirements

- Python 3.11+
- Dependencies listed in requirements.txt:
  - rich: For beautiful terminal output
  - pandas: For data handling in simulator
  - tqdm: For progress bars
