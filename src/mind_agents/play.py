"""Script to play The Mind card game."""

import argparse
import asyncio
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

from mind_agents.core.display import console, display_game_state, display_player_action
from mind_agents.core.game import GameState, GameStateInfo, PlayerAction, PlayerStats
from mind_agents.llm.prompts.wait_n_seconds_prompts import play_game_template
from mind_agents.llm.types import MODELS, Card
from mind_agents.llm.utilities import get_llm_client
from rich.logging import RichHandler

# Set up logging - set httpx to WARNING to hide request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
)
logger = logging.getLogger(__name__)

# Create a secondary console for the game state that will stay at the top
game_state_console = console


def load_config() -> dict[str, str]:
    """Load API keys from environment variables or config file.

    Returns:
        Dict containing API keys for different providers

    Raises:
        ValueError: If neither environment variables nor config file are properly set up
    """
    # First try environment variables
    config = {}
    env_keys = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
        "groq_api_key": "GROQ_API_KEY",
    }

    # Check environment variables and collect whatever is available
    for config_key, env_key in env_keys.items():
        value = os.getenv(env_key)
        if value:
            config[config_key] = value

    # If we have at least one key from environment, return what we have
    if config:
        return config

    # Fall back to config file if no environment variables are set
    config_path = Path.home() / ".config" / "llm_keys" / "config.json"
    if not config_path.exists():
        raise ValueError(
            f"Neither environment variables ({', '.join(env_keys.values())}) "
            f"nor config file at {config_path} are properly set up. "
            "Please set environment variables or create config file with your API keys."
        )

    with open(config_path) as f:
        raw_config = json.load(f)

        # Validate config format
        if not isinstance(raw_config, dict):
            raise ValueError("Config must be a dictionary")

        # Extract and validate required API keys
        required_keys = {
            "openai_api_key": "OpenAI",
            "anthropic_api_key": "Anthropic",
            "google_api_key": "Google",
            "groq_api_key": "Groq",
        }

        config: dict[str, str] = {}
        for key, provider in required_keys.items():
            if key not in raw_config:
                raise ValueError(f"Missing {provider} API key in config")
            value = raw_config[key]
            if not isinstance(value, str):
                raise ValueError(f"{provider} API key must be a string")
            config[key] = value

        return config


async def get_player_action(
    game: GameState,
    player_id: int,
    card: Card,
    force_wait: bool = False,
) -> Optional[PlayerAction]:
    """Get a player's intended action.

    Args:
        game: Current game state
        player_id: ID of the player
        card: Card to analyze
        force_wait: Whether to force a wait time instead of allowing star usage

    Returns:
        PlayerAction if successful, None if error
    """
    # Get game state info
    state_info = GameStateInfo.from_game_state(game, player_id, card)

    # Get LLM client with the player's assigned model
    model_name = game.player_models[player_id - 1]  # player_id is 1-indexed
    client = await get_llm_client(model_name)

    # Generate response using the template
    response = await client.generate(
        template=play_game_template,
        dynamic_content=state_info.dynamic_content,
    )

    if not response or not response.tool_calls:
        logger.error(
            f"Failed to get valid response for player {player_id}:\n"
            f"Response: {response}\n"
            f"Model: {model_name}\n"
            f"Game state: {state_info.dynamic_content}"
        )
        return None

    tool_call = response.tool_calls[0]  # We only use the first tool call

    # Parse the wait time
    try:
        if tool_call["tool"] == "wait_for_n_seconds":
            wait_time = float(tool_call["parameters"]["seconds"])
            reason = tool_call["parameters"]["reason"]
        elif tool_call["tool"] == "use_star" and not force_wait:
            # Only allow star usage if not forcing wait time
            wait_time = float("inf")
            reason = tool_call["parameters"]["reason"]
        elif tool_call["tool"] == "use_star" and force_wait:
            # If model tries to use star when forced to wait,
            # calculate a reasonable wait time based on card value
            card_value = card.number
            last_played = game.played_cards[-1] if game.played_cards else 0
            gap = card_value - last_played

            # Base wait time on card value and gap
            if card_value <= 20:
                wait_time = 1.0
            elif card_value <= 60:
                wait_time = 3.0 + (gap / 20.0)  # Increases with gap
            else:
                wait_time = 8.0 + (gap / 10.0)  # Increases more with gap

            # Cap at 15 seconds
            wait_time = min(wait_time, 15.0)
            reason = f"Calculated {wait_time:.1f} second wait based on card value {card_value} and gap of {gap}"
        else:
            logger.error(
                f"Unknown tool used by player {player_id}:\n"
                f"Tool: {tool_call['tool']}\n"
                f"Model: {model_name}\n"
                f"Parameters: {tool_call.get('parameters', {})}\n"
                f"Game state: {state_info.dynamic_content}"
            )
            return None

        return PlayerAction(
            player_id=player_id,
            card=card,
            wait_time=wait_time,
            reason=reason,
        )
    except (KeyError, ValueError) as e:
        logger.error(
            f"Failed to parse response for player {player_id}:\n"
            f"Error: {str(e)}\n"
            f"Tool call: {tool_call}\n"
            f"Model: {model_name}\n"
            f"Game state: {state_info.dynamic_content}"
        )
        return None


async def play_round(game: GameState, verbose: bool = False) -> None:
    """Play a single round of The Mind.

    Args:
        game: Current game state
        verbose: Whether to show detailed information
    """
    console.print(f"\n[bold cyan]Round {game.current_round}[/bold cyan]")
    display_game_state(game, verbose)

    # Reset for new round
    game.played_cards = []  # Clear played cards from previous round
    game.deck = list(range(1, 101))  # Reset to full deck
    random.shuffle(game.deck)  # Shuffle the deck

    # Deal cards for this round
    game.deal_cards()
    display_game_state(game, verbose)
    console.print()  # Add a blank line after initial state

    # Keep track of active players
    active_players = list(game.players)

    # Continue until all cards are played or game is over
    while active_players and not game.game_over():
        # If only one player left, just play their cards in order
        if len(active_players) == 1:
            player = active_players[0]
            while player.hand:
                card = min(player.hand, key=lambda x: x.number)
                auto_action = PlayerAction(
                    player_id=player.id,
                    card=card,
                    wait_time=0.0,  # No need to wait
                    reason="Last player - must play all remaining cards in order",
                )
                display_player_action(auto_action, True, verbose=verbose, game_state=game)
                success, _ = game.play_card(player.id, card)
                display_game_state(game, verbose)
                console.print()  # Add spacing after game state
            break

        # Get all players' intended actions
        actions = []
        star_users = []

        # First pass: collect all actions and identify star users
        for player in active_players[:]:
            if not player.hand:
                active_players.remove(player)
                continue

            # Always play lowest card in hand
            card = min(player.hand, key=lambda x: x.number)
            action = await get_player_action(game, player.id, card)

            if action:
                if math.isinf(action.wait_time):  # Player wants to use a star
                    star_users.append(player.id)
                actions.append(action)
                display_player_action(action, verbose=verbose, game_state=game)
            else:
                logger.error(f"Failed to get action for player {player.id}")
                # Exit immediately on player action failure
                raise RuntimeError(f"Critical error: Failed to get action for player {player.id}")

        if not actions:
            logger.error("No valid actions found!")
            # Exit immediately if no valid actions were found
            raise RuntimeError("Critical error: No valid actions found!")

        # Handle star usage
        if star_users:
            if len(star_users) == len(active_players) and game.stars > 0:
                # All players want to use a star
                console.print("[bold yellow]All players chose to use a star![/bold yellow]")
                game.stars -= 1

                # Update star attempt stats for all players
                for player_id in star_users:
                    game.player_stats[player_id - 1].star_attempts += 1

                # Show everyone's lowest card
                console.print("\n[bold yellow]Revealing and playing lowest cards:[/bold yellow]")
                lowest_cards = []  # List of (player_id, card) tuples
                for player in active_players:
                    lowest_card = min(player.hand, key=lambda x: x.number)
                    lowest_cards.append((player.id, lowest_card))
                    console.print(f"Player {player.id}: {lowest_card.number}")
                console.print()  # Add spacing

                # Sort lowest cards by number and play them automatically
                lowest_cards.sort(key=lambda x: x[1].number)
                for player_id, card in lowest_cards:
                    auto_action = PlayerAction(
                        player_id=player_id,
                        card=card,
                        wait_time=0.0,  # Auto-played cards have no wait time
                        reason="Card was auto-played after all players used a star",
                    )
                    display_player_action(auto_action, True, verbose=verbose, game_state=game)
                    success, _ = game.play_card(player_id, card)
                    # Update cards played stat
                    game.player_stats[player_id - 1].cards_played += 1

                # Update game state
                display_game_state(game, verbose)
                console.print()  # Add spacing

                # Update active players list after playing all cards
                active_players = [p for p in game.players if p.hand]
                if not active_players:
                    # If no players have cards left, round is over
                    return

                # Only get new actions if there are still players with cards
                actions = []
                for player in active_players:
                    card = min(player.hand, key=lambda x: x.number)
                    action = await get_player_action(game, player.id, card, force_wait=True)
                    if action:
                        actions.append(action)
                        display_player_action(action, verbose=verbose, game_state=game)
                    else:
                        logger.error(f"Failed to get action for player {player.id}")
                        # Exit immediately on player action failure
                        raise RuntimeError(f"Critical error: Failed to get action for player {player.id}")

                # If no actions (shouldn't happen since we checked active_players), return
                if not actions:
                    logger.error("No valid actions found after star usage!")
                    # Exit immediately if no valid actions were found
                    raise RuntimeError("Critical error: No valid actions found after star usage!")
            else:
                # Not all players used a star, or no stars remaining
                if game.stars == 0:
                    console.print("[yellow]Some players wanted to use a star, but none remaining![/yellow]")
                else:
                    console.print("[yellow]Some players wanted to use a star, but not all players agreed![/yellow]")
                console.print()  # Add spacing

                # Update star attempt stats for players who tried
                for player_id in star_users:
                    game.player_stats[player_id - 1].star_attempts += 1

                # Get new actions from star users with forced wait times
                for i, action in enumerate(actions):
                    if math.isinf(action.wait_time):
                        player_id = action.player_id
                        card = action.card
                        new_action = await get_player_action(game, player_id, card, force_wait=True)
                        if new_action:
                            actions[i] = new_action
                            display_player_action(new_action, verbose=verbose, game_state=game)
                        else:
                            logger.error(f"Failed to get new action for player {player_id}")
                            # Exit immediately on player action failure
                            raise RuntimeError(f"Critical error: Failed to get new action for player {player_id}")

        # Sort actions by wait time, using random tiebreaker for equal wait times
        actions.sort(key=lambda x: (x.wait_time, x.random_tiebreaker))

        # Execute the action with shortest wait time
        action = actions[0]

        # If there are tied actions, mention it
        tied_actions = [a for a in actions if abs(a.wait_time - action.wait_time) < 0.001]
        if len(tied_actions) > 1:
            console.print(
                f"[yellow]Players {[a.player_id for a in tied_actions]} tied - "
                f"randomly selected Player {action.player_id}[/yellow]"
            )
            console.print()  # Add spacing after tie message

        # Check if this would be a valid play by checking all remaining cards
        all_remaining_cards: list[int] = []
        for player in active_players:
            all_remaining_cards.extend(card.number for card in player.hand)
        would_be_valid = action.card.number == min(all_remaining_cards)

        # Show the result of the action
        display_player_action(action, would_be_valid, verbose=verbose, game_state=game)

        if not would_be_valid:
            # Remove a life for invalid move
            game.lives -= 1
            console.print("[red]Lost a life due to invalid move![/red]")
            console.print()

            # Update lives lost stat for the player
            game.player_stats[action.player_id - 1].lives_lost += 1

            # If invalid play, auto-play all lower cards first
            console.print(f"[yellow]Auto-playing all cards lower than {action.card.number}[/yellow]")
            console.print()

            # Find all cards that need to be auto-played
            auto_play_cards = []  # List of (player_id, card) tuples
            for player in active_players:
                lower_cards = [c for c in player.hand if c.number < action.card.number]
                for card in lower_cards:
                    auto_play_cards.append((player.id, card))

            # Sort by card number and play them in order
            auto_play_cards.sort(key=lambda x: x[1].number)

            # Play each card and show it
            for player_id, card in auto_play_cards:
                success, _ = game.play_card(player_id, card)
                auto_action = PlayerAction(
                    player_id=player_id,
                    card=card,
                    wait_time=0.0,  # Auto-played cards have no wait time
                    reason="Card was auto-played because it was lower than a card that was played out of order",
                )
                display_player_action(auto_action, True, verbose=verbose, game_state=game)

            # Now play the original card
            success, _ = game.play_card(action.player_id, action.card)
        else:
            # Valid play, just execute it
            success, _ = game.play_card(action.player_id, action.card)
            # Update cards played stat
            game.player_stats[action.player_id - 1].cards_played += 1

        # Update game state without clearing screen
        display_game_state(game, verbose)
        console.print()  # Add spacing after game state

        # Update active players list - only those who still have cards
        active_players = [p for p in game.players if p.hand]


def should_award_bonus_life(round_completed: int, num_players: int) -> bool:
    """Determine if a bonus life should be awarded.

    Args:
        round_completed: The round that was just completed
        num_players: Number of players in the game

    Returns:
        True if a bonus life should be awarded, False otherwise
    """
    # Common bonus life rounds for 2-3 players
    common_rounds = {2, 3, 5, 6, 8, 9}

    # 4-player games also get a bonus life after round 1
    four_player_rounds = {1} | common_rounds

    if num_players == 4:
        return round_completed in four_player_rounds
    else:
        return round_completed in common_rounds


async def main(verbose: bool = False, models: Optional[list[str]] = None, max_turns: Optional[int] = None) -> None:
    """Run the game.

    Args:
        verbose: Whether to show detailed output
        models: List of models to use for each player. If fewer models than players,
            will cycle through the provided models.
        max_turns: Maximum number of turns to play
    """
    # Initialize game state
    game = GameState(num_players=3)

    # If no models specified, use default for all players
    if not models:
        models = ["GPT35_TURBO"] * len(game.players)

    # If fewer models than players, cycle through them
    models = models * (len(game.players) // len(models) + 1)
    models = models[: len(game.players)]

    # Store models in game state for display purposes
    game.player_models = models

    # Display initial game state
    if verbose:
        print("\nInitial game state:")
        print("Using models:")
        for i, model_name in enumerate(models, 1):
            print(f"Player {i}: {model_name}")
        display_game_state(game)

    # Initialize player statistics
    game.player_stats = [PlayerStats() for _ in range(len(game.players))]

    # Play rounds until game over
    while not game.game_over():
        # Set cards per player equal to round number
        game.cards_per_player = game.current_round

        # Play the round
        await play_round(game, verbose)

        # Check if we lost
        if game.game_over():
            console.print("\n[bold red]Game Over - Out of lives![/bold red]")
            break

        # Check if we hit max turns
        if max_turns is not None and game.current_round >= max_turns:
            console.print("\n[bold yellow]Game Over - Maximum turns reached![/bold yellow]")
            break

        # If we completed the round successfully, check for bonus life
        if should_award_bonus_life(game.current_round, len(game.players)):
            game.lives += 1
            console.print(f"\n[bold green]Bonus life awarded for completing round {game.current_round}![/bold green]")

        # Increment round
        game.current_round += 1

        # Check if we won (completed all rounds)
        if game.current_round > 12:  # The Mind goes up to 12 rounds
            console.print("\n[bold green]Congratulations! You've won The Mind![/bold green]")
            break

    # Print final game state with statistics
    console.print("\n[bold green]Game Summary[/bold green]")
    display_game_state(game, verbose, show_stats=True)


async def test_specific_scenario(game_state_json: str, verbose: bool = False) -> None:
    """Test a specific game scenario.

    Args:
        game_state_json: JSON string containing the game state
        verbose: Whether to show detailed information
    """
    try:
        # Parse the game state
        state_dict = json.loads(game_state_json)

        # Extract played cards and held cards count
        played_cards = state_dict.get("P", [])
        other_cards_count = state_dict.get("H", 0)
        player_cards = state_dict.get("1", [])

        if not player_cards:
            console.print("[red]Error: No cards specified for player![/red]")
            return

        # Create minimal game state with 2 players
        game = GameState(num_players=2)
        game.played_cards = played_cards

        # Set up player's hand
        game.players[0].hand = [Card(number=n) for n in player_cards]

        # Create a dummy player 2 with the right number of cards
        if other_cards_count > 0:
            # Use high numbers that won't affect decision making
            game.players[1].hand = [Card(number=n) for n in range(90, 90 + other_cards_count)]

        # Get action for the player's lowest card
        card = min(game.players[0].hand, key=lambda x: x.number)
        action = await get_player_action(game, 1, card)

        if action:
            if verbose:
                display_player_action(action, verbose=verbose, game_state=game)
            else:
                # Just print the wait time for non-verbose mode
                print(f"{action.wait_time:.1f}")
        else:
            console.print("[red]Failed to get action for player[/red]")

    except json.JSONDecodeError:
        console.print("[red]Error: Invalid JSON format for game state[/red]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def display_available_models() -> None:
    """Display the list of available models."""
    console = console
    console.print("\n[bold cyan]Available Models:[/bold cyan]")

    table = Table(show_header=True)
    table.add_column("Model", style="bold blue")
    table.add_column("Description", style="green")

    # Add each model from the MODELS dictionary
    for model_name, metadata in MODELS.items():
        description = f"{metadata.provider.title()} model with {metadata.context_length} context length"
        table.add_row(model_name, description)

    console.print(Panel(table, title="[bold blue]The Mind - Model Options[/bold blue]"))


def cli_main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Play The Mind card game")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        choices=list(MODELS.keys()),
        help="Models to use for each player (e.g., GPT35_TURBO CLAUDE3_SONNET)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Display all available models and their descriptions",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--max-turns", type=int, help="Maximum number of turns to play")
    args = parser.parse_args()

    if args.list_models:
        display_available_models()
        return

    # Run the game
    asyncio.run(main(verbose=args.verbose, models=args.models, max_turns=args.max_turns))


if __name__ == "__main__":
    cli_main()
