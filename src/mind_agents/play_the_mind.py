"""Script to play The Mind card game."""

import argparse
import asyncio
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from mind_agents.api_utils import get_llm_client
from mind_agents.game import GameState, PlayerStats
from mind_agents.prompt_assets.parsers import parse_response_for_model
from mind_agents.prompt_assets.prompts import get_prompt_for_model
from mind_agents.prompt_assets.prompts.play_game import (
    TOOLS,
    generate_play_content,
    play_game_template,
)
from mind_agents.prompt_assets.types import Card, Model, PromptComponent, PromptTemplate, Response, Role

# Set up rich console and logging
console = Console()

# Configure logging - set httpx to WARNING to hide request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
)
logger = logging.getLogger(__name__)

# Create a secondary console for the game state that will stay at the top
game_state_console = Console()


@dataclass
class PlayerAction:
    """Represents a player's intended action."""

    player_id: int
    card: Card
    wait_time: float
    reason: str
    random_tiebreaker: float = field(default_factory=lambda: random.random())


def display_game_state(game: GameState, verbose: bool = False, show_stats: bool = False) -> None:
    """Display the current game state using Rich."""
    # Create a table for game info
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold blue")
    table.add_column("Value")

    table.add_row("Round", str(game.current_round))
    table.add_row("Lives", "❤️  " * game.lives_remaining)
    table.add_row("Stars", "⭐ " * game.stars_remaining)

    if game.played_cards:
        table.add_row("Played", " → ".join(str(c) for c in game.played_cards))

    # Always show player hands
    for player in game.players:
        if player.hand:
            cards = sorted(player.hand, key=lambda x: x.number)
            # Get model name if available
            model_name = ""
            if hasattr(game, "player_models"):
                model = game.player_models[player.id - 1]
                model_name = f" ({model.name})"
            table.add_row(
                f"Player {player.id}{model_name}",
                ", ".join(str(c.number) for c in cards),
            )

    # Print the game state
    console.print("─" * console.width)
    console.print(Panel(table, title="[bold blue]The Mind[/bold blue]"))

    # Add player statistics if requested
    if show_stats and hasattr(game, "player_stats"):
        # Create statistics table
        stats_table = Table(show_header=True, box=None)
        stats_table.add_column("Player", style="bold blue")
        stats_table.add_column("Cards Played", justify="right")
        stats_table.add_column("Star Attempts", justify="right")
        stats_table.add_column("Lives Lost", justify="right")

        for i, stats in enumerate(game.player_stats):
            # Get model name
            model_name = ""
            if hasattr(game, "player_models"):
                model = game.player_models[i]
                model_name = f" ({model.name})"

            stats_table.add_row(
                f"Player {i + 1}{model_name}",
                str(stats.cards_played),
                str(stats.star_attempts),
                str(stats.lives_lost),
            )

        # Add spacing and display stats in a panel
        console.print()
        console.print(
            Panel(
                stats_table,
                title="[bold cyan]Player Statistics[/bold cyan]",
                border_style="cyan",
            )
        )

    console.print()  # Add a blank line for separation


def display_player_action(
    action: PlayerAction,
    success: Optional[bool] = None,
    verbose: bool = False,
    game_state: Optional[GameState] = None,
) -> None:
    """Display a player's action in a clear format."""
    # Create a table for the action
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold blue")
    table.add_column("Value")

    # Get model name for the player if available
    model_name = ""
    if game_state and hasattr(game_state, "player_models"):
        model = game_state.player_models[action.player_id - 1]
        model_name = f" ({model.name})"

    table.add_row("Player", f"{action.player_id}{model_name}")
    table.add_row("Card", str(action.card.number))
    # Show "⭐ suggested" if wait time is infinite, otherwise show seconds
    wait_display = "⭐ suggested" if math.isinf(action.wait_time) else f"{action.wait_time:.1f} seconds"
    table.add_row("Wait", wait_display)
    table.add_row("Reason", action.reason)

    # In verbose mode, show the prompt that was used
    if verbose and game_state is not None:
        # Calculate cards for other players
        other_players = [p for p in game_state.players if p.hand and p.id != action.player_id]
        other_players_cards = [len(p.hand) for p in other_players]
        total_other_cards = sum(other_players_cards)

        # Get all cards in the player's hand
        player = game_state.players[action.player_id - 1]
        all_cards = sorted([c.number for c in player.hand])

        # Use the same prompt generation as analyze_card
        prompt_str = (
            f"There are {len(game_state.players)} players in the game. "
            f"Other players have {total_other_cards} cards in total. "
            f"The following cards have already been played in order: {game_state.played_cards}.\n"
            f"I have these cards: {all_cards}. I must play my lowest card ({action.card.number}). What should I do?"
        )
        table.add_row("Prompt", prompt_str)

    title_style = "bold green" if success else "bold red" if success is False else "bold blue"
    title = "Played Successfully" if success else "Invalid Move!" if success is False else "Planning Move"

    console.print(Panel(table, title=f"[{title_style}]{title}[/{title_style}]"))
    console.print()  # Add a blank line for separation


def load_config() -> dict[str, str]:
    """Load API keys from config file.

    Returns:
        Dict containing API keys for different providers

    Raises:
        ValueError: If config file not found or has invalid format
    """
    config_path = Path.home() / ".config" / "llm_keys" / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found at {config_path}. Please create it with your API keys.")

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


async def analyze_card(
    card: Card,
    num_players: int,
    cards_per_player: int,
    played_cards: Optional[list[int]] = None,
) -> Optional[Response]:
    """Analyze a card using the LLM.

    Args:
        card: The card to analyze
        num_players: Number of players in the game
        cards_per_player: Number of cards each player has
        played_cards: List of cards already played in ascending order

    Returns:
        Response containing the LLM's analysis and tool calls, or None if error
    """
    try:
        config = load_config()
        client = get_llm_client(config)
        content = generate_play_content(
            card.number,
            num_players,
            cards_per_player,
            played_cards=played_cards,
        )
        response = await client.generate_response(play_game_template, content)

        if not response.success:
            logger.error(f"Failed to generate response for card {card.number}: {response.error}")
            return None

        return response

    except Exception as e:
        logger.error(f"Error analyzing card {card.number}: {str(e)}")
        return None


async def get_player_action(
    game: GameState,
    player_id: int,
    card: Card,
    verbose: bool = False,
    force_wait: bool = False,
) -> Optional[PlayerAction]:
    """Get a player's intended action.

    Args:
        game: Current game state
        player_id: ID of the player
        card: Card to analyze
        verbose: Whether to show detailed information
        force_wait: Whether to force a wait time instead of allowing star usage

    Returns:
        PlayerAction if successful, None if error
    """
    # First, let's calculate cards for other players
    other_players = [p for p in game.players if p.hand and p.id != player_id]
    other_players_cards = [len(p.hand) for p in other_players]
    total_other_cards = sum(other_players_cards)

    # Get all cards in the player's hand
    player = game.players[player_id - 1]
    all_cards = sorted([c.number for c in player.hand])

    # Create the prompt
    prompt = (
        f"There are {len(game.players)} players in the game. "
        f"Other players have {total_other_cards} cards in total. "
        f"The following cards have already been played in order: {game.played_cards}.\n"
        f"I have these cards: {all_cards}. I must play my lowest card ({card.number}). "
    )

    # If forcing wait time, add context about why star isn't allowed
    if force_wait:
        prompt += "Star power cannot be used in this situation. " "You must specify a wait time in seconds."

    # Create a prompt template
    prompt_template = PromptTemplate(
        name="analyze_card",
        version="1.0",
        components=[
            PromptComponent(
                role=Role.SYSTEM,
                static_content=prompt,
            ),
        ],
        temperature=0.7,
        top_p=0.9,
        model=Model.GPT35,
        available_tools=[tool.name for tool in TOOLS],
    )

    # Get LLM client
    config = load_config()
    client = get_llm_client(config)

    # Generate response using the template
    response = await client.generate_response(
        template=prompt_template,
        content={"prompt": prompt},
    )

    if not response or not response.tool_calls:
        return None

    tool_call = response.tool_calls[0]  # We only use the first tool call

    if tool_call.tool == "wait_for_n_seconds":
        return PlayerAction(
            player_id=player_id,
            card=card,
            wait_time=float(tool_call.parameters["seconds"]),
            reason=tool_call.parameters["reason"],
        )
    elif tool_call.tool == "use_star" and not force_wait:
        # Only allow star usage if not forcing wait time
        return PlayerAction(
            player_id=player_id,
            card=card,
            wait_time=float("inf"),
            reason=tool_call.parameters["reason"],
        )
    elif tool_call.tool == "use_star" and force_wait:
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

        return PlayerAction(
            player_id=player_id,
            card=card,
            wait_time=wait_time,
            reason=f"Calculated {wait_time:.1f} second wait based on card value {card_value} and gap of {gap}",
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
            action = await get_player_action(game, player.id, card, verbose)

            if action:
                if math.isinf(action.wait_time):  # Player wants to use a star
                    star_users.append(player.id)
                actions.append(action)
                display_player_action(action, verbose=verbose, game_state=game)
            else:
                logger.error(f"Failed to get action for player {player.id}")

        if not actions:
            logger.error("No valid actions found!")
            return

        # Handle star usage
        if star_users:
            if len(star_users) == len(active_players) and game.stars_remaining > 0:
                # All players want to use a star
                console.print("[bold yellow]All players chose to use a star![/bold yellow]")
                game.stars_remaining -= 1

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
                    action = await get_player_action(game, player.id, card, verbose, force_wait=True)
                    if action:
                        actions.append(action)
                        display_player_action(action, verbose=verbose, game_state=game)
                    else:
                        logger.error(f"Failed to get action for player {player.id}")

                # If no actions (shouldn't happen since we checked active_players), return
                if not actions:
                    return
            else:
                # Not all players used a star, or no stars remaining
                if game.stars_remaining == 0:
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
                        new_action = await get_player_action(game, player_id, card, verbose, force_wait=True)
                        if new_action:
                            actions[i] = new_action
                            display_player_action(new_action, verbose=verbose, game_state=game)
                        else:
                            logger.error(f"Failed to get new action for player {player_id}")

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
            game.lives_remaining -= 1
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


async def get_wait_time(state: dict[str, Any], model: Model = Model.GPT35) -> float:
    """Get the wait time for a card based on the game state.

    Args:
        state: Current game state
        model: Model to use for analysis

    Returns:
        Wait time in seconds
    """
    try:
        # Get appropriate prompt for the model
        prompt = get_prompt_for_model(model, state)

        # Create a prompt template
        prompt_template = PromptTemplate(
            name="analyze_card",
            version="1.0",
            components=[
                PromptComponent(
                    role=Role.SYSTEM,
                    static_content=prompt,
                ),
            ],
            temperature=0.7,
            top_p=0.9,
            model=Model.GPT35,
            available_tools=[tool.name for tool in TOOLS],
        )

        # Get LLM client
        config = load_config()
        client = get_llm_client(config)

        # Get response from model
        response = await client.generate_response(
            template=prompt_template,
            content={"prompt": prompt},
        )

        # Parse the response
        if not response.raw_response:
            logger.error(f"Failed to get valid response from {model.name}")
            return 5.0  # Default wait time

        raw_response = str(response.raw_response)
        parsed_response = parse_response_for_model(model, raw_response)

        if not parsed_response or not parsed_response.success or not parsed_response.tool_calls:
            logger.error(f"Failed to get valid response from {model.name}")
            return 5.0  # Default wait time

        # Extract wait time from tool call
        tool_call = parsed_response.tool_calls[0]
        if tool_call.tool == "wait_for_n_seconds":
            return float(tool_call.parameters["seconds"])
        elif tool_call.tool == "use_star":
            return float("inf")  # Treat using a star as infinite wait

        return 5.0  # Default wait time if no valid tool call

    except Exception as e:
        logger.error(f"Error getting wait time from {model.name}: {str(e)}")
        return 5.0  # Default wait time


async def main(verbose: bool = False, models: Optional[list[Model]] = None, max_turns: Optional[int] = None) -> None:
    """Main function to play The Mind.

    Args:
        verbose: Whether to show detailed information
        models: List of models to use for each player. If fewer models than players,
               will cycle through the provided models.
        max_turns: Optional maximum number of turns before ending the game
    """
    # Initialize game with 3 players
    game = GameState(num_players=3)

    # If no models specified, use default for all players
    if not models:
        models = [Model.GPT35] * len(game.players)
    else:
        # If fewer models than players, cycle through them
        models = models * (len(game.players) // len(models) + 1)
        models = models[: len(game.players)]

    # Store models in game state for display purposes
    game.player_models = models

    # Initialize player statistics
    game.player_stats = [PlayerStats() for _ in range(len(game.players))]

    print("Using models:")
    for i, model in enumerate(models, 1):
        print(f"Player {i}: {model.name}")

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
            game.lives_remaining += 1
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

        # Set up player's hand
        game.players[0].hand = [Card(number=n) for n in player_cards]

        # Create a dummy player 2 with the right number of cards
        if other_cards_count > 0:
            # Use high numbers that won't affect decision making
            game.players[1].hand = [Card(number=n) for n in range(90, 90 + other_cards_count)]

        # Set played cards
        game.played_cards = played_cards

        # Only show game state in verbose mode
        if verbose:
            display_game_state(game, verbose)

        # Get action for the player
        card = min(game.players[0].hand, key=lambda x: x.number)
        action = await get_player_action(game, 1, card, verbose)

        if action:
            if verbose:
                display_player_action(action, verbose=verbose, game_state=game)
            else:
                # Just print the wait time
                print(f"{action.wait_time:.1f}")
        else:
            console.print("[red]Failed to get action for player[/red]")

    except json.JSONDecodeError:
        console.print("[red]Error: Invalid JSON format for game state[/red]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def display_available_models() -> None:
    """Display the list of available models."""
    console.print("\n[bold cyan]Available Models:[/bold cyan]")
    table = Table(show_header=True, box=None)
    table.add_column("Model", style="bold blue")
    table.add_column("API String", style="orange3")  # Set orange style for API strings

    # Add each model from the Model enum with its mapped string value
    for model in Model:
        table.add_row(model.name, model.value)

    console.print(Panel(table, title="[bold blue]The Mind - Model Options[/bold blue]"))
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play The Mind card game")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--test", help="Test mode: Provide game state as JSON string")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m.name for m in Model],
        help="Models to use for each player (e.g., GPT35 CLAUDE3_SONNET)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Display all available models and their descriptions",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of turns before ending the game",
    )
    args = parser.parse_args()

    if args.list_models:
        display_available_models()
    elif args.test:
        asyncio.run(test_specific_scenario(args.test, args.verbose))
    else:
        # Convert model names to enum values
        models = [Model[m] for m in args.models] if args.models else None
        asyncio.run(main(verbose=args.verbose, models=models, max_turns=args.max_turns))
