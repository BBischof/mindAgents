"""Display utilities for The Mind game."""

import math
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mind_agents.game import GameState, GameStateInfo, PlayerAction

# Set up rich console for display
console = Console()


def display_game_state(game: GameState, verbose: bool = False, show_stats: bool = False) -> None:
    """Display the current game state using Rich.

    Args:
        game: Current game state
        verbose: Whether to show detailed information
        show_stats: Whether to show player statistics
    """
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
    """Display a player's action in a clear format.

    Args:
        action: The player action to display
        success: Whether the action was successful (None if not yet executed)
        verbose: Whether to show detailed information
        game_state: Optional game state for additional context
    """
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
        # Get game state info
        state_info = GameStateInfo.from_game_state(game_state, action.player_id, action.card)
        # Show only user messages
        table.add_row("User Prompt", state_info.prompt_messages[1]["content"])

    title_style = "bold green" if success else "bold red" if success is False else "bold blue"
    title = "Played Successfully" if success else "Invalid Move!" if success is False else "Planning Move"

    console.print(Panel(table, title=f"[{title_style}]{title}[/{title_style}]"))
    console.print()  # Add a blank line for separation
