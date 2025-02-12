"""Script to simulate multiple scenarios of The Mind game."""

import argparse
import asyncio
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from core.game import GameState
from llm.prompts.wait_n_seconds_prompts import play_game_template
from llm.types import MODELS, Card
from llm.utilities import generate_play_content, get_llm_client
from tqdm import tqdm


@dataclass
class SimulationResult:
    """Result of a single simulation."""

    player_cards: tuple[int, ...]
    other_cards: int
    played_cards: tuple[int, ...]
    wait_time: float
    model_name: str


async def run_test_scenario(
    player_cards: list[int],
    other_cards: int,
    played_cards: list[int],
    model_name: str = "CLAUDE3_SONNET",
) -> Optional[float]:
    """Run a single test scenario.

    Args:
        player_cards: List of cards the test player has
        other_cards: Number of cards held by other players
        played_cards: List of cards already played
        model_name: The name of the model to use for the test player

    Returns:
        The wait time returned by the model, or None if there was an error
    """
    if other_cards == 0:
        raise ValueError("Cannot run simulation with 0 other cards - would cause division by zero in game logic")

    # Create a minimal game state
    game = GameState(num_players=2)
    game.played_cards = played_cards

    # Set up player's hand
    game.players[0].hand = [Card(number=n) for n in player_cards]

    # Create a dummy player 2 with the right number of cards
    if other_cards > 0:
        # Use high numbers that won't affect decision making
        game.players[1].hand = [Card(number=n) for n in range(90, 90 + other_cards)]

    # Get action for the player's lowest card
    card = min(game.players[0].hand, key=lambda x: x.number)
    client = await get_llm_client(model_name)

    # Generate dynamic content
    dynamic_content = generate_play_content(
        card_number=card.number,
        num_players=len(game.players),
        total_other_cards=other_cards,
        all_cards=sorted(c.number for c in game.players[0].hand),
        played_cards=game.played_cards,
    )

    response = await client.generate(
        template=play_game_template,
        dynamic_content=dynamic_content,
    )

    if not response or not response.tool_calls:
        return None

    tool_call = response.tool_calls[0]
    if tool_call["tool"] == "wait_for_n_seconds":
        return float(tool_call["parameters"]["seconds"])
    elif tool_call["tool"] == "use_star":
        return float("inf")
    return None


def get_ordered_tuples(size: int, max_val: int = 100, resolution: int = 1) -> list[tuple[int, ...]]:
    """Get all possible ordered tuples of a given size.

    Args:
        size: Size of each tuple
        max_val: Maximum value to include
        resolution: Space between consecutive values

    Returns:
        List of ordered tuples
    """
    # Generate values using resolution
    values = list(range(1, max_val + 1, resolution))
    return [tuple(sorted(combo)) for combo in combinations(values, size)]


async def run_simulations(
    player_cards: int = 1,
    other_cards: int = 1,
    played_cards: Union[int, list[int]] = 0,
    model_name: str = "CLAUDE3_SONNET",
    max_val: int = 100,
    resolution: int = 1,
) -> pd.DataFrame:
    """Run multiple simulation scenarios.

    Args:
        player_cards: Number of cards each player has
        other_cards: Number of cards held by other players
        played_cards: List of specific cards already played or number of cards to simulate
        model_name: Name of the model to use for decision making
        max_val: Maximum card value
        resolution: Space between consecutive card values

    Returns:
        DataFrame containing simulation results
    """
    print("Running simulations with parameters:")
    print(f"- Player cards: {player_cards}")
    print(f"- Other cards: {other_cards}")
    print(f"- Played cards: {played_cards}")
    print(f"- Model: {model_name}")
    print(f"- Resolution: {resolution}")

    # Generate all possible combinations
    player_card_combos = get_ordered_tuples(player_cards, max_val, resolution)

    # Handle played cards based on input type
    if isinstance(played_cards, list):
        # Generate all possible combinations of the provided played cards
        played_cards_sorted = sorted(played_cards)
        if not played_cards_sorted:
            played_card_combos: list[tuple[int, ...]] = [()]
        else:
            # Validate played cards
            if any(not 1 <= card <= max_val for card in played_cards_sorted):
                raise ValueError(f"All played cards must be between 1 and {max_val}")
            if len(played_cards_sorted) != len(set(played_cards_sorted)):
                raise ValueError("Duplicate cards in played_cards list")
            played_card_combos = [tuple(played_cards_sorted)]
    else:
        # Generate all possible combinations of n played cards
        played_card_combos = [()] if played_cards == 0 else get_ordered_tuples(played_cards, max_val, resolution)

    results: list[SimulationResult] = []

    # Filter and count valid scenarios
    valid_scenarios = []
    for p_cards in player_card_combos:
        for p_played in played_card_combos:
            # Skip if any player card is already in played cards
            if any(card in p_played for card in p_cards):
                continue

            # Skip if any player card is less than the highest played card
            # Note: We allow equal values here as they represent different cards
            max_played_current = max(p_played) if p_played else 0
            if any(card < max_played_current for card in p_cards):
                continue

            valid_scenarios.append((p_cards, p_played))

    total_scenarios = len(valid_scenarios)
    print(f"\nFound {total_scenarios} valid scenarios")
    if total_scenarios == 0:
        print("Warning: No valid scenarios found. This might indicate an issue with the input parameters.")
        return pd.DataFrame()  # Return empty DataFrame if no valid scenarios

    with tqdm(total=total_scenarios, desc="Running simulations") as pbar:
        for p_cards, p_played in valid_scenarios:
            wait_time = await run_test_scenario(
                player_cards=list(p_cards),
                other_cards=other_cards,
                played_cards=list(p_played),
                model_name=model_name,
            )
            results.append(
                SimulationResult(
                    player_cards=p_cards,
                    other_cards=other_cards,
                    played_cards=p_played,
                    wait_time=wait_time if wait_time is not None else float("nan"),
                    model_name=model_name,
                )
            )
            pbar.update(1)

    # Convert results to DataFrame
    df = pd.DataFrame([vars(r) for r in results])

    # Save results
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)

    # Format played cards for filename
    played_str = (
        f"played{'_'.join(str(x) for x in played_cards)}" if isinstance(played_cards, list) else f"played{played_cards}"
    )

    output_path = (
        output_dir / f"simulation_p{player_cards}_h{other_cards}_{played_str}_r{resolution}_{model_name.lower()}.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return df


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Run The Mind game simulations")
    parser.add_argument("-p", "--player-cards", type=int, default=1, help="Number of cards each player has")
    parser.add_argument("-o", "--other-cards", type=int, default=1, help="Number of cards held by other players")

    # Create a mutually exclusive group for played cards options
    played_cards_group = parser.add_mutually_exclusive_group()
    played_cards_group.add_argument(
        "-l",
        "--played-cards-list",
        type=lambda x: [int(i) for i in x.split(",")],
        help="Comma-separated list of specific played cards (e.g., '1,4,7' or '3')",
    )
    played_cards_group.add_argument(
        "-n", "--num-played-cards", type=int, default=0, help="Number of played cards to simulate"
    )

    parser.add_argument("-r", "--resolution", type=int, default=1, help="Space between consecutive card values")
    parser.add_argument("-m", "--model", type=str, choices=list(MODELS.keys()), default="GPT35_TURBO")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for model_name in MODELS:
            print(f"- {model_name}")
        return

    # Determine played cards value
    played_cards = args.played_cards_list if args.played_cards_list is not None else args.num_played_cards

    # Run simulations
    df = await run_simulations(
        player_cards=args.player_cards,
        other_cards=args.other_cards,
        played_cards=played_cards,
        model_name=args.model,
        resolution=args.resolution,
    )

    # Display results
    print("\nSimulation Results:")
    print(df)


if __name__ == "__main__":
    asyncio.run(main())
