"""Script to simulate multiple scenarios of The Mind game."""

import argparse
import asyncio
import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from mind_agents.prompt_assets.types import Model


async def run_test_scenario(
    test_state: dict[str, Any],
    model: Model = Model.CLAUDE3_SONNET,
    verbose: bool = False,
) -> float:
    """Run a single test scenario using the play_the_mind.py script.

    Args:
        test_state: Dictionary containing the test state (e.g. {"1": [5], "H": 1, "P": []})
        model: The model to use for the test player
        verbose: Whether to run in verbose mode

    Returns:
        The wait time returned by the model

    Raises:
        ValueError: If H (other cards) is 0
    """
    if test_state["H"] == 0:
        raise ValueError("Cannot run simulation with 0 other cards - would cause division by zero in game logic")

    # Convert test state to JSON string
    json_str = json.dumps(test_state)

    # Construct command
    cmd = ["python", "-m", "src.mind_agents.play_the_mind", "--test", json_str]
    if verbose:
        cmd.append("-v")
    # Add model parameter
    cmd.extend(["--models", model.name])

    # Run command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        # Parse the wait time from output
        wait_time = float(result.stdout.strip())
        return wait_time
    except ValueError:
        print(f"Error parsing output: {result.stdout}")
        return float("nan")


def get_ordered_tuples(size: int, max_val: int = 100, resolution: int = 1) -> list[tuple[int, ...]]:
    """Get all ordered tuples of a given size with values spaced by resolution.

    Args:
        size: Size of each tuple
        max_val: Maximum value to consider (default 100)
        resolution: Space between consecutive card values (default 1)

    Returns:
        List of ordered tuples where each value is spaced by resolution
    """
    # Generate possible values based on resolution
    possible_values = list(range(1, max_val + 1, resolution))

    # Get all possible combinations
    all_combs = list(combinations(possible_values, size))

    # Only keep combinations where elements are in ascending order
    # and no equal values (which shouldn't happen with resolution > 1 anyway)
    valid_combs = []
    for comb in all_combs:
        # Check if values are strictly increasing (no equals)
        if all(comb[i] < comb[i + 1] for i in range(len(comb) - 1)):
            valid_combs.append(comb)

    return valid_combs


async def run_simulations(
    player_cards: int = 1,
    other_cards: int = 1,
    played_cards: int = 0,
    model: Model = Model.CLAUDE3_SONNET,
    max_val: int = 100,
    resolution: int = 1,
) -> pd.DataFrame:
    """Run simulations with specified parameters.

    Args:
        player_cards: Number of cards the player has
        other_cards: Number of cards held by other players
        played_cards: Number of played cards to consider
        model: The model to use for the test player
        max_val: Maximum card value to consider
        resolution: Space between consecutive card values

    Returns:
        DataFrame containing test inputs and results
    """
    results = []

    # Get all possible player card combinations with resolution
    player_card_combinations = get_ordered_tuples(player_cards, max_val, resolution)

    # Get all possible played card combinations with resolution
    played_card_combinations = get_ordered_tuples(played_cards, max_val, resolution) if played_cards > 0 else [()]

    # Calculate total number of iterations
    total_iterations = len(player_card_combinations) * len(played_card_combinations)

    # Create progress bar
    with tqdm(total=total_iterations, desc="Running simulations") as pbar:
        # For each combination of player cards
        for player_cards_tuple in player_card_combinations:
            # For each combination of played cards
            for played_cards_tuple in played_card_combinations:
                # Skip if any player cards are in played cards
                if any(card in played_cards_tuple for card in player_cards_tuple):
                    pbar.update(1)
                    continue

                # Skip if played cards aren't all lower than player's lowest card
                if played_cards_tuple and max(played_cards_tuple) > min(player_cards_tuple):
                    pbar.update(1)
                    continue

                test_state = {
                    "1": list(player_cards_tuple),
                    "H": other_cards,
                    "P": list(played_cards_tuple),
                }

                wait_time = await run_test_scenario(test_state, model=model)

                # Store results
                results.append(
                    {
                        "player_cards": player_cards_tuple,
                        "other_cards": other_cards,
                        "played_cards": played_cards_tuple,
                        "wait_time": wait_time,
                        "model": model.name,
                    }
                )

                pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


async def main() -> None:
    """Main function to run simulations and display results."""
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Run Mind game simulations")
    parser.add_argument(
        "-p",
        "--player-cards",
        type=int,
        default=1,
        help="Number of cards the player has",
    )
    parser.add_argument(
        "-o",
        "--other-cards",
        type=int,
        default=1,
        help="Number of cards held by other players",
    )
    parser.add_argument(
        "-l",
        "--played-cards",
        type=int,
        default=0,
        help="Number of played cards to consider",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=[m.name for m in Model],
        default=Model.GPT35.name,
        help="Model to use for the test player",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=1,
        help="Space between consecutive card values (default: 1)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Display all available models and their descriptions",
    )
    args = parser.parse_args()

    if args.list_models:
        from mind_agents.play_the_mind import display_available_models

        display_available_models()
        return

    # Validate parameters
    if args.player_cards < 1:
        parser.error("Player must have at least 1 card")
    if args.played_cards < 0:
        parser.error("Number of played cards cannot be negative")
    if args.resolution < 1:
        parser.error("Resolution must be at least 1")

    print("Running simulations with parameters:")
    print(f"- Player cards: {args.player_cards}")
    print(f"- Other cards: {args.other_cards}")
    print(f"- Played cards: {args.played_cards}")
    print(f"- Model: {args.model}")
    print(f"- Resolution: {args.resolution}")

    try:
        df = await run_simulations(
            player_cards=args.player_cards,
            other_cards=args.other_cards,
            played_cards=args.played_cards,
            model=Model[args.model],
            resolution=args.resolution,
        )

        # Display results
        print("\nSimulation Results:")
        print(df.to_string(index=False))

        # Create filename with parameters
        filename = f"""simulation_p{args.player_cards}_h{args.other_cards}_played{args.played_cards}_r{args.resolution}_{args.model.lower()}.csv"""  # noqa: E501

        # Save results
        output_dir = Path("simulation_results")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / filename, index=False)
        print(f"\nResults saved to {output_dir}/{filename}")

    except ValueError as e:
        print(f"\nError: {str(e)}")
        parser.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
