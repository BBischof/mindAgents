"""Functions for generating prompt content from game state."""

from typing import Any


def generate_play_content(
    card_number: int,
    num_players: int,
    total_other_cards: int,
    all_cards: list[int],
    played_cards: list[int] | None = None,
) -> dict[str, Any]:
    """Generate content for playing a card.

    Args:
        card_number: The number on the card (1-100)
        num_players: Number of players in the game
        total_other_cards: Total number of cards held by other players
        all_cards: List of all cards in the player's hand
        played_cards: List of cards already played in ascending order

    Returns:
        Dict containing the dynamic content for the prompt template

    Raises:
        ValueError: If card number is not between 1 and 100
        ValueError: If num_players is less than 1
        ValueError: If played cards are not in ascending order
    """
    if not 1 <= card_number <= 100:
        raise ValueError("Card number must be between 1 and 100")
    if num_players < 1:
        raise ValueError("Game must have at least 1 player")

    played_cards = played_cards or []

    # Validate played cards are in ascending order
    if played_cards and any(played_cards[i] >= played_cards[i + 1] for i in range(len(played_cards) - 1)):
        raise ValueError("Played cards must be in ascending order")

    return {
        "card_number": card_number,
        "num_players": num_players,
        "total_other_cards": total_other_cards,
        "all_cards": all_cards,
        "played_cards": played_cards,
    }
