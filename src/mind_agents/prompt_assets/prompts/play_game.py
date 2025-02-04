"""Prompt template for playing The Mind card game."""


from ..types import Model, PromptComponent, PromptTemplate, Role, ToolSpec

# Define available tools
TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="wait_for_n_seconds",
        description="""
        Wait for N seconds before playing your card. Choose a wait time based on:
        1. Your card's value (1-100):
           - Cards 1-20: Use 0.5-2 seconds
           - Cards 21-60: Use 2-5 seconds
           - Cards 61-100: Use 5-10 seconds
        2. The gap between your card and the last played card:
           - Small gap (1-10): Add 0-1 seconds
           - Medium gap (11-30): Add 1-3 seconds
           - Large gap (31+): Add 3-5 seconds
        3. Number of cards other players have:
           - More cards = slightly longer wait times

        The maximum wait time is 15 seconds.
        """,
        parameters={
            "seconds": {
                "type": "number",
                "description": "Number of seconds to wait before playing the card",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this wait time was chosen",
            },
        },
        required_params=["seconds", "reason"],
    )
]

# Create a mapping of tool names to tool instances for easy lookup
AVAILABLE_TOOLS = {tool.name: tool for tool in TOOLS}


def create_play_game_template() -> PromptTemplate:
    """Create the prompt template for playing The Mind card game."""
    system_content = """
    You are playing The Mind card game.
    In this game, players must play their cards in ascending order without communicating.
    You must decide how long to wait before playing your card.

Key Rules:
1. You can only see your own cards
2. Cards must be played in ascending order (1-100)
3. If a card is played out of order, the team loses a life
4. You must choose a wait time based on your card's value and the game state

Strategy Tips:
1. Lower cards (1-20) should be played quickly (0.5-2 seconds)
2. Higher cards (61-100) need longer waits (5-10 seconds)
3. Consider the gap between your card and the last played card:
   - If your card is close to the last played card, wait less time
   - If there's a big gap, wait longer to allow others to play
4. Account for how many cards other players still have:
   - More cards = higher chance of lower numbers
   - Adjust your wait time accordingly
5. Be more cautious when the team has few lives left

Your goal is to choose an appropriate wait time that allows lower cards
to be played first while not waiting unnecessarily long for high cards."""

    return PromptTemplate(
        name="play_game",
        version="1.0",
        components=[
            PromptComponent(
                role=Role.SYSTEM,
                static_content=system_content,
            ),
        ],
        temperature=0.7,
        top_p=0.9,
        model=Model.GPT35,
        available_tools=[tool.name for tool in TOOLS],
    )


# Create the template instance
play_game_template = create_play_game_template()


def generate_play_content(
    card_number: int,
    num_players: int,
    cards_per_player: int,
    played_cards: list[int] | None = None,
) -> dict[str, int | list[int]]:
    """Generate content for playing a card.

    Args:
        card_number: The number on the card (1-100)
        num_players: Number of players in the game
        cards_per_player: Number of cards each player has
        played_cards: List of cards already played in ascending order, defaults to empty list

    Returns:
        Dict containing the card number and game state information

    Raises:
        ValueError: If card number is not between 1 and 100
        ValueError: If num_players is less than 1
        ValueError: If cards_per_player is less than 1
        ValueError: If played cards are not in ascending order
    """
    if not 1 <= card_number <= 100:
        raise ValueError("Card number must be between 1 and 100")
    if num_players < 1:
        raise ValueError("Game must have at least 1 player")
    if cards_per_player < 1:
        raise ValueError("Each player must have at least 1 card")

    played_cards = played_cards or []

    # Validate played cards are in ascending order
    if played_cards and any(played_cards[i] >= played_cards[i + 1] for i in range(len(played_cards) - 1)):
        raise ValueError("Played cards must be in ascending order")

    return {
        "card_number": card_number,
        "num_players": num_players,
        "cards_per_player": cards_per_player,
        "played_cards": played_cards,
    }
