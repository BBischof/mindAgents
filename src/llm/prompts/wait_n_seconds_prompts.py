"""Prompt template for playing The Mind card game."""


from llm.types import PromptComponent, PromptTemplate, Role, ToolSpec

# System component explaining the game rules and strategy
system_component = PromptComponent(
    role=Role.SYSTEM,
    dynamic_content="""
    You are playing The Mind card game.
    In this game, players must play their cards in ascending order without communicating.
    You must decide how long to wait before playing your card.

    Key Rules:
    1. You can only see your own cards
    2. Cards must be played in ascending order (1-100)
    3. If a card is played out of order, the team loses a life
    4. You must choose a wait time based on your card's value and the game state

    Strategy Tips:
    1. Consider the gap between your card and the last played card:
    - If your card is close to the last played card, wait less time
    - If there's a big gap, wait longer to allow others to play
    2. Account for how many cards other players still have:
    - More cards = higher chance of lower numbers
    - Adjust your wait time accordingly
    3. MOST IMPORTANT: Always wait longer if you think other players might have lower cards!

    You MUST use the wait_for_n_seconds tool to specify your wait time.
    Do not just describe what you want to do - use the tool to take action.
    """,
)

# User component for the current game state
user_component = PromptComponent(
    role=Role.USER,
    dynamic_content="""
    There are {num_players} players in the game.
    Other players have {total_other_cards} cards in total.
    The following cards have already been played in order: {played_cards}.
    I have these cards: {all_cards}. I must play my lowest card ({card_number}).
    What should I do?
    """,
)

# Define available tools
TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="wait_for_n_seconds",
        description="""
        Wait for N seconds before playing your card. Choose a wait time based on:
        """,
        parameters={
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this wait time was chosen",
            },
            "seconds": {
                "type": "integer",
                "description": "Number of seconds to wait before playing the card",
            },
        },
        required_params=["seconds", "reason"],
    )
]

# Create the template instance
play_game_template = PromptTemplate(
    name="play_game",
    version="1.0",
    components=[system_component, user_component],
    temperature=0,
    top_p=1,
    model_name="GPT35_TURBO",
    available_tools=TOOLS,
)
