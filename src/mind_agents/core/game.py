"""Game logic for The Mind card game."""

import random
from dataclasses import dataclass, field
from typing import Any

from mind_agents.llm.prompts.wait_n_seconds_prompts import play_game_template
from mind_agents.llm.types import Card
from mind_agents.llm.utilities import generate_play_content


@dataclass
class PlayerStats:
    """Track statistics for a player during the game."""

    cards_played: int = 0
    star_attempts: int = 0
    lives_lost: int = 0


@dataclass
class PlayerAction:
    """Represents a player's intended action."""

    player_id: int
    card: Card
    wait_time: float
    reason: str
    random_tiebreaker: float = field(default_factory=lambda: random.random())


@dataclass
class GameStateInfo:
    """Encapsulates game state information needed for prompts and display."""

    card: Card
    num_players: int
    total_other_cards: int
    all_cards: list[int]
    played_cards: list[int]
    dynamic_content: dict[str, Any]
    prompt_messages: list[dict[str, Any]]

    @classmethod
    def from_game_state(cls, game: "GameState", player_id: int, card: Card) -> "GameStateInfo":
        """Create GameStateInfo from a game state.

        Args:
            game: Current game state
            player_id: ID of the player
            card: Card being played

        Returns:
            GameStateInfo containing all calculated information
        """
        player = game.players[player_id - 1]
        all_cards = sorted([c.number for c in player.hand])
        other_players = [p for p in game.players if p.hand and p.id != player_id]
        total_other_cards = sum(len(p.hand) for p in other_players)

        # Generate content and prompt
        dynamic_content = generate_play_content(
            card_number=card.number,
            num_players=len(game.players),
            total_other_cards=total_other_cards,
            all_cards=all_cards,
            played_cards=game.played_cards,
        )
        prompt_messages = play_game_template.construct_prompt(dynamic_content)

        return cls(
            card=card,
            num_players=len(game.players),
            total_other_cards=total_other_cards,
            all_cards=all_cards,
            played_cards=game.played_cards,
            dynamic_content=dynamic_content,
            prompt_messages=prompt_messages,
        )


@dataclass
class Player:
    """Represents a player in the game."""

    id: int
    hand: list[Card] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Player {self.id} (cards: {[card.number for card in self.hand]})"


@dataclass
class GameState:
    """State of a game of The Mind."""

    num_players: int
    current_round: int = 1
    players: list[Player] = field(default_factory=list)
    played_cards: list[int] = field(default_factory=list)
    lives: int = field(default=3)
    stars: int = field(default=1)
    deck: list[int] = field(default_factory=list)  # List of available cards
    player_stats: list[PlayerStats] = field(default_factory=list)  # Stats for each player
    player_models: list[str] = field(default_factory=list)  # Changed from Model to str - Models used by each player
    cards_per_player: int = field(default=0)  # Number of cards per player in current round

    def __post_init__(self) -> None:
        """Initialize players and stats."""
        self.players = [Player(id=i + 1) for i in range(self.num_players)]
        self.player_stats = [PlayerStats() for _ in range(self.num_players)]
        self.deck = list(range(1, 101))  # Initialize deck with cards 1-100
        self.cards_per_player = self.current_round  # Initialize cards per player

    def deal_cards(self) -> None:
        """Deal cards for the current round.

        In The Mind, each player gets the same number of cards as the current round number.
        """
        # Create a deck of cards (1-100)
        available_cards = list(range(1, 101))
        random.shuffle(available_cards)

        # Clear existing hands
        for player in self.players:
            player.hand.clear()

        # Deal cards to each player
        cards_per_player = self.current_round
        for player in self.players:
            for _ in range(cards_per_player):
                if available_cards:  # Check if we still have cards to deal
                    card_number = available_cards.pop()
                    player.hand.append(Card(number=card_number))
            # Sort each player's hand
            player.hand.sort(key=lambda x: x.number)

    def play_all_lower_cards(self, threshold: int) -> set[int]:
        """Play all cards lower than the given threshold.

        Args:
            threshold: All cards lower than this number will be played

        Returns:
            Set of player IDs whose cards were played
        """
        affected_players = set()

        # Find all cards that need to be played
        for player in self.players:
            # Find all cards in hand lower than threshold
            lower_cards = [card for card in player.hand if card.number < threshold]
            if lower_cards:
                affected_players.add(player.id)
                # Remove these cards from hand
                player.hand = [card for card in player.hand if card.number >= threshold]
                # Add to played cards in order
                self.played_cards.extend(sorted(card.number for card in lower_cards))

        return affected_players

    def play_card(self, player_id: int, card: Card) -> tuple[bool, set[int]]:
        """Attempt to play a card.

        Args:
            player_id: ID of the player playing the card
            card: The card being played

        Returns:
            Tuple of:
            - bool: True if the play was valid, False if it was invalid
            - Set[int]: IDs of players whose cards were auto-played

        Side effects:
            - Removes the card from the player's hand if played
            - Adds the card to played_cards if valid
            - Updates lives if invalid
            - Auto-plays all lower cards if invalid
        """
        # Check if game is already over
        if self.game_over():
            return False, set()

        # Verify it's a valid play
        if self.played_cards and card.number <= self.played_cards[-1]:
            # Invalid play - lose one life
            self.lives -= 1

            # Find all cards that need to be auto-played (all cards lower than the highest played card)
            affected_players = set()
            cards_to_play = []  # List of (player_id, card) tuples

            # Collect all cards that need to be played
            for player in self.players:
                lower_cards = [c for c in player.hand if c.number < self.played_cards[-1]]
                if lower_cards:
                    affected_players.add(player.id)
                    for c in lower_cards:
                        cards_to_play.append((player.id, c))

            # Sort cards by number to play them in order
            cards_to_play.sort(key=lambda x: x[1].number)

            # Play all the cards in order
            for pid, c in cards_to_play:
                player = self.players[pid - 1]
                # Remove card from player's hand
                player.hand = [card for card in player.hand if card.number != c.number]
                # Add to played cards
                self.played_cards.append(c.number)
                # Sort played cards to maintain order
                self.played_cards.sort()

            return False, affected_players

        # Valid play - remove card from player's hand
        player = self.players[player_id - 1]
        player.hand = [c for c in player.hand if c.number != card.number]

        # Add to played cards
        self.played_cards.append(card.number)

        return True, set()

    def use_star(self) -> bool:
        """Use a star power.

        Returns:
            bool: True if star was available and used, False otherwise
        """
        if self.stars > 0:
            self.stars -= 1
            return True
        return False

    def round_complete(self) -> bool:
        """Check if the current round is complete.

        Returns:
            bool: True if all players have played all their cards
        """
        return all(len(player.hand) == 0 for player in self.players)

    def game_over(self) -> bool:
        """Check if the game is over.

        Returns:
            bool: True if players have lost all lives
        """
        return self.lives <= 0

    def advance_round(self) -> None:
        """Advance to the next round."""
        self.current_round += 1
        self.played_cards.clear()
        self.deal_cards()

    def __str__(self) -> str:
        """String representation of the game state."""
        status = [
            f"Round: {self.current_round}",
            f"Lives: {self.lives}",
            f"Stars: {self.stars}",
            f"Played cards: {self.played_cards}",
            "Players:",
        ]
        for player in self.players:
            status.append(f"  {player}")
        return "\n".join(status)
