"""
Tetris Environment for AI Training
A Pygame-based Tetris implementation structured as an RL environment.
"""

import pygame
import numpy as np
import random
from typing import Tuple, Dict, Optional, List


class TetrisEnv:
    """
    Tetris game environment for reinforcement learning.

    Action space:
        0 = move left
        1 = move right
        2 = rotate clockwise
        3 = hard drop (drop to bottom instantly)
    """

    # Game constants
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    CELL_SIZE = 30
    RENDER_TICK_SPEED = 5

    # Colors (RGB)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)

    # Tetromino colors
    COLORS = [
        (0, 255, 255),    # I - Cyan
        (255, 255, 0),    # O - Yellow
        (128, 0, 128),    # T - Purple
        (0, 255, 0),      # S - Green
        (255, 0, 0),      # Z - Red
        (0, 0, 255),      # J - Blue
        (255, 165, 0)     # L - Orange
    ]

    @staticmethod
    def _build_shapes() -> Dict[str, List[List[Tuple[int, int]]]]:
        """Construct normalized rotation states for each tetromino."""
        raw_shapes = {
            'I': [
                (
                    "....",
                    "####",
                    "....",
                    "....",
                ),
                (
                    ".#..",
                    ".#..",
                    ".#..",
                    ".#..",
                ),
            ],
            'O': [
                (
                    ".##.",
                    ".##.",
                    "....",
                    "....",
                ),
            ],
            'T': [
                (
                    ".#..",
                    "###.",
                    "....",
                    "....",
                ),
                (
                    ".#..",
                    ".##.",
                    ".#..",
                    "....",
                ),
                (
                    "....",
                    "###.",
                    ".#..",
                    "....",
                ),
                (
                    ".#..",
                    "##..",
                    ".#..",
                    "....",
                ),
            ],
            'S': [
                (
                    "..##",
                    ".##.",
                    "....",
                    "....",
                ),
                (
                    ".#..",
                    ".##.",
                    "..#.",
                    "....",
                ),
            ],
            'Z': [
                (
                    ".##.",
                    "..##",
                    "....",
                    "....",
                ),
                (
                    "..#.",
                    ".##.",
                    ".#..",
                    "....",
                ),
            ],
            'J': [
                (
                    "#...",
                    "###.",
                    "....",
                    "....",
                ),
                (
                    ".##.",
                    ".#..",
                    ".#..",
                    "....",
                ),
                (
                    "....",
                    "###.",
                    "..#.",
                    "....",
                ),
                (
                    ".#..",
                    ".#..",
                    "##..",
                    "....",
                ),
            ],
            'L': [
                (
                    "...#",
                    ".###",
                    "....",
                    "....",
                ),
                (
                    ".#..",
                    ".#..",
                    ".##.",
                    "....",
                ),
                (
                    "....",
                    "###.",
                    "#...",
                    "....",
                ),
                (
                    "##..",
                    ".#..",
                    ".#..",
                    "....",
                ),
            ],
        }

        shapes: Dict[str, List[List[Tuple[int, int]]]] = {}
        for name, rotations in raw_shapes.items():
            normalized: List[List[Tuple[int, int]]] = []
            for rotation in rotations:
                coords: List[Tuple[int, int]] = []
                for row_idx, row in enumerate(rotation):
                    for col_idx, cell in enumerate(row):
                        if cell == '#':
                            coords.append((row_idx, col_idx))

                min_row = min(r for r, _ in coords)
                min_col = min(c for _, c in coords)
                normalized.append([(r - min_row, c - min_col) for r, c in coords])
            shapes[name] = normalized
        return shapes

    # Tetromino shapes (represented as rotation states)
    SHAPES = _build_shapes()

    NUMBER_OF_SHAPES = len(SHAPES)
    NUMBER_OF_ROTATIONS = max(len(rotations) for rotations in SHAPES.values())
    STATE_SIZE = BOARD_WIDTH * BOARD_HEIGHT + NUMBER_OF_SHAPES +  NUMBER_OF_ROTATIONS
    NUMBER_OF_ACTIONS = 4  # Left, Right, Rotate, Hard Drop

    def __init__(self, render_mode: bool = True):
        """
        Initialize the Tetris environment.

        Args:
            render_mode: If True, enables Pygame rendering
        """
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.BOARD_WIDTH * self.CELL_SIZE + 200,
                 self.BOARD_HEIGHT * self.CELL_SIZE)
            )
            pygame.display.set_caption("Tetris AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Initialize game state
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.

        Returns:
            Initial board state as numpy array
        """
        # Board: 0 = empty, 1-7 = colored blocks
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=int)

        # Game statistics
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.total_pieces = 0

        # Current piece state
        self.current_piece = None
        self.current_shape_name = None
        self.current_rotation = 0
        self.current_pos = [0, 0]  # [row, col]
        self.current_color_idx = 0

        # Spawn first piece
        self._spawn_piece()

        return self.get_state()

    def _spawn_piece(self) -> bool:
        """
        Spawn a new tetromino at the top of the board.

        Returns:
            True if piece spawned successfully, False if game over
        """
        # Select random piece
        self.current_shape_name = random.choice(list(self.SHAPES.keys()))
        self.current_rotation = 0
        self.current_piece = self.SHAPES[self.current_shape_name][self.current_rotation]
        self.current_color_idx = list(self.SHAPES.keys()).index(self.current_shape_name)

        # Start position (top center)
        self.current_pos = [0, self.BOARD_WIDTH // 2 - 1]

        # Check if spawning is possible (game over check)
        if self._check_collision(self.current_piece, self.current_pos):
            self.game_over = True
            return False

        self.total_pieces += 1
        return True

    def _check_collision(self, piece: list, pos: list) -> bool:
        """
        Check if piece collides with board boundaries or other pieces.

        Args:
            piece: List of (row, col) offsets for the piece
            pos: [row, col] position of piece anchor

        Returns:
            True if collision detected, False otherwise
        """
        for dy, dx in piece:
            row = pos[0] + dy
            col = pos[1] + dx

            # Check boundaries
            if row < 0 or row >= self.BOARD_HEIGHT:
                return True
            if col < 0 or col >= self.BOARD_WIDTH:
                return True

            # Check existing blocks
            if self.board[row, col] != 0:
                return True

        return False

    def _lock_piece(self):
        """Lock the current piece into the board."""
        for dy, dx in self.current_piece:
            row = self.current_pos[0] + dy
            col = self.current_pos[1] + dx
            if 0 <= row < self.BOARD_HEIGHT and 0 <= col < self.BOARD_WIDTH:
                self.board[row, col] = self.current_color_idx + 1

    def _clear_lines(self) -> int:
        """
        Clear completed lines and update score.

        Returns:
            Number of lines cleared
        """
        lines_to_clear = []

        # Find completed lines
        for row in range(self.BOARD_HEIGHT):
            if np.all(self.board[row] != 0):
                lines_to_clear.append(row)

        # Remove completed lines
        if lines_to_clear:
            # Remove lines and add empty lines at top
            self.board = np.delete(self.board, lines_to_clear, axis=0)
            empty_lines = np.zeros((len(lines_to_clear), self.BOARD_WIDTH), dtype=int)
            self.board = np.vstack([empty_lines, self.board])

        # Update statistics
        num_lines = len(lines_to_clear)
        self.lines_cleared += num_lines

        # Scoring system (standard Tetris scoring)
        line_scores = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_scores.get(num_lines, 0)

        return num_lines

    def _move_left(self) -> bool:
        """Move piece left. Returns True if successful."""
        if self.current_piece is None:
            return False
        new_pos = [self.current_pos[0], self.current_pos[1] - 1]
        if not self._check_collision(self.current_piece, new_pos):
            self.current_pos = new_pos
            return True
        return False

    def _move_right(self) -> bool:
        """Move piece right. Returns True if successful."""
        if self.current_piece is None:
            return False
        new_pos = [self.current_pos[0], self.current_pos[1] + 1]
        if not self._check_collision(self.current_piece, new_pos):
            self.current_pos = new_pos
            return True
        return False

    def _rotate(self) -> bool:
        """Rotate piece clockwise. Returns True if successful."""
        if self.current_piece is None or self.current_shape_name is None:
            return False
        shapes = self.SHAPES[self.current_shape_name]
        new_rotation = (self.current_rotation + 1) % len(shapes)
        new_piece = shapes[new_rotation]

        # Try simple wall kicks to allow rotation near boundaries
        kick_offsets = [(0, 0), (0, -1), (0, 1), (0, -2), (0, 2), (-1, 0)]
        for dy, dx in kick_offsets:
            new_pos = [self.current_pos[0] + dy, self.current_pos[1] + dx]
            if not self._check_collision(new_piece, new_pos):
                self.current_rotation = new_rotation
                self.current_piece = new_piece
                self.current_pos = new_pos
                return True
        return False

    def _soft_drop(self) -> bool:
        """
        Move piece down one cell.

        Returns:
            True if moved successfully, False if locked
        """
        if self.current_piece is None:
            return False
        new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
        if not self._check_collision(self.current_piece, new_pos):
            self.current_pos = new_pos
            return True
        else:
            # Lock piece and spawn new one
            self._lock_piece()
            return False

    def _hard_drop(self) -> int:
        """
        Drop piece to the bottom instantly.

        Returns:
            Number of rows dropped
        """
        if self.current_piece is None:
            return 0
        rows_dropped = 0
        while True:
            new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
            if not self._check_collision(self.current_piece, new_pos):
                self.current_pos = new_pos
                rows_dropped += 1
            else:
                break

        # Lock the piece
        self._lock_piece()
        return rows_dropped

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one action in the environment.

        Args:
            action: Integer action (0=left, 1=right, 2=rotate, 3=hard drop)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.game_over:
            return self.get_state(), 0, True, self._get_info()

        reward = 0
        piece_locked = False

        # Execute action
        if action == 0:  # Move left
            self._move_left()
        elif action == 1:  # Move right
            self._move_right()
        elif action == 2:  # Rotate
            self._rotate()
        elif action == 3:  # Hard drop (now action index 3)
            rows_dropped = self._hard_drop()
            reward += rows_dropped * 2  # Bonus for hard drop
            piece_locked = True
        else:
            # Invalid action: ignore (could also raise)
            pass

        # For all non-hard-drop actions, advance the piece by one row (soft drop).
        # If the soft drop cannot move the piece, it becomes locked and will be processed below.
        if action != 3 and not piece_locked:
            moved = self._soft_drop()
            if not moved:
                piece_locked = True

        # Handle piece locking
        if piece_locked:
            # Clear lines
            lines_cleared = self._clear_lines()

            # Smaller, scaled rewards for clearing lines (keep relative values)
            # Use modest values so cumulative reward doesn't explode
            line_rewards = {0: 0.0, 1: 1.0, 2: 2.5, 3: 4.0, 4: 6.0}
            reward += float(line_rewards.get(lines_cleared, 1.0))

            # Penalty for increasing height (scaled down)
            height_penalty = -self._get_max_height()
            reward += float(height_penalty)

            # Penalty for holes
            holes_penalty = -self._calculate_holes()
            reward += float(holes_penalty)

            # Spawn next piece
            if not self._spawn_piece():
                # Smaller game over penalty
                reward -= 5.0
                self.game_over = True

        return encode_state(self), reward, self.game_over, self._get_info()
    
    def _get_max_height(self) -> float:
        """Calculate maximum column height."""
        heights = []
        for col in range(self.BOARD_WIDTH):
            height = 0
            for row in range(self.BOARD_HEIGHT):
                if self.board[row, col] != 0:
                    height = self.BOARD_HEIGHT - row
                    break
            heights.append(height)
        return max(heights)

    def get_features(self) -> Dict[str, float]:
        """
        Extract useful features from the current board state.

        Returns:
            Dictionary containing:
                - aggregate_height: Sum of column heights
                - holes: Number of empty cells with the blocks above
                - bumpiness: Sum of absolute height differences between adjacent columns
                - completed_lines: Number of complete lines (before clearing)
        """
        return {
            'aggregate_height': self._calculate_aggregate_height(),
            'holes': self._calculate_holes(),
            'bumpiness': self._calculate_bumpiness(),
            'completed_lines': self._calculate_completed_lines()
        }

    def _calculate_aggregate_height(self) -> float:
        """Calculate sum of all column heights."""
        heights = []
        for col in range(self.BOARD_WIDTH):
            height = 0
            for row in range(self.BOARD_HEIGHT):
                if self.board[row, col] != 0:
                    height = self.BOARD_HEIGHT - row
                    break
            heights.append(height)
        return sum(heights)

    def _calculate_holes(self) -> int:
        """Count holes (empty cells with at least one block above)."""
        holes = 0
        for col in range(self.BOARD_WIDTH):
            block_found = False
            for row in range(self.BOARD_HEIGHT):
                if self.board[row, col] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def _calculate_bumpiness(self) -> float:
        """Calculate sum of absolute height differences between adjacent columns."""
        heights = []
        for col in range(self.BOARD_WIDTH):
            height = 0
            for row in range(self.BOARD_HEIGHT):
                if self.board[row, col] != 0:
                    height = self.BOARD_HEIGHT - row
                    break
            heights.append(height)

        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        return bumpiness

    def _calculate_completed_lines(self) -> int:
        """Count number of completed lines."""
        completed = 0
        for row in range(self.BOARD_HEIGHT):
            if np.all(self.board[row] != 0):
                completed += 1
        return completed

    def _get_info(self) -> Dict:
        """Get additional game information."""
        if not self.render_mode:
            return {}
        return {
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'total_pieces': self.total_pieces,
            'game_over': self.game_over
        }

    def get_state(self) -> np.ndarray:
        """Return the encoded state for the environment."""
        # encode_state is defined later in this module
        return encode_state(self)

    def render(self):
        """Render the game using Pygame."""
        if not self.render_mode or self.screen is None:
            return

        # Clear screen
        self.screen.fill(self.BLACK)

        # Draw board grid
        for row in range(self.BOARD_HEIGHT):
            for col in range(self.BOARD_WIDTH):
                rect = pygame.Rect(
                    col * self.CELL_SIZE,
                    row * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )

                # Draw cell
                if self.board[row, col] != 0:
                    color = self.COLORS[self.board[row, col] - 1]
                    pygame.draw.rect(self.screen, color, rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.GRAY, rect, 1)

        # Draw current piece
        if not self.game_over and self.current_piece is not None:
            color = self.COLORS[self.current_color_idx]
            for dy, dx in self.current_piece:
                row = self.current_pos[0] + dy
                col = self.current_pos[1] + dx
                if 0 <= row < self.BOARD_HEIGHT and 0 <= col < self.BOARD_WIDTH:
                    rect = pygame.Rect(
                        col * self.CELL_SIZE,
                        row * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, self.GRAY, rect, 1)

        # Draw stats panel
        panel_x = self.BOARD_WIDTH * self.CELL_SIZE + 10
        stats_texts = [
            f"Score: {self.score}",
            f"Lines: {self.lines_cleared}",
            f"Pieces: {self.total_pieces}",
            "",
            "Features:",
            f"Height: {self._get_max_height():.0f}",
            f"Holes: {self._calculate_holes()}",
            f"Bumpy: {self._calculate_bumpiness():.0f}",
        ]

        for i, text in enumerate(stats_texts):
            surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(surface, (panel_x, 20 + i * 40))

        # Game over message
        if self.game_over:
            game_over_surface = self.font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_surface.get_rect(
                center=(self.BOARD_WIDTH * self.CELL_SIZE // 2,
                        self.BOARD_HEIGHT * self.CELL_SIZE // 2)
            )
            self.screen.blit(game_over_surface, text_rect)

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.RENDER_TICK_SPEED)

    def close(self):
        """Clean up Pygame resources."""
        if self.render_mode:
            pygame.quit()


# Example usage for testing
if __name__ == "__main__":
    # Create an environment with rendering
    env = TetrisEnv(render_mode=True)

    print("Tetris Environment Test")
    print("Actions: 0=Left, 1=Right, 2=Rotate, 3=Hard Drop")
    print("\nRunning random agent...")

    state = env.reset()
    done = False
    step_count = 0

    while not done:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        # Random action (soft drop removed)
        action = random.choice([0, 1, 2, 3])

        # Take step
        next_state, reward, done, info = env.step(action)

        # Render
        env.render()

        step_count += 1

        # Print info every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: Score={info['score']}, Lines={info['lines_cleared']}")
            features = env.get_features()
            print(f"  Features: {features}")

    print(f"\nGame Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")

    # Keep window open briefly
    pygame.time.wait(2000)
    env.close()


def encode_shape(env: TetrisEnv) -> np.ndarray:
    shape = np.zeros(env.NUMBER_OF_SHAPES, dtype=int)
    if env.current_shape_name is not None:
        shape[env.current_color_idx] = 1
    return shape

def encode_rotation(env: TetrisEnv) -> np.ndarray:
    rotation = np.zeros(env.NUMBER_OF_ROTATIONS, dtype=int)
    if env.current_shape_name is not None:
        rotation[env.current_rotation] = 1
    return rotation

def encode_state(env: TetrisEnv) -> np.ndarray:
    board_flat = env.board.flatten()
    shape_enc = encode_shape(env)
    rotation_enc = encode_rotation(env)
    return np.concatenate([board_flat, shape_enc, rotation_enc])