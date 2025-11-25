"""
Custom Gymnasium Environment: Retail Store Clerk
A new retail store clerk learns where to stock items by exploring the store.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple


class RetailStoreEnv(gym.Env):
    """
    Custom environment simulating a retail store clerk learning where to stock items.
    
    The agent (clerk) starts at the back office (middle bottom) and must learn
    where each type of item belongs in the store.
    
    ### Action Space
    The action space is discrete with 6 actions:
    - 0: Move south (down)
    - 1: Move north (up)
    - 2: Move east (right)
    - 3: Move west (left)
    - 4: Pickup box (automatic at start, returns to office)
    - 5: Dropoff item (attempt to place item at current location)
    
    ### Observation Space
    The observation is an integer encoding the agent's position, 
    which item they're carrying, and whether they have an item.
    
    ### Rewards
    - At exact correct location: +1000
    - Within 4 blocks (Manhattan distance): +2
    - Within 8 blocks: -10
    - More than 8 blocks away: -500
    - Each step: -1 (to encourage efficiency)
    
    ### Starting State
    Agent starts at the back office (middle bottom of the grid) with a randomly
    assigned item to stock.
    
    ### Episode Termination
    Episode ends when the agent successfully places the item at the correct location
    or after a maximum number of steps.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 10):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Define item locations (destination for each item type)
        # Format: {item_id: (row, col)}
        self.item_locations = {
            0: (1, 1),    # Dairy (top-left)
            1: (1, 8),    # Frozen Foods (top-right)
            2: (4, 2),    # Produce (left-middle)
            3: (4, 7),    # Bakery (right-middle)
            4: (7, 4),    # Canned Goods (center-lower)
            5: (2, 5),    # Beverages (top-center)
            6: (6, 1),    # Cleaning Supplies (left-lower)
            7: (6, 8),    # Personal Care (right-lower)
        }
        
        self.num_items = len(self.item_locations)
        
        # Back office location (middle bottom)
        self.office_location = (grid_size - 1, grid_size // 2)
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: encode position, item type, and has_item status
        # State encoding: row * grid_size * (num_items + 1) + col * (num_items + 1) + item_encoding
        # where item_encoding is: item_id if has_item else num_items
        self.observation_space = spaces.Discrete(
            grid_size * grid_size * (self.num_items + 1)
        )
        
        # Movement actions
        self.action_to_direction = {
            0: (1, 0),   # South (down)
            1: (-1, 0),  # North (up)
            2: (0, 1),   # East (right)
            3: (0, -1),  # West (left)
        }
        
        # Initialize state variables
        self.agent_pos = None
        self.current_item = None
        self.has_item = False
        self.steps = 0
        self.max_steps = 200
        
        # Rendering
        self.window = None
        self.clock = None
        self.cell_size = 60
        
    def _get_obs(self) -> int:
        """Encode the current state as an integer observation."""
        row, col = self.agent_pos
        item_encoding = self.current_item if self.has_item else self.num_items
        return row * self.grid_size * (self.num_items + 1) + col * (self.num_items + 1) + item_encoding
    
    def _get_info(self) -> dict:
        """Return auxiliary information."""
        distance_to_target = self._manhattan_distance(
            self.agent_pos, 
            self.item_locations[self.current_item]
        )
        return {
            "agent_pos": self.agent_pos,
            "current_item": self.current_item,
            "target_location": self.item_locations[self.current_item],
            "has_item": self.has_item,
            "distance_to_target": distance_to_target,
            "steps": self.steps,
        }
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Start at back office
        self.agent_pos = self.office_location
        
        # Randomly assign an item to stock
        self.current_item = self.np_random.integers(0, self.num_items)
        
        # Start with item in hand
        self.has_item = True
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action: int):
        """Execute one step in the environment."""
        self.steps += 1
        reward = -1  # Small negative reward for each step
        terminated = False
        
        if action < 4:
            # Movement action
            direction = self.action_to_direction[action]
            new_row = self.agent_pos[0] + direction[0]
            new_col = self.agent_pos[1] + direction[1]
            
            # Check boundaries
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                self.agent_pos = (new_row, new_col)
        
        elif action == 4:
            # Pickup/Return to office
            # If not at office and don't have item, try to go back to office
            if not self.has_item and self.agent_pos == self.office_location:
                # Pick up a new item
                self.current_item = self.np_random.integers(0, self.num_items)
                self.has_item = True
                reward += 0  # Neutral for picking up
        
        elif action == 5:
            # Dropoff item
            if self.has_item:
                target_location = self.item_locations[self.current_item]
                distance = self._manhattan_distance(self.agent_pos, target_location)
                
                if distance == 0:
                    # Perfect placement!
                    reward = 1000
                    terminated = True
                elif distance <= 4:
                    # Within 4 blocks
                    reward = 2
                    self.has_item = False
                elif distance <= 8:
                    # Within 8 blocks
                    reward = -10
                    self.has_item = False
                else:
                    # Too far away
                    reward = -500
                    self.has_item = False
                
                # If item was dropped (not at correct location), go back to office
                if not terminated and not self.has_item:
                    # Agent needs to return to office for a new item
                    pass  # In next reset, they'll get a new item
            else:
                # Tried to drop off but don't have an item
                reward = -10
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame of the environment."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((255, 255, 255))  # White background
        
        # Draw grid
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, i * self.cell_size),
                (self.grid_size * self.cell_size, i * self.cell_size),
                1
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (i * self.cell_size, 0),
                (i * self.cell_size, self.grid_size * self.cell_size),
                1
            )
        
        # Draw item locations
        item_colors = [
            (173, 216, 230),  # Light blue - Dairy
            (135, 206, 250),  # Sky blue - Frozen
            (144, 238, 144),  # Light green - Produce
            (255, 218, 185),  # Peach - Bakery
            (255, 255, 153),  # Light yellow - Canned
            (255, 182, 193),  # Light pink - Beverages
            (221, 160, 221),  # Plum - Cleaning
            (255, 192, 203),  # Pink - Personal Care
        ]
        
        for item_id, (row, col) in self.item_locations.items():
            color = item_colors[item_id]
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    col * self.cell_size + 5,
                    row * self.cell_size + 5,
                    self.cell_size - 10,
                    self.cell_size - 10
                )
            )
            # Draw item number
            if pygame.font.get_init():
                font = pygame.font.Font(None, 24)
                text = font.render(str(item_id), True, (0, 0, 0))
                text_rect = text.get_rect(center=(
                    col * self.cell_size + self.cell_size // 2,
                    row * self.cell_size + self.cell_size // 2
                ))
                canvas.blit(text, text_rect)
        
        # Draw back office
        office_row, office_col = self.office_location
        pygame.draw.rect(
            canvas,
            (128, 128, 128),  # Gray for office
            pygame.Rect(
                office_col * self.cell_size + 5,
                office_row * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10
            )
        )
        
        # Draw agent
        agent_row, agent_col = self.agent_pos
        pygame.draw.circle(
            canvas,
            (255, 0, 0) if self.has_item else (0, 0, 255),  # Red if carrying, blue otherwise
            (
                agent_col * self.cell_size + self.cell_size // 2,
                agent_row * self.cell_size + self.cell_size // 2
            ),
            self.cell_size // 3
        )
        
        # Draw current item indicator if carrying
        if self.has_item and pygame.font.get_init():
            font = pygame.font.Font(None, 20)
            text = font.render(f"Item: {self.current_item}", True, (255, 255, 255))
            text_rect = text.get_rect(center=(
                agent_col * self.cell_size + self.cell_size // 2,
                agent_row * self.cell_size + self.cell_size // 2
            ))
            canvas.blit(text, text_rect)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Register the environment
gym.register(
    id='RetailStore-v0',
    entry_point='retail_store_env:RetailStoreEnv',
    max_episode_steps=200,
)


