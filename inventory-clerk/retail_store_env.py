"""
Custom Gymnasium Environment: Retail Store Clerk
A new retail store clerk learns where to stock items by exploring the store.
Now the clerk navigates with a movable stocking cart instead of a fixed back office.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import json
from typing import Optional, Tuple, Dict, List, Set


class RetailStoreEnv(gym.Env):
    """
    Custom environment simulating a retail store clerk learning where to stock items.
    
    The agent (clerk) starts at the stocking cart (middle bottom) and must learn
    where each type of item belongs in the store.
    
    ### Action Space
    The action space is discrete with 8 actions:
    - 0: Move south (down)
    - 1: Move north (up)
    - 2: Move east (right)
    - 3: Move west (left)
    - 4: Pickup box (at stocking cart)
    - 5: Dropoff item (attempt to place item at current location)
    - 6: Start pushing cart (when adjacent to cart)
    - 7: Leave cart (stop pushing; cart stays at its current cell)
    
    ### Observation Space
    The observation is an integer encoding the agent's position, 
    which item they're carrying, and whether they have an item.
    
    ### Rewards
    - At exact correct location: **+1000**
    - Within 4 blocks (Manhattan distance): **+2**
    - Within 8 blocks: **-10**
    - More than 8 blocks away: **-500**
    - Each step: **-1** (to encourage efficiency)
    - Adjacent to a customer NPC (on or 4-neighbor cell): **-800** additional penalty
    
    ### Starting State
    Agent starts at the stocking cart (middle bottom of the grid) with a randomly
    assigned item to stock.
    
    ### Episode Termination
    Episode ends when the agent successfully places the item at the correct location
    or after a maximum number of steps.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 20, enable_customers: bool = True):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.enable_customers = enable_customers

        # Load item configuration from JSON
        script_dir = os.path.dirname(os.path.abspath(__file__))
        inventory_dir = os.path.join(script_dir, "inventory")
        items_path = os.path.join(inventory_dir, "items.json")

        self.items_config: Dict[str, dict] = self._load_json(items_path, root_key="items")

        # Stable ordering of items: use the order from items.json
        self.item_names: List[str] = list(self.items_config.keys())

        # ------------------------------------------------------------------
        # Grid layout (20x20) encoding departments and static obstacles.
        # Each cell is:
        #   '.' : empty walkable cell
        #   '*' : obstacle (blocked for agent, cart, and customers)
        #   '0'-'9' : department ID for an item location
        # ------------------------------------------------------------------
        self.grid_layout: List[List[str]] = [
            list("*..............*****"),  # row 0
            list("0..................."),  # row 1
            list("*..................."),  # row 2: dept 0 and 1
            list("........**..**.....*"),  # row 3
            list("...*....5....*.....*"),  # row 4: dept 5
            list("...*............*..6"),  # row 5
            list("...2............*..*"),  # row 6: dept 2 and 3
            list("...*....*....*......"),  # row 7
            list("...*....**..*3......"),  # row 8
            list("*..................."),  # row 9
            list("*..................."),  # row 10: dept 4
            list("........*4..**......"),  # row 11
            list("..***...*....*...***"),  # row 12
            list("...................."),  # row 13
            list("..***............*7*"),  # row 14: dept 6 and 7
            list("........*....*......"),  # row 15
            list("........**..**......"),  # row 16
            list("*..................."),  # row 17
            list("*..................."),  # row 18
            list("*1*................."),  # row 19
        ]

        # Obstacles: initialize from layout and then add special ones around
        # department 1 (one cell two left, one cell directly below).
        self.obstacles: Set[Tuple[int, int]] = set()
        dept_locations: Dict[int, Tuple[int, int]] = {}

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid_layout[r][c]
                if cell == "*":
                    self.obstacles.add((r, c))
                elif cell.isdigit():
                    dept_id = int(cell)
                    dept_locations[dept_id] = (r, c)

        # Ensure department 1 exists before placing related obstacles
        dept1_pos: Optional[Tuple[int, int]] = dept_locations.get(1)
        if dept1_pos is not None:
            r, c = dept1_pos
            # One cell two steps to the left of department 1
            left_col = c - 2
            if 0 <= left_col < self.grid_size:
                self.obstacles.add((r, left_col))
            # One cell directly below department 1
            below_row = r + 1
            if 0 <= below_row < self.grid_size:
                self.obstacles.add((below_row, c))

        # Build item -> department, locations, and colors using the layout
        self.item_departments: List[int] = []
        self.item_locations: Dict[int, Tuple[int, int]] = {}
        self.item_colors: List[Tuple[int, int, int]] = []

        for item_id, item_name in enumerate(self.item_names):
            meta = self.items_config[item_name]
            dept = int(meta["department"])
            self.item_departments.append(dept)
            # Use department location as this item's target location
            self.item_locations[item_id] = dept_locations[dept]
            # Convert hex color (e.g., "#ADD8E6") to RGB tuple
            color_hex = meta.get("color", "#FFFFFF")
            self.item_colors.append(self._hex_to_rgb(color_hex))

        self.num_items = len(self.item_names)
        # Fixed per-episode order in which items are presented so that
        # each item type appears at most once per episode.
        self.item_order: List[int] = []
        self.item_order_idx: int = 0
        # Track which item types have been correctly stocked (perfect placement)
        # during the current episode.
        self.items_stocked: List[bool] = [False] * self.num_items
        
        # Stocking cart starting location (middle bottom)
        self.cart_start = (grid_size - 1, grid_size // 2)

        # Customers (NPCs) randomly moving around the store.
        # These are additional dynamic obstacles the clerk should avoid.
        self.num_customers = 6 if self.enable_customers else 0
        self.customer_positions = []  # List[Tuple[int, int]]
        # Customer movement parameters: move only occasionally and with some inertia
        # so their paths are less jittery and more realistic.
        self.customer_move_prob = 0.4  # Chance a given customer tries to move on a step
        self.customer_prev_dirs: List[Tuple[int, int]] = []
        
        # Action space: 8 discrete actions (4 moves + pickup + dropoff + cart controls)
        self.action_space = spaces.Discrete(8)
        
        # Observation space:
        # We expose a structured Dict observation so agents can directly see
        # whether they are pushing the cart, in addition to their position and
        # which item encoding they currently have.
        #
        #   - agent_row, agent_col: agent position on the grid
        #   - item: item_encoding (item_id if has_item else num_items)
        #   - pushing_cart: 1 if pushing, 0 otherwise
        self.observation_space = spaces.Dict(
            {
                "agent_row": spaces.Discrete(grid_size),
                "agent_col": spaces.Discrete(grid_size),
                "item": spaces.Discrete(self.num_items + 1),
                "pushing_cart": spaces.Discrete(2),
            }
        )
        # Internal discrete state index size used by tabular / embedding-based
        # agents. This extends the original encoding by a factor of 2 to account
        # for the pushing_cart flag.
        self.state_index_size = grid_size * grid_size * (self.num_items + 1) * 2
        
        # Movement actions
        self.action_to_direction = {
            0: (1, 0),   # South (down)
            1: (-1, 0),  # North (up)
            2: (0, 1),   # East (right)
            3: (0, -1),  # West (left)
        }
        # Human-readable action names for debugging / chatbox display
        self.action_names: List[str] = [
            "Move South",
            "Move North",
            "Move East",
            "Move West",
            "Pickup box",
            "Dropoff item",
            "Start pushing cart",
            "Leave cart",
        ]
        
        # Initialize state variables
        self.agent_pos = None
        self.current_item = None
        self.has_item = False
        self.steps = 0
        self.episode_reward = 0.0
        self.max_steps = 400
        # Track items that have been dropped on the floor: (row, col, item_id)
        self.dropped_items = []
        # Stocking cart position and whether the agent is currently pushing it
        self.cart_pos: Tuple[int, int] = self.cart_start
        self.pushing_cart: bool = False
        
        # Rendering
        self.window = None
        self.clock = None
        # Choose cell size based on grid_size so the window fits on screen.
        # For grid_size=40 this gives 20px cells -> 800x800 window.
        max_window_pixels = 800
        base_cell = max_window_pixels // self.grid_size
        # Clamp to a reasonable range in case grid_size is very small/large.
        self.cell_size = max(10, min(60, base_cell))
        # Lazy-loaded images for each item type (used for carried and dropped items).
        # Mapping: item_id -> pygame.Surface or False (if load failed).
        self.item_images: Dict[int, object] = {}
        # Track last action taken for rendering a simple "chat box"
        self.last_action: Optional[int] = None
        
    def _encode_state_index(self) -> int:
        """
        Encode the current environment state into a single integer index.
        This is used internally by tabular / embedding-based agents.
        """
        row, col = self.agent_pos
        item_encoding = self.current_item if self.has_item else self.num_items
        base_index = (
            row * self.grid_size * (self.num_items + 1)
            + col * (self.num_items + 1)
            + item_encoding
        )
        pushing_bit = 1 if self.pushing_cart else 0
        return base_index * 2 + pushing_bit
    
    def _get_obs(self) -> Dict[str, int]:
        """
        Return the current observation as a structured Dict.
        
        Keys:
            - agent_row (int)
            - agent_col (int)
            - item (int): item_id if carrying an item, else num_items sentinel
            - pushing_cart (int): 1 if pushing the cart, 0 otherwise
        """
        row, col = self.agent_pos
        item_encoding = self.current_item if self.has_item else self.num_items
        return {
            "agent_row": int(row),
            "agent_col": int(col),
            "item": int(item_encoding),
            "pushing_cart": int(self.pushing_cart),
        }
    
    def _get_info(self) -> dict:
        """Return auxiliary information."""
        distance_to_target = self._manhattan_distance(
            self.agent_pos,
            self.item_locations[self.current_item],
        )

        # Customer proximity information
        nearby_customers = self._get_nearby_customers(radius=5)
        min_customer_distance = (
            min(
                self._manhattan_distance(self.agent_pos, c_pos)
                for c_pos in self.customer_positions
            )
            if self.customer_positions
            else None
        )
        return {
            "agent_pos": self.agent_pos,
            "current_item": self.current_item,
            "target_location": self.item_locations[self.current_item],
            "has_item": self.has_item,
            "distance_to_target": distance_to_target,
            "steps": self.steps,
            "customer_positions": list(self.customer_positions),
            "nearby_customers": nearby_customers,
            "min_customer_distance": min_customer_distance,
            "all_items_stocked": all(self.items_stocked),
            # Expose the compact integer state index so existing tabular /
            # discrete-state agents can continue to work even though the
            # observation_space is now a Dict.
            "state_index": self._encode_state_index(),
        }
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _hex_to_rgb(self, hex_str: str) -> Tuple[int, int, int]:
        """Convert a hex color string like '#RRGGBB' to an (R, G, B) tuple."""
        hex_str = hex_str.lstrip("#")
        if len(hex_str) != 6:
            # Fallback to white on malformed input
            return (255, 255, 255)
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b)

    def _load_json(self, path: str, root_key: Optional[str] = None):
        """
        Load a JSON file from `path`. If `root_key` is provided, return data[root_key].
        This keeps configuration for items and departments outside the Python code.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[root_key] if root_key is not None else data
        except Exception:
            # Fail gracefully with empty data; environment will still construct,
            # but item-related behavior may not be meaningful.
            return {} if root_key is None else {}

    def _get_nearby_customers(self, radius: int = 5):
        """
        Return a list of customer positions within `radius` (Manhattan distance)
        of the agent.
        """
        if not self.customer_positions:
            return []
        nearby = []
        for c_pos in self.customer_positions:
            if self._manhattan_distance(self.agent_pos, c_pos) <= radius:
                nearby.append(c_pos)
        return nearby

    def _spawn_customers(self):
        """
        Initialize customer NPC positions at random grid cells that are not
        the stocking cart. Multiple customers may share a cell; that's fine.
        """
        self.customer_positions = []
        for _ in range(self.num_customers):
            while True:
                row = int(self.np_random.integers(0, self.grid_size))
                col = int(self.np_random.integers(0, self.grid_size))
                if (row, col) != self.cart_start and (row, col) not in self.obstacles:
                    self.customer_positions.append((row, col))
                    break
        # Reset previous directions when customers are (re)spawned
        self.customer_prev_dirs = [(0, 0) for _ in self.customer_positions]

    def _move_customers(self):
        """
        Move each customer one step in a random cardinal direction or stay put,
        respecting grid boundaries.
        """
        if not self.customer_positions:
            return

        # Ensure prev_dirs is aligned with positions
        if len(self.customer_prev_dirs) != len(self.customer_positions):
            self.customer_prev_dirs = [(0, 0) for _ in self.customer_positions]

        new_positions: List[Tuple[int, int]] = []
        new_dirs: List[Tuple[int, int]] = []
        directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

        for idx, (row, col) in enumerate(self.customer_positions):
            # With some probability, customer does not move at all this step.
            if float(self.np_random.random()) > self.customer_move_prob:
                new_positions.append((row, col))
                new_dirs.append(self.customer_prev_dirs[idx])
                continue

            prev_dr, prev_dc = self.customer_prev_dirs[idx]

            # With high probability, keep moving in the same direction (inertia).
            if float(self.np_random.random()) < 0.7:
                dr, dc = prev_dr, prev_dc
            else:
                d_idx = int(self.np_random.integers(0, len(directions)))
                dr, dc = directions[d_idx]

            nr, nc = row + dr, col + dc
            if (
                0 <= nr < self.grid_size
                and 0 <= nc < self.grid_size
                and (nr, nc) not in self.obstacles
            ):
                new_positions.append((nr, nc))
                new_dirs.append((dr, dc))
            else:
                # If movement would leave the grid, stay put and reset direction.
                new_positions.append((row, col))
                new_dirs.append((0, 0))

        self.customer_positions = new_positions
        self.customer_prev_dirs = new_dirs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset stocking cart and agent position
        self.cart_pos = self.cart_start
        self.agent_pos = self.cart_pos
        
        # Sample a random order of items for this episode so each item
        # is seen at most once.
        self.item_order = list(range(self.num_items))
        self.np_random.shuffle(self.item_order)
        self.item_order_idx = 0

        # Assign the first item to stock
        self.current_item = self.item_order[self.item_order_idx]
        
        # Start with item in hand
        self.has_item = True
        self.steps = 0
        self.episode_reward = 0.0
        self.dropped_items = []
        self.pushing_cart = False
        self.last_action = None
        # Reset per-episode stocked tracking
        self.items_stocked = [False] * self.num_items

        # Spawn customers at random locations
        self._spawn_customers()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action: int):
        """Execute one step in the environment."""
        self.steps += 1
        # Small negative reward for each step to encourage efficiency, but
        # not so large that exploration becomes overwhelmingly bad.
        reward = -3
        terminated = False
        self.last_action = action
        # Distance to cart before taking the action (for shaping when empty-handed)
        prev_dist_to_cart = self._manhattan_distance(self.agent_pos, self.cart_pos)
        # Distance from cart to current target item location (for shaping when pushing cart)
        prev_cart_to_target = self._manhattan_distance(
            self.cart_pos, self.item_locations[self.current_item]
        )
        
        # Actions:
        #   0: Move South (down)
        #   1: Move North (up)
        #   2: Move East (right)
        #   3: Move West (left)
        #   4: Pickup box (at stocking cart)
        #   5: Dropoff item
        #   6: Start pushing cart (when adjacent to cart)
        #   7: Leave cart (stop pushing)
        
        if action < 4:
            # Actions 0–3: movement
            direction = self.action_to_direction[action]
            new_row = self.agent_pos[0] + direction[0]
            new_col = self.agent_pos[1] + direction[1]
            
            # Check boundaries and obstacles
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                if (new_row, new_col) in self.obstacles:
                    # Ran into a static obstacle (e.g., shelving) – stronger penalty.
                    reward -= 10
                else:
                    # If pushing the cart, treat item goal cells as blocked so the
                    # cart can never sit on a shelf location.
                    if self.pushing_cart and (new_row, new_col) in self.item_locations.values():
                        # Block the move and apply a small penalty for trying to
                        # push the cart onto a shelf.
                        reward -= 10
                    else:
                        self.agent_pos = (new_row, new_col)
                        # If pushing the cart, it follows the agent
                        if self.pushing_cart:
                            self.cart_pos = self.agent_pos
            else:
                # Invalid attempt (off-grid); small penalty
                reward -= 20
        
        elif action == 4:
            # Action 4: Pickup from stocking cart
            if self.agent_pos == self.cart_pos:
                if self.has_item:
                    # Trying to pick up while already carrying an item:
                    # strongly discouraged.
                    reward -= 40
                else:
                    # Move to the next item in the fixed episode order, if any.
                    if self.item_order_idx < self.num_items - 1:
                        self.item_order_idx += 1
                        self.current_item = self.item_order[self.item_order_idx]
                        self.has_item = True
                        reward += 0  # Neutral for picking up
                    else:
                        # No more items to stock this episode
                        reward -= 50
            else:
                # Attempting to pick up when not at the cart is also wasteful.
                reward -= 20
        
        elif action == 5:
            # Action 5: Dropoff item
            if self.has_item:
                # Cannot drop off while pushing the cart
                if self.pushing_cart:
                    # Small penalty for invalid drop attempt
                    reward -= 1
                else:
                    target_location = self.item_locations[self.current_item]
                    distance = self._manhattan_distance(self.agent_pos, target_location)

                    # Record where this item was actually left on the floor.
                    drop_row, drop_col = self.agent_pos
                    self.dropped_items.append((drop_row, drop_col, self.current_item))

                    # Base dropoff reward based purely on agent-to-target distance.
                    if distance == 0:
                        # Perfect placement!
                        reward = 1000
                        # Mark this item type as correctly stocked for this episode.
                        if 0 <= self.current_item < self.num_items:
                            self.items_stocked[self.current_item] = True
                    elif distance <= 4:
                        # Within 4 blocks
                        reward = 300
                    elif distance <= 10:
                        # Within 10 blocks
                        reward = 50
                    else:
                        # Too far away
                        reward = -20

                    # Additional shaping based on how close the cart itself is to
                    # the item's true shelf location. This makes it more desirable
                    # to move the cart toward the department before dropping.
                    cart_dist = self._manhattan_distance(self.cart_pos, target_location)
                    cart_penalty_per_step = 1.0
                    reward -= cart_penalty_per_step * cart_dist

                    # If every item type has been perfectly stocked at least once
                    # during this episode, give an additional completion bonus
                    # and end the episode.
                    if all(self.items_stocked):
                        completion_bonus = 2000.0
                        reward += completion_bonus
                        terminated = True

                    # After any drop, the clerk no longer has the item and must
                    # return to the cart (action 4 at cart) for a new one.
                    self.has_item = False
            else:
                # Tried to drop off but don't have an item
                reward = -50
        
        elif action == 6:
            # Action 6: Start pushing the cart (must be at or adjacent to it and not already pushing)
            if not self.pushing_cart:
                # Treat standing on the cart or in any 4-neighbor cell as valid.
                if self._manhattan_distance(self.agent_pos, self.cart_pos) <= 1:
                    self.pushing_cart = True
                    # Snap cart onto agent's cell so they move together
                    self.cart_pos = self.agent_pos
                else:
                    # Invalid attempt; softer penalty so exploration of action 6
                    # is not overly discouraged.
                    reward -= 20
            else:
                # Already pushing; no-op with small penalty
                reward -= 10

        elif action == 7:
            # Action 7: Leave the cart; it stays where it is
            if self.pushing_cart:
                self.pushing_cart = False
                reward -= 10
            else:
                # Already not pushing; no-op with small penalty
                reward -= 50

        # Distance-based shaping and penalty when moving without an item.
        # Being empty-handed is still treated as undesirable, but the per-step
        # penalty is softened so that returning to the cart is not suicidal.
        if action < 4 and not self.has_item:
            new_dist_to_cart = self._manhattan_distance(self.agent_pos, self.cart_pos)
            alpha = 2.0  # Shaping: prefer moves that reduce distance to cart
            reward += alpha * (prev_dist_to_cart - new_dist_to_cart)
            # Moderate per-step penalty for being empty-handed
            reward -= 10

        # When pushing the cart, we do not provide any extra positive shaping
        # beyond what the agent would receive by walking without the cart.
        # This keeps "riding in the cart" from being inherently more rewarding
        # than simply moving around on foot.
        if action < 4 and self.pushing_cart:
            new_cart_to_target = self._manhattan_distance(
                self.cart_pos, self.item_locations[self.current_item]
            )
            alpha_cart = 0.0
            reward += alpha_cart * (prev_cart_to_target - new_cart_to_target)
        
        # Move customers randomly after the agent acts
        self._move_customers()

        # Customer proximity penalty: large negative reward when adjacent.
        # If this happens, "pull" the agent back to the cart and respawn customers
        # to simulate being interrupted to help a customer.
        if self.customer_positions:
            min_cust_dist = min(
                self._manhattan_distance(self.agent_pos, c_pos)
                for c_pos in self.customer_positions
            )
        else:
            min_cust_dist = None

        # Customer proximity penalties only apply when the agent is NOT
        # actively pushing the cart. When pushing, customers "see the clerk
        # is busy" and do not interrupt or penalize them.
        if min_cust_dist is not None and not self.pushing_cart: #TODO: Remove this condition for test
            if min_cust_dist <= 1:
                # Strong (but softened) penalty for being on or right next to a customer
                reward -= 70
                # Teleport agent back to the cart (wherever it currently is)
                self.agent_pos = self.cart_pos
                # No longer pushing the cart after being interrupted
                self.pushing_cart = False
                # Respawn customers at fresh random locations
                self._spawn_customers()
            elif min_cust_dist <= 4:
                # Mild penalty when within 4 blocks of any customer.
                # This nudges the agent away from busy areas without making
                # exploration prohibitively expensive.
                reward -= 30

        # Check if max steps reached
        if self.steps >= self.max_steps:
            terminated = True
        
        # Accumulate episode reward for rendering/debugging.
        self.episode_reward += reward
        
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

        # Light grey highlight for cells within 4-block (Manhattan) radius of any dropoff location.
        highlight_color = (235, 235, 235)
        radius = 4
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell_pos = (row, col)
                # Check if cell is close to any item location
                close_to_dropoff = any(
                    self._manhattan_distance(cell_pos, drop_pos) <= radius
                    for drop_pos in self.item_locations.values()
                )
                if close_to_dropoff:
                    pygame.draw.rect(
                        canvas,
                        highlight_color,
                        pygame.Rect(
                            col * self.cell_size,
                            row * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
        
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
        
        # Draw item locations using colors loaded from items.json
        item_colors = self.item_colors
        
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
            # Draw department number for this location so it matches items.json
            if pygame.font.get_init():
                font = pygame.font.Font(None, 24)
                dept_id = self.item_departments[item_id]
                text = font.render(str(dept_id), True, (0, 0, 0))
                text_rect = text.get_rect(
                    center=(
                    col * self.cell_size + self.cell_size // 2,
                        row * self.cell_size + self.cell_size // 2,
                    )
                )
                canvas.blit(text, text_rect)

        # Draw static obstacles as solid black cells
        for (row, col) in self.obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )
        
        # Ensure item images are loaded once (used for carried and dropped items)
        if not self.item_images:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            inventory_dir = os.path.join(script_dir, "inventory")
            for item_id, item_name in enumerate(self.item_names):
                meta = self.items_config.get(item_name, {})
                rel_path = meta.get("image")
                if not rel_path:
                    self.item_images[item_id] = False
                    continue
                try:
                    img_path = os.path.normpath(os.path.join(inventory_dir, rel_path))
                    if os.path.exists(img_path):
                        raw_img = pygame.image.load(img_path).convert_alpha()
                        scale = int(self.cell_size * 0.8)
                        self.item_images[item_id] = pygame.transform.smoothscale(
                            raw_img, (scale, scale)
                        )
                    else:
                        self.item_images[item_id] = False
                except Exception:
                    self.item_images[item_id] = False

        # Draw items that have been dropped on the floor.
        # Use the same sized colored background as their corresponding shelf
        # locations so the visual language is consistent.
        for drop_row, drop_col, drop_item in self.dropped_items:
            color = item_colors[drop_item]
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    drop_col * self.cell_size + 5,
                    drop_row * self.cell_size + 5,
                    self.cell_size - 10,
                    self.cell_size - 10,
                ),
            )
            # Overlay the specific item icon at the drop location, if available
            item_img = self.item_images.get(drop_item)
            if item_img not in (None, False):
                img_rect = item_img.get_rect(
                    center=(
                        drop_col * self.cell_size + self.cell_size // 2,
                        drop_row * self.cell_size + self.cell_size // 2,
                    )
                )
                canvas.blit(item_img, img_rect)

        # Draw stocking cart
        office_row, office_col = self.cart_pos
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
        
        # Draw customers (NPCs) as green circles
        for cust_row, cust_col in self.customer_positions:
            pygame.draw.circle(
                canvas,
                (0, 200, 0),
                (
                    cust_col * self.cell_size + self.cell_size // 2,
                    cust_row * self.cell_size + self.cell_size // 2,
                ),
                self.cell_size // 4,
            )

        # Draw agent
        agent_row, agent_col = self.agent_pos
        # When carrying an item, tint the agent with that item's color; otherwise blue.
        if self.has_item and 0 <= self.current_item < self.num_items:
            agent_color = self.item_colors[self.current_item]
        else:
            agent_color = (0, 0, 255)

        pygame.draw.circle(
            canvas,
            agent_color,
            (
                agent_col * self.cell_size + self.cell_size // 2,
                agent_row * self.cell_size + self.cell_size // 2
            ),
            self.cell_size // 3
        )

        # Draw item image if carrying an item
        if self.has_item:
            item_img = self.item_images.get(self.current_item)
            if item_img not in (None, False):
                img_rect = item_img.get_rect(
                    center=(
                        agent_col * self.cell_size + self.cell_size // 2,
                        agent_row * self.cell_size + self.cell_size // 2,
                    )
                )
                canvas.blit(item_img, img_rect)

        # Draw a simple "chat box" in the bottom-right showing the last action,
        # the Manhattan distance to the current target (if any), and the
        # cumulative episode reward.
        box_width = int(self.grid_size * self.cell_size * 0.2)
        box_height = 100
        box_x = self.grid_size * self.cell_size - box_width - 10
        box_y = self.grid_size * self.cell_size - box_height - 10

        chat_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(canvas, (245, 245, 245), chat_rect)
        pygame.draw.rect(canvas, (180, 180, 180), chat_rect, width=1)

        if pygame.font.get_init():
            font = pygame.font.Font(None, 22)
            if self.last_action is not None and 0 <= self.last_action < len(self.action_names):
                action_num_text = f"Action #: {self.last_action}"
                action_name_text = f"Action: {self.action_names[self.last_action]}"
            else:
                action_num_text = "Action #: N/A"
                action_name_text = "Action: N/A"
            if self.has_item and 0 <= self.current_item < self.num_items:
                target_pos = self.item_locations[self.current_item]
                dist_to_target = self._manhattan_distance(self.agent_pos, target_pos)
                dist_text = f"Dist to target: {dist_to_target}"
            else:
                dist_text = "Dist to target: N/A"

            reward_text = f"Episode reward: {self.episode_reward:.1f}"

            text1 = font.render(action_num_text, True, (0, 0, 0))
            text2 = font.render(action_name_text, True, (0, 0, 0))
            text3 = font.render(dist_text, True, (0, 0, 0))
            text4 = font.render(reward_text, True, (0, 0, 0))
            canvas.blit(text1, (box_x + 8, box_y + 4))
            canvas.blit(text2, (box_x + 8, box_y + 24))
            canvas.blit(text3, (box_x + 8, box_y + 44))
            canvas.blit(text4, (box_x + 8, box_y + 64))
        
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
    max_episode_steps=400,
)


