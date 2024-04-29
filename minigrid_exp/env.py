from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import random

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=11,
        agent_start_pos=(1, 1),
        agent_start_dir=3,
        eval_env=False,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        # if max_steps is None:
        #     max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.eval_env = eval_env
        self.goal_pos = None
        self.num_stitch = 0

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height, goal_pos):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for i in range(1, 10):
            if i not in {2, 9}:
                self.grid.set(i, 5, Wall())
            if i not in {4, 8}:
                self.grid.set(5, i, Wall())

        # Place the agent
        self.place_agent_and_goal(goal_pos)

        self.mission = "grand mission"

    def place_agent_and_goal(self, goal_pos):
        left_up = [(x, y) for x in range(1, 5) for y in range(1, 5)]
        left_bottom = [(x, y) for x in range(1, 5) for y in range(6, 10)]
        right_up = [(x, y) for x in range(6, 10) for y in range(1, 5)]
        right_bottom = [(x, y) for x in range(6, 10) for y in range(6, 10)]

        self.all_empty_cells = left_up + left_bottom + right_up + right_bottom + [(2, 5), (5, 4), (5, 8), (9, 5)]
        
        empty_sections = [left_up, left_bottom, right_up, right_bottom]
        bottleneck_states = [[(2, 5), (5, 4)], [(2, 5), (5, 8)], [(5, 4), (9, 5)], [(9, 5), (5, 8)]]

        if self.eval_env:
            start_section, goal_section = random.sample(empty_sections, k=2)
            self.agent_pos = random.choice(start_section)
            self.goal_pos = random.choice(goal_section)
        else:
            start_section = random.choice(empty_sections)
            agent_goal_pos_list = random.sample(start_section, k=2)
            bottleneck_state = random.choice(bottleneck_states[empty_sections.index(start_section)])
            p = np.random.rand()
            if p < 0.15:
                self.agent_pos = bottleneck_state
                self.goal_pos = agent_goal_pos_list[1]
            elif 0.15 <= p < 0.3:
                self.agent_pos = agent_goal_pos_list[0]
                self.goal_pos = bottleneck_state
            else:
                self.agent_pos = agent_goal_pos_list[0]
                self.goal_pos = agent_goal_pos_list[1]
        
        if goal_pos is None:
            self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())
        else:
            self.goal_pos = goal_pos
            self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())

        self.agent_dir = self._rand_int(0, 4)

    def reset(self, *, seed: int | None = None, options: None = None, goal_pos=None):
        seed_seq = np.random.SeedSequence(seed)
        np_seed = seed_seq.entropy
        rng = np.random.Generator(np.random.PCG64(seed_seq))
        self._np_random, seed = rng, np_seed

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height, goal_pos)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}