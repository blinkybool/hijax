"""
Gridworld navigation environment, accelerated with JAX.
"""

import functools

import jax
import jax.numpy as jnp
import strux

from jaxtyping import Array, Int, Float, Bool, PRNGKeyArray

import readchar
import mattplotlib as mp


@strux.struct
class EnvState:
    wall_map: Bool[Array, "size size"]
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    got_goal: bool
    num_steps: int


@functools.partial(strux.struct, static_fieldnames=["size"])
class MazeEnvironment:
    size: int
    wall_prob: float
    max_num_steps: int


    @jax.jit
    def reset(self, rng: PRNGKeyArray) -> EnvState:
        # random wall layout
        rng_walls, rng = jax.random.split(rng)
        wall_map = jnp.ones((self.size, self.size), dtype=bool)
        wall_map = wall_map.at[1:-1,1:-1].set(jax.random.bernoulli(
            rng_walls,
            shape=(self.size-2, self.size-2),
            p=self.wall_prob,
        ))

        # spawn them
        rng_spawn, rng = jax.random.split(rng)
        hero_posid, goal_posid = jax.random.choice(
            key=rng_spawn,
            a=self.size*self.size,
            shape=(2,),
            p=~wall_map.flatten(),
            replace=False,
        )
        hero_pos = jnp.array((
            hero_posid // self.size,
            hero_posid % self.size,
        ))
        goal_pos = jnp.array((
            goal_posid // self.size,
            goal_posid % self.size,
        ))

        return EnvState(
            wall_map=wall_map,
            hero_pos=hero_pos,
            goal_pos=goal_pos,
            got_goal=False,
            num_steps=0,
        )

    
    @jax.jit
    def step(self, state: EnvState, action: int) -> EnvState:
        # update hero pos
        step = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))[action]
        next_pos = state.hero_pos + step
        hit_wall = state.wall_map[next_pos[0], next_pos[1]]
        next_pos = jnp.where(
            hit_wall,
            state.hero_pos,
            next_pos,
        )
        # check if hero hit goal
        hit_goal = (next_pos == state.goal_pos).all()
        # update state
        return state.replace(
            hero_pos=next_pos,
            got_goal=state.got_goal | hit_goal,
            num_steps=state.num_steps+1,
        )


    @jax.jit
    def reward(self, prev_state: EnvState, next_state: EnvState) -> float:
        just_hit = next_state.got_goal & ~prev_state.got_goal
        return just_hit.astype(float)


    @jax.jit
    def done(self, state: EnvState) -> bool:
        complete = state.got_goal
        terminal = state.num_steps >= self.max_num_steps
        return complete | terminal
        

    @jax.jit
    def render(self, state: EnvState) -> Float[Array, "size size 3"]:
        # color palette
        path = jnp.array((0.0, 0.0, 0.0))
        wall = jnp.array((0.2, 0.2, 0.2))
        hero = jnp.array((0.0, 0.8, 0.0))
        goal = jnp.array((1.0, 1.0, 0.0))
        
        # construct the image
        # colour the walls and path
        img = jnp.where(
            state.wall_map[:, :, jnp.newaxis],
            wall,
            path,
        )
        # colour the hero
        img = img.at[
            state.hero_pos[0],
            state.hero_pos[1],
        ].set(hero)
        # colour the goal (if it's still there)
        img = img.at[
            state.goal_pos[0],
            state.goal_pos[1],
        ].set(jnp.where(
            state.got_goal,
            path,
            goal,
        ))
        return img


    @functools.partial(jax.jit, static_argnames=["num_states"])
    def vmap_reset(
        self,
        rng: PRNGKeyArray,
        num_states: int,
    ) -> EnvState: # EnvState[num_states]
        return jax.vmap(self.reset)(jax.random.split(rng, num_states))


    @jax.jit
    def vmap_step(
        self,
        states: EnvState, # EnvState[num_states]
        actions: Int[Array, "num_states"],
    ) -> EnvState:   # EnvState[num_states]
        return jax.vmap(self.step)(states, actions)


    @jax.jit
    def vmap_done(
        self,
        states: EnvState, # EnvState[num_states]
    ) -> Bool[Array, "num_states"]:
        return jax.vmap(self.done)(states)

    
    @jax.jit
    def vmap_reward(
        self,
        prev_states: EnvState,   # EnvState[num_states]
        next_states: EnvState,   # EnvState[num_states]
    ) -> float:
        return jax.vmap(reward)(prev_states, next_states)


    @jax.jit
    def vmap_render(
        self,
        states: EnvState, # EnvState[num_states]
    ) -> Float[Array, "num_states size size 3"]:
        return jax.vmap(self.render)(states)


