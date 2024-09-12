"""
Gridworld navigation environment, accelerated with JAX.
"""

import functools
import dataclasses

import jax
import jax.numpy as jnp
import strux

from jaxtyping import Array, Int, Float, Bool, PRNGKeyArray

import readchar
import mattplotlib as mp


# # # 
# Environment state


@strux.struct
class EnvState:
    wall_map: Bool[Array, "size size"]
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    got_goal: bool

@functools.partial(strux.struct, static_fieldnames=["size"])
class MazeEnvironment:
    size: int
    wall_prob: float
    
    @jax.jit
    def reset(self, rng: PRNGKeyArray) -> EnvState:
        rng_walls, rng = jax.random.split(rng)
        wall_map = jnp.ones((self.size, self.size), dtype=bool)
        wall_map = wall_map.at[1:-1, 1:-1].set(
            jax.random.bernoulli(
                key=rng_walls,
                shape=(self.size-2, self.size-2),
                p=self.wall_prob,
            )
        )

        rng_spawn, rng = jax.random.split(rng)
        hero_index, goal_index = jax.random.choice(
            key=rng_spawn,
            a=self.size*self.size,
            shape=(2,),
            replace=False,
            p= ~wall_map.flatten(),
        )
        hero_pos = jnp.array(
            [hero_index // self.size,
             hero_index % self.size])
        goal_pos = jnp.array(
            [goal_index // self.size,
             goal_index % self.size])
        
        return EnvState(
            wall_map=wall_map,
            hero_pos=hero_pos,
            goal_pos=goal_pos,
            got_goal=False,
        )

    def step(self, state: EnvState, action: Int[Array, ""]) -> tuple[EnvState, float, bool]:
        # step the hero
        step = jnp.array((
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
        ))[action]
        next_pos = state.hero_pos + step
        hit_wall = state.wall_map[next_pos[0], next_pos[1]]
        next_pos = jnp.where(hit_wall, state.hero_pos, next_pos)
        state = state.replace(hero_pos=next_pos)

        # check if we got the goal
        hit_goal = (state.hero_pos == state.goal_pos).all()
        first_hit = hit_goal & ~state.got_goal
        state = state.replace(
            got_goal=state.got_goal | hit_goal,
        )

        reward = first_hit.astype(float)
        done = state.got_goal
        return (state, reward, done)

    @jax.jit
    def render(self, state: EnvState) -> Float[Array, "size size 3"]:
        wall = jnp.array((0.2, 0.2, 0.2))
        path = jnp.array((0.0, 0.0, 0.0))
        hero = jnp.array((0.0, 0.0, 0.0))
        goal = jnp.array((1.0, 1.0, 0.0))

        img = jnp.where(
            state.wall_map.reshape(self.size, self.size, 1),
            wall,
            path,
        )
        img = img.at[
            state.hero_pos[0],
            state.hero_pos[1],
        ].set(hero)
        img = img.at[
            state.goal_pos[0],
            state.goal_pos[1],
        ].set(goal)

        return img


# # # 
# Controller


def main(
    size: int = 16,
    wall_prob: float = 0.25,
    num_environments: int = 8,
    seed: int = 42,
):
    rng = jax.random.key(seed=seed)

    # initialise environment class
    env = MazeEnvironment(size=size, wall_prob=wall_prob)

    # initialise first env states
    rng_reset, rng = jax.random.split(rng)
    states = jax.vmap(
        env.reset,
    )(
        jax.random.split(rng_reset, num_environments)
    )
    print(render(
        env,
        states=states,
        actions=jnp.zeros(num_environments, dtype=int),
        rewards=jnp.zeros(num_environments),
        dones=jnp.zeros(num_environments, dtype=bool),
    ))

    
    # play loop
    while True:
        # choose an action
        key = readchar.readkey()
        if key == "q":
            print("bye!")
            return
        if key not in "wsda":
            continue
        action = "wasd".index(key)
        actions = jnp.full(num_environments, action)
        
        # apply the action
        states, rewards, dones = jax.vmap(
            env.step,
        )(states, actions)

        # reset if done
        rng_reset, rng = jax.random.split(rng)
        reset_states = jax.vmap(
            env.reset,
        )(
            jax.random.split(rng_reset, num_environments)
        )

        states = jnp.where(
            dones.reshape(num_environments, 1, 1, 1),
            reset_states,
            states,
        )

        # render
        plot = render(
            env=env,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        print(f"\x1b[{plot.height}A{plot}")


# # # 
# Visualisation


def render(
    env: MazeEnvironment,
    states: EnvState, # EnvState[num_environments]
    actions: Int[Array, "num_environments"],
    rewards: Float[Array, "num_environments"],
    dones: Bool[Array, "num_environments"],
) -> mp.plot:
    imgs = jax.vmap(env.render)(states)
    plots = []
    for img, a, r, d in zip(imgs, actions, rewards, dones):
        plots.append(mp.border(
            mp.image(img)
            ^ mp.text(f"a={a}")
            ^ mp.text(f"r={r.astype(int)}")
            ^ mp.text(f"d={d.astype(int)}")
        ))
    return mp.wrap(*plots)



# # #
# Entry point


if __name__ == "__main__":
    import tyro
    tyro.cli(main)


