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


# # # 
# Environment state


@strux.struct
class EnvState:
    "TODO"


@functools.partial(strux.struct, static_fieldnames=["size"])
class MazeEnvironment:
    "TODO"


# # # 
# Controller


def main(
    size: int = 17,
    wall_prob: float = 0.25,
    num_environments: int = 8,
    seed: int = 42,
):
    rng = jax.random.key(seed=seed)

    # initialise environment class
    # TODO

    # initialise first env states
    # TODO
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
        # TODO

        # reset if done
        # TODO

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


