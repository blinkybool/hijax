"""
Game of life with JAX.
"""


import itertools
import pathlib
import time
from typing import Literal

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp

def main(
    width: int = 32,
    height: int = 32,
    num_steps: int = 127,
    init: Literal["random", "glider"] = "glider",
    seed: int = 42,
    animate: bool = False,
    fps: None | float = None,

    save_image: None | pathlib.Path = None,
    upscale: int = 1,
):
    
    print("initialising state...")
    match init:
        case "glider":
            state = jnp.zeros((height,width), dtype=jnp.uint8)
            state = state.at[1,2].set(1)
            state = state.at[2,3].set(1)
            state = state.at[3,1].set(1)
            state = state.at[3,2].set(1)
            state = state.at[3,3].set(1)
        case "random":
            key = jax.random.key(seed)
            key, key_to_be_used = jax.random.split(key)
            state = jax.random.randint(
                key=key_to_be_used,
                minval=0,
                maxval=2, # not included
                shape=(height,width),
                dtype=jnp.uint8,
            )
            
    print("generating rule bits")
    rule_bits = gol_rule_bits()

    if animate:
        print(vis(state, overwrite=False))

    print("simulating automata...")
    start_time = time.perf_counter()
    history = jax.jit(
        simulate,
        static_argnames=('num_steps',),
    )(
        rule_bits=rule_bits,
        init_state=state,
        num_steps=num_steps,
    )
    end_time = time.perf_counter()
    print("simulation complete!")
    print("result shape", history.shape)
    print(f"time taken {end_time - start_time:.4f} seconds")

    if animate:
        print("rendering...")
        for state in history:
            print(vis(state))
            if fps is not None: time.sleep(1/fps)

    if save_image is not None:
        # raise NotImplementedError("save_image not implemented")
        print("rendering to", save_image, "...")

        images_for_gif = []
        for state in history:
            state_greyscale = 255 * (1-state)
            state_upscaled = (state_greyscale
                .repeat(upscale, axis=0)
                .repeat(upscale, axis=1)
            )
            image = Image.fromarray(np.array(state_upscaled))
            images_for_gif.append(image)

        images_for_gif[0].save(
            save_image,
            save_all=True,
            append_images=images_for_gif[1:],
            duration=len(images_for_gif)/fps,
            loop = 0,
        )

def vis(state, overwrite=True):
    figure_str = (
        '-' * (len(state[0])*2) + '\n'
        + '\n'.join(''.join(["█░"[s]*2 for s in row]) for row in state)

    )
    reset = f"\x1b[{len(figure_str.splitlines())}A" if overwrite else ""
    return reset + figure_str

def gol_rule_bits():
    rule_bits = np.zeros(512, dtype=np.uint8)

    for i, neighbourhood in enumerate(itertools.product((0,1), repeat=9)):
        neighbourhood = jnp.array(neighbourhood)
        alive = neighbourhood[4] == 1
        live_neighbours = jnp.sum(neighbourhood) - alive
        if alive:
            if 2 <= live_neighbours and live_neighbours <= 3:
                rule_bits[i] = 1
        elif live_neighbours == 3:
            rule_bits[i] = 1

    return rule_bits

        
def simulate(
    rule_bits: jax.Array,       
    init_state: jax.Array,    # uint8[height, width]
    num_steps: int,
) -> jax.Array:               # uint8[num_steps, height, width]
    # parse rule
    rule_table = rule_bits.reshape(2,2,2,2,2,2,2,2,2)

    # parse initial state
    init_state = jnp.pad(init_state, 1, mode='wrap')

    def step(prev_state, _input):
        next_state = rule_table[
            # yeehaw lets hope this works
            prev_state[0:-2, 0:-2],
            prev_state[0:-2, 1:-1],
            prev_state[0:-2, 2:  ],
            prev_state[1:-1, 0:-2],
            prev_state[1:-1, 1:-1],
            prev_state[1:-1, 2:  ],
            prev_state[2:  , 0:-2],
            prev_state[2:  , 1:-1],
            prev_state[2:  , 2:  ],
        ]
        next_state = jnp.pad(next_state, 1, mode='wrap')
        return next_state, next_state
    
    final_state, history_tail = jax.lax.scan(
        step,
        init_state,
        jnp.zeros(num_steps)
    )

    history = jnp.concatenate(
        [init_state[jnp.newaxis,:], history_tail],
        axis=0
    )

    # return a view of the array without the width padding
    return history[:, 1:-1, 1:-1]


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

