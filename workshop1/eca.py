"""
Elementary cellular automata simulator in numpy.
"""


import itertools
import pathlib
import time
from typing import Literal

import numpy as np
from PIL import Image
import tqdm

import jax
import jax.numpy as jnp
import einops

def main(
    width: int = 32,
    height: int = 32,
    init: Literal["random", "middle"] = "middle",
    seed: int = 42,
    animate: bool = True,
    fps: None | float = None,
    save_image: None | pathlib.Path = None,
    upscale: int = 1,
):
    
    print("initialising state...")
    match init:
        case "middle":
            state = jnp.zeros(width, dtype=jnp.uint8)
            state = state.at[width//2].set(1)
        case "random":
            key = jax.random.key(seed)
            key, key_to_be_used = jax.random.split(key)
            state = jax.random.randint(
                key=key_to_be_used,
                minval=0,
                maxval=2, # not included
                shape=(width,),
                dtype=jnp.uint8,
            )
    print("initial state:", state)

    print("simulating automata...")
    start_time = time.perf_counter()
    histories = jax.jit(
        jax.vmap(
            simulate,
            in_axes=(0,None,None),
            out_axes=0, # the default
        ),
        static_argnames=('height',),
    )(
        jnp.arange(256),
        state,
        height,
    )
    end_time = time.perf_counter()
    print("simulation complete!")
    print("result shape", histories.shape)
    print(f"time taken {end_time - start_time:.4f} seconds")

    if animate:
        print("rendering...")
        for i, history in enumerate(histories):
            print("rule", i)
            for row in history:
                print(''.join(["█░"[s]*2 for s in row]))
                if fps is not None: time.sleep(1/fps)
            if fps is not None: time.sleep(1/fps)

    if save_image is not None:
        print("rendering to", save_image, "...")
        histories_arranged = einops.rearrange(
            histories,
            '(r1 r2) h w -> (r1 h) (r2 w)',
            r1=16,
            r2=16,
        )
        histories_greyscale = 255 * (1-histories_arranged)
        histories_upscaled = (histories_greyscale
            .repeat(upscale, axis=0)
            .repeat(upscale, axis=1)
        )
        Image.fromarray(np.asarray(histories_upscaled)).save(save_image)

        
def simulate(
    rule: int,
    init_state: jax.Array,    # uint8[width]
    height: int,
) -> jax.Array:               # uint8[height, width]
    # parse rule
    rule_uint8 = jnp.uint8(rule)
    rule_bits = jnp.unpackbits(rule_uint8, bitorder='little')
    rule_table = rule_bits.reshape(2,2,2)

    # parse initial state
    init_state = jnp.pad(init_state, 1, mode='wrap')

    def step(prev_state, _input):
        next_state = rule_table[
            prev_state[0:-2],
            prev_state[1:-1],
            prev_state[2:],
        ]
        next_state = jnp.pad(next_state, 1, mode='wrap')
        return next_state, next_state
    
    final_state, history_tail = jax.lax.scan(
        step,
        init_state,
        jnp.zeros(height-1)
    )

    history = jnp.concatenate(
        [init_state[jnp.newaxis,:], history_tail],
        axis=0
    )

    # return a view of the array without the width padding
    return history[:, 1:-1]


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

