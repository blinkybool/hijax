import jax
import jax.numpy as jnp
import itertools

@jax.jit
def step(input: jax.Array) -> jnp.uint8:
    # . . .
    # . c .
    # . . .

    # alive ∈ [0,1]
    alive = input[4]
    # neighbours ∈ [0,8]
    neighbours = jnp.sum(input) - alive

    return jnp.where(
        neighbours == 3,
        1,
        jnp.where(alive * neighbours == 2, 1, 0)
    )

def all_inputs() -> jax.Array:
    return [jnp.array(xs) for xs in itertools.product((0,1), repeat=9)]