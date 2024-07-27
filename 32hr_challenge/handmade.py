import jax
import jax.numpy as jnp
from jax.nn import relu
import time
import conway

def step_relu(input):
    '''
    Implements the game of life step rule, using linear combinations of the
    input, plus uses of relu. Written to be readable.
    '''
    a,b,c,d,e,f,g,h,i = input

    neighbours = a + b + c + d + f + g + h + i
    alive = e

    def equals(target, value):
        '''
        Returns 1 iff target==value and at most 0 otherwise,
        assuming target and value are integers.
        '''
        return 1 - relu(target - value) - relu(value - target)

    has_three = equals(3, neighbours)
    has_two = equals(2, neighbours)
    supported = relu(alive - 1 + has_two)

    return has_three + supported

def step_relu2(xs):
    '''
    Implements the game of life step rule.
    Equivalent to step_relu, just closer to the MLP structure
    '''
    a,b,c,d,e,f,g,h,i = xs

    # a nice variable to use for first layer
    neighbours = a + b + c + d + f + g + h + i

    # Hidden layer 1
    p1 = relu(neighbours - 3)
    p2 = relu(3 - neighbours)
    q1 = relu(neighbours - 2)
    q2 = relu(2 - neighbours)
    alive = relu(e)

    # Hidden layer 2
    has_three = relu(1 - p1 - p2)
    has_two_and_alive = relu(alive - q1 - q2)

    # Output
    out = relu(has_three + has_two_and_alive)

    return out

def forward_mlp(params, input: jax.Array):
    '''
    MLP architecture with arbitrary parametrised layers
    '''
    layer = input
    for w,b in params[:-1]:
        layer = relu(jnp.dot(w,layer) + b)
    w, b = params[-1]
    layer = jnp.dot(w,layer) + b
    
    return layer

CONWAY_PARAMS = [
    (
        jnp.asarray([
            [ 1, 1, 1,  1, 0, 1,  1, 1, 1], # neighbours
            [-1,-1,-1, -1, 0,-1, -1,-1,-1], # -neighbours
            [ 1, 1, 1,  1, 0, 1,  1, 1, 1], # neighbours
            [-1,-1,-1, -1, 0,-1, -1,-1,-1], # -neighbours
            [ 0, 0, 0,  0, 1, 0,  0, 0, 0], # alive
        ]),
        jnp.array([-3,3,-2,2,0])
    ),
    (
        jnp.asarray([
            [-1,-1, 0, 0, 0], # -p1 - p2
            [ 0, 0,-1,-1, 1], # alive - q1 - q2
        ]),
        jnp.array([1, 0])
    ),
    (
        jnp.asarray([
            [1, 1] # has_three + has_two_and_alive
        ]),
        jnp.array([0, 0])
    )
]

@jax.jit
def step_mlp(input):
    return forward_mlp(CONWAY_PARAMS, input)[0]

if __name__ == "__main__":

    inputs = conway.all_inputs()
    outputs = [conway.step(input) for input in inputs]

    predicter = step_relu
    # predicter = step_relu2
    # predicter = step_mlp

    tick = time.perf_counter()
    for input, answer in zip(inputs, outputs):
        prediction = 1 if predicter(input) > 0 else 0
        if answer != prediction:
            print(f'input: {input}, answer: {answer}, prediction: {prediction}')
    tock = time.perf_counter()
    print(f'{tock - tick:.4f}s')