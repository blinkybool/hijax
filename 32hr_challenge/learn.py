import jax
import jax.numpy as jnp
import itertools
import conway
import tqdm
import time

def random_layer_params(m, n, key):
    '''
    init weights and biases for 
    '''
    w_key, b_key = jax.random.split(key)
    return jax.random.normal(w_key, (n,m)), jax.random.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes)-1)
    return [random_layer_params(m,n,k) for m,n,k in zip(sizes[:-1], sizes[1:], keys)]

def update_params(params, gradient, learning_rate=0.01):
    new_params = []
    for (w,b), (dw, db) in zip(params, gradient):
        new_params.append(
            (w - learning_rate * dw, b - learning_rate * db)
        )
    return new_params

def forward(params, input):
    layer = input
    for w,b in params[:-1]:
        layer = jax.nn.relu(jnp.dot(w,layer) + b)
    w, b = params[-1]
    layer = jnp.dot(w,layer) + b
    
    return layer

LAYER_SIZES = [9, 9, 3, 1]

def main(
    seed: int = 0,
    learning_rate: float = 0.01,
    epochs: int = 10000,
):
    
    key = jax.random.key(seed)

    key, key_params = jax.random.split(key)
    params = init_network_params(LAYER_SIZES, key_params)

    inputs = jnp.asarray(list(itertools.product((0,1), repeat=9)))
    outputs = jnp.asarray([conway.step(input) for input in inputs])

    def predict(params, input):
        output = forward(params, input)[0]
        return jax.nn.sigmoid(output)

    @jax.jit
    def loss(params):
        predictions = jax.vmap(
            predict,
            in_axes=(None,0)
        )(params, inputs)
        # This isn't really what we want, we'd rather do something
        # like -ve for 0 and +ve for 1
        return jnp.mean((predictions - outputs) ** 2)

    value_and_grad = jax.value_and_grad(loss)

    for t in tqdm.trange(epochs):
        l, g = value_and_grad(params)
        params = update_params(params, g, learning_rate=learning_rate)
        if t % 10 == 0:
            tqdm.tqdm.write(f'loss: {l:.6f}')
            time.sleep(.02)

    num_correct = len(inputs)
    for input, answer in zip(inputs, outputs):
        prediction = 1 if predict(params, input) > 0 else 0
        if answer != prediction:
            num_correct -= 1
            print(f'input: {input}, answer: {answer}, prediction: {prediction}')

    print(f'{num_correct}/{len(inputs)} correct')

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
