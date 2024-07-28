import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray as Key
from typing import List
import itertools
import conway
import equinox
import tqdm

class ConwayModel(equinox.Module):
    layers: List[equinox.nn.Linear]

    def __init__(self, key: Key, hidden_dims: List[int]):
        self.layers = []
        layer_dims = [9] + hidden_dims + [1]
        keys = jax.random.split(key, len(layer_dims)-1)
        for in_features, out_features, key_layer in zip(layer_dims[:-1], layer_dims[1:], keys):
            self.layers.append(equinox.nn.Linear(
                key=key_layer,
                in_features=in_features,
                out_features=out_features,
                use_bias=True,
            ))

    def __call__(self, input: Array):
        x = input
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
        x = self.layers[-1](x)
        x = jax.nn.sigmoid(x)
        return x[0]

def cross_entropy(
    model: ConwayModel,
    input: Array,
    alive: Int
) -> float:
    """
    Hx(q, p) = -Sum_i p(i) log q(i)
    """
    prob_alive = model(input)
    # prob_true is prob_alive if alive == 1, and 1-prob_alive if alive == 0
    prob_true = prob_alive * alive + (1 - prob_alive) * (1 - alive)
    return -jnp.log(prob_true)

@jax.jit
def total_correct(model: ConwayModel, inputs, outputs):
    predictions = jax.vmap(lambda x: model(x) > 0.5)(inputs)
    return jnp.sum(predictions == outputs)

def main(
    seed: int = 0,
    hidden_dims: List[int] = [6,6,3],
    learning_rate: float = 0.01,
    epochs: int = 10000,
):

    inputs = jnp.array(list(itertools.product((0,1), repeat=9)))
    outputs = jax.vmap(conway.step)(inputs)

    def loss_fn(model):
        entropies = jax.vmap(
            cross_entropy,
            in_axes=(None,0,0)
        )(model, inputs, outputs)
        return jnp.mean(entropies)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    key = jax.random.key(seed)

    model = ConwayModel(key=key, hidden_dims=hidden_dims)

    for t in tqdm.trange(epochs):
        loss, grads = value_and_grad(model)
        model = jax.tree.map(
            lambda w, g: w - g * learning_rate,
            model,
            grads
        )
        if t % 100 == 0 or t == epochs-1:
            tqdm.tqdm.write(f'loss: {loss:.6f}, {total_correct(model, inputs, outputs)}/{len(outputs)} correct')

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
