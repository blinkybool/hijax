"""
CNN for handwritten digit classification, implemented with equinox and optax.

Preliminaries:

* any questions from homework?
* installations:
  * today we'll see `optax` (already installed?)
* download data! same as last week `cp ../workshop3/mnist.npz .

Notes:

* 'jax ecosystem'
  https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/

Workshop plan:

* starting from code similar to last time
* implement CNN with `equinox.nn` modules
* configure stateful optimiser with `optax`
* train the CNN on MNIST

Challenge (choose one, both, or your own):

* implement a drop-in replacement for `optax.adam`
* replicate some architecture, optimiser and error rate from
  Yann LeCun's table at https://yann.lecun.com/exdb/mnist/
"""

from typing import Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray as Key

import jax
import jax.numpy as jnp
import einops
import optax
import equinox

import tqdm
import draft_mattplotlib as mp


# # # 
# Architecture


# TODO


# # # 
# Training loop


def main(
    num_hidden: int = 300,
    learning_rate: float = 0.05,
    lr_schedule: bool = False,
    opt: Literal["sgd", "adam", "adamw"] = "sgd",
    batch_size: int = 512,
    num_steps: int = 256,
    steps_per_visualisation: int = 4,
    num_digits_per_visualisation: int = 15,
    seed: int = 42,
):
    key = jax.random.key(seed)


    print("initialising model...")

    
    print("loading and preprocessing data...")
    with jnp.load('mnist.npz') as datafile:
        x_train = datafile['x_train']
        x_test = datafile['x_test']
        y_train = datafile['y_train']
        y_test = datafile['y_test']
    x_train, x_test, y_train, y_test = jax.tree.map(
        jnp.array,
        (x_train, x_test, y_train, y_test),
    )
    x_train, x_test = jax.tree.map(
        lambda x: x/255,
        (x_train, x_test),
    )

    
    print("initialising optimiser...")
    # configure the optimiser
    if lr_schedule:
        learning_rate = optax.linear_schedule(
            init_value=learning_rate,
            end_value=learning_rate/100,
            transition_steps=num_steps,
        )
    if opt == 'sgd':
        optimiser = optax.sgd(learning_rate)
    elif opt == 'adam':
        optimiser = optax.adam(learning_rate)
    elif opt == 'adamw':
        optimiser = optax.adamw(learning_rate)
    # initialise the optimiser state
    opt_state = optimiser.init(model)
    
    # print(opt_state)


    print("begin training...")
    losses = []
    accuracies = []
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
        # sample a batch
        key_batch, key = jax.random.split(key)
        batch = jax.random.choice(
            key=key_batch,
            a=y_train.size,
            shape=(batch_size,),
            replace=False,
        )
        x_batch = x_train[batch]
        y_batch = y_train[batch]

        # compute the batch loss and grad
        loss, grads = jax.value_and_grad(cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # compute update, update optimiser and model
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)

        # track metrics
        losses.append((step, loss))
        test_acc = accuracy(model, x_test[:1000], y_test[:1000])
        accuracies.append((step, test_acc))


        # visualisation!
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            digit_plot = vis_digits(
                digits=x_test[:num_digits_per_visualisation],
                true_labels=y_test[:num_digits_per_visualisation],
                pred_labels=model(
                    x_test[:num_digits_per_visualisation]
                ).argmax(axis=-1),
            )
            metrics_plot = vis_metrics(
                losses=losses,
                accuracies=accuracies,
                total_num_steps=num_steps,
            )
            # TODO: better state visualisation
            opt_state_str = str(opt_state)
            output_height = (
                digit_plot.height
                + metrics_plot.height
                + 1+len(opt_state_str.splitlines())
            )
            tqdm.tqdm.write(
                (f"\x1b[{output_height}A" if step > 0 else "")
                + f"{digit_plot}\n"
                + f"{metrics_plot}\n"
                + f"optimiser state:\n{opt_state_str}"
            )


# # # 
# Metrics


def cross_entropy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    """
    Hx(q, p) = - Sum_i p(i) log q(i)
    """
    batch_size, = y_batch.shape
    pred_prob_all_classes = model(x_batch)          # -> batch_size 10
    pred_prob_true_class = pred_prob_all_classes[   # advanced indexing
        jnp.arange(batch_size),                     # for each example
        y_batch,                                    # select prob of true class
    ]                                               # -> batch_size
    log_prob_true_class = jnp.log(pred_prob_true_class)
    avg_cross_entropy = -jnp.mean(log_prob_true_class)
    return avg_cross_entropy


def accuracy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    pred_prob_all_classes = model(x_batch)
    highest_prob_class = pred_prob_all_classes.argmax(axis=-1)
    return jnp.mean(y_batch == highest_prob_class)


# # # 
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    true_labels: Int[Array, "n"],
    pred_labels: Int[Array, "n"] | None = None,
) -> mp.plot:
    # downsample images
    ddigits = digits[:,::2,::2]
    dwidth = digits.shape[2] // 2

    # if predictions provided, classify as true or false
    if pred_labels is not None:
        corrects = (true_labels == pred_labels)
        cmaps = [None if correct else mp.reds for correct in corrects]
        labels = [f"{t} ({p})" for t, p in zip(true_labels, pred_labels)]
    else:
        cmaps = [None] * len(true_labels)
        labels = [str(t) for t in true_labels]
    array = mp.wrap(*[
        mp.border(
            mp.image(ddigit, colormap=cmap)
            ^ mp.center(mp.text(label), width=dwidth)
        )
        for ddigit, label, cmap in zip(ddigits, labels, cmaps)
    ], cols=5)
    return array


def vis_metrics(
    losses: list[tuple[int, float]],
    accuracies: list[tuple[int, float]],
    total_num_steps: int,
) -> mp.plot:
    loss_plot = (
        mp.center(mp.text("train loss (cross entropy)"), width=40)
        ^ mp.border(mp.scatter(
            data=losses,
            xrange=(0, total_num_steps-1),
            yrange=(0, max(l for s, l in losses)),
            color=(1,0,1),
            width=38,
            height=11,
        ))
        ^ mp.text(f"loss: {losses[-1][1]:.3f}")
    )
    acc_plot = (
        mp.center(mp.text("test accuracy"), width=40)
        ^ mp.border(mp.scatter(
            data=accuracies,
            xrange=(0, total_num_steps-1),
            yrange=(0, 1),
            color=(0,1,0),
            width=38,
            height=11,
        ))
        ^ mp.text(f"acc: {accuracies[-1][1]:.2%}")
    )
    return loss_plot & acc_plot


# # # 
# Entry point

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
