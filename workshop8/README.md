Workshop 8: Hi, branching computation!
===============================================================================

Preliminaries:

* Any questions?
* Installations: `pip install readchar`
* Data: no, we'll make our own

Today's files:

```
workshop8/
├── README.md             # (you are here)
├── maze.py               # we'll implement an accelerated maze environment
├── strux.py              # (couldn't help myself, bye equinox...!)
└── mattplotlib.py        # (coming some day to PyPI...)
```

Reinforcement learning in JAX
-----------------------------

RL systems have three components:

1. An 'environment', some interactive state machine.
2. An 'agent', a function from observation to actions.
3. A 'reward function' that scores interactions between the two.

*Deep reinforcement learning* uses a (hardware-accelerated) neural network to
'learn' an agent from samples of experience scored by the reward function.

*Reward modelling* uses a (hardware-accelerated) neural network trained on
e.g. human feedback (as essentially a supervised learning problem).

But until recently, it was normal for the environment itself to be a CPU
program, for example:

* a classical game console emulator, or
* a physics engine with a robotics simulator.

The field developed complex algorithms that can learn from experience
collected across massive numbers of CPU workers while training the neural
network agent on the GPU.

JAX lets us hardware-accelerate environments too! Yay!

But, environment code is often pretty different from neural network code.

* Neural network forward passes are mostly about computating straightforward
  expressions without different 'cases'.

* Environments (e.g. games) often involve **branching computation,** for
  example:

  * Move the hero forward unless there is a wall in the way.

  * Kill the monster if its heath drops to 0.

  * Reset the game if the hero reaches the goal.

Branching computation
---------------------

Branching computation is not just about when a function's values depend on
its inputs. Of course, that just means the function is not a constant.

Branching computation is when the *abstract computational steps* required to
get from the inputs to the outputs depends on the input values.[^abstract]

In python this is one of the first things you learn how to do:

```
def max(x, y):
  if y - x > 0:
      return y
  else:
      return x
```

By contrast, the above isn't a jittable JAX function.

The tracer can only follow one of the two paths, and it needs to know the
concrete value of `condition(x)` to figure out which one.

This is necessary so that JAX can output one linear sequence of program steps
for the hardware, and the hardware can just follow them step by step
(parallelised over many inputs) without stopping all the time to check the
values of each input.


[^abstract]:
    The definition of branching is therefore relative to what counts as an
    abstract computational step. Philosophically this is an important choice
    with different interesting answers, but for today there's one clear
    answer: XLA decides what is an abstract computational step.


How to do branching computation in JAX
--------------------------------------

Yet, we see what appear to be branching computations in JAX frequently:

```
jnp.max(x, y) # works fine
```

How can we have branching computations in JAX?

The answer is... **we don't.**

When our Python-native brain reaches for branching computation, we need to
rewrite that computation as a linear sequence of computational steps.

This works because some of the computational steps handle the branching
themselves. We need to implement the branching computation as a linear
sequence of these kinds of steps.

Examples using things we have seen so far:

1.  Indexing with dynamic values is allowed, and can be used for branching.
    ```
    def max(x, y):
        both_values = jnp.array([x, y])
        y_is_greater = (y - x > 0)
        index_of_max = y_is_greater.astype(int)
        return both_values[index_of_max]            # XLA 'scatter' opteration
    ```
    (not ideal: tedious, requires moving values to contiguous memory)

2.  Summation with boolean coefficients can also be used for branching:
    ```
    def max(x, y):
        y_is_greater = (y - x > 0)
        return y * (y_is_greater) + x * (~y_is_greater)
    ```
    (not ideal: messy and requires some gratuitous flops)

New examples we'll meet today:

* `jax.numpy.where`...
* `jax.lax.select`...
* `jax.lax.cond`...


Workshop
--------

So... let's use this to implement an environment!


Challenges
----------

There are lots of potentially interesting ways to build on today's demo:

1. Determine solvability of a generated level by implementing an accelerated
   DFS or BFS algorithm.

2. Generate perfect-maze level layouts using Kruskal's algorithm. Lots of
   branching.

3. Implement an environment with an array of 'keys' that can be collected to
   unlock an array of 'chests' containing reward. Lots and lots of branching.

Next week
---------

We now have an environment (and a reward function), we'll turn this into an
RL system by implementing an agent network and an RL training algorithm.
