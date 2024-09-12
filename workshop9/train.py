"""
Reinforcement learning with proximal policy optimisation and generalised
advantage estimation in a simple maze environment.

Needs tuning.
"""

import functools
import collections

import numpy as np
import jax
import jax.numpy as jnp
import einops
import optax

from jaxtyping import Array, Float, Bool, PRNGKeyArray

import tqdm
import mattplotlib as mp

import strux
import maze
import agent


# # # 
# Training loop


def main(
    seed: int = 42,
    # environment parameters
    env_size: int = 8,
    env_wall_prob: float = 0.25,
    env_max_num_steps: int = 32,
    # agent parameters
    net_channels: int = 8,
    net_width: int = 32,
    num_conv_layers: int = 2,
    num_dense_layers: int = 4,
    # rollout parameters
    num_parallel_envs: int = 32,
    num_steps_per_rollout: int = 32,
    # training parameters
    learning_rate: float = 0.001,
    num_updates: int = 512,
    # PPO loss parameters
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
    max_grad_norm: float = 0.5,
    # visualisation params
    num_demo_envs: int = 8,
    num_train_steps_per_demo_step: int = 1,
    num_train_steps_per_visualisation: int = 1,
):
    rng = jax.random.key(seed)
    rng_setup, rng_train, rng_demo = jax.random.split(rng, 3)


    print("configuring environment...")
    env = maze.MazeEnvironment(
        size=env_size,
        wall_prob=env_wall_prob,
        max_num_steps=env_max_num_steps,
    )


    print("configuring agent...")
    net = agent.ActorCriticNetwork(
        obs_height=env_size,
        obs_width=env_size,
        obs_channels=3, # r g b
        net_channels=net_channels,
        net_width=net_width,
        num_conv_layers=num_conv_layers,
        num_dense_layers=num_dense_layers,
        num_actions=4,
    )
    rng_net, rng_setup = jax.random.split(rng_setup)
    params = net.init(rng_net)


    print("configuring optimiser...")
    optimiser = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )
    opt_state = optimiser.init(params)


    print("define training step...")
    @jax.jit
    def train_step(params, opt_state, rng_train):
        # collect experience with current policy...
        rng_rollouts, rng_train = jax.random.split(rng_train)
        rollouts = collect_rollouts(
            env=env,
            net=net,
            rng=rng_rollouts,
            params=params,
            num_envs=num_parallel_envs,
            num_steps=num_steps_per_rollout,
        )
        # estimate advantages on the collected experience...
        advantages = vmap_generalised_advantage_estimation(
            rewards=rollouts.transitions.reward,
            dones=rollouts.transitions.done,
            values=rollouts.transitions.value,
            final_values=rollouts.final_value,
            eligibility_rate=eligibility_rate,
            discount_rate=discount_rate,
        )
        # update the policy on the collected experience...
        loss, grads = jax.value_and_grad(ppo_loss_fn)(
            params,
            net=net,
            transitions=rollouts.transitions,
            advantages=advantages,
            discount_rate=discount_rate,
            proximity_eps=proximity_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # metrics
        train_metrics = {
            'loss': loss,
            'return': vmap_compute_average_return(
                rewards=rollouts.transitions.reward,
                dones=rollouts.transitions.done,
                discount_rate=discount_rate,
            ).mean(),
        }
        return params, opt_state, rng_train, train_metrics
    

    print("set up demo environments...")
    rng_demo_reset, rng_demo = jax.random.split(rng_demo)
    demo_states = env.vmap_reset(
        rng=rng_demo_reset,
        num_states=num_demo_envs,
    )
    # print(vis_demo(env, demo_states))


    print("define demo step...")
    @jax.jit
    def demo_step(demo_states, rng_demo):
        rng_demo_step, rng_demo = jax.random.split(rng_demo)
        transitions = vmap_step_and_reset(
            env=env,
            net=net,
            rng=rng_demo_step,
            states=demo_states,
            params=params,
        )
        demo_states = transitions.next_state
        return demo_states, rng_demo
    
    
    print("run training loop...")
    metrics = collections.defaultdict(list)
    for t in tqdm.trange(num_updates):
        # train step
        params, opt_state, rng_train, train_metrics = train_step(
            params,
            opt_state,
            rng_train,
        )
        
        # log the per-step metrics reported
        for name, val in train_metrics.items():
            metrics[name].append((t, val))

        # demo step
        if t % num_train_steps_per_demo_step == 0:
            demo_states, rng_demo = demo_step(demo_states, rng_demo)

        # render
        if t % num_train_steps_per_visualisation == 0:
            plot = (
                vis_metrics(metrics=metrics, total=num_updates)
                ^ vis_demo(env=env, states=demo_states)
            )
            if t == 0:
                tqdm.tqdm.write(str(plot))
            else:
                tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")


# # # 
# Collecting experience


@strux.struct
class Transition:
    state: maze.EnvState
    observation: Float[Array, "h w c"]
    value: float
    action: int
    action_prob: float
    reward: float
    done: bool
    next_state: maze.EnvState


@strux.struct
class Rollout:
    transitions: Transition
    final_observation: Float[Array, "h w c"]
    final_value: float


@functools.partial(jax.jit, static_argnames=["env", "net"])
def step_and_reset(
    env: maze.MazeEnvironment,
    net: agent.ActorCriticNetwork,
    rng: PRNGKeyArray,
    state: maze.EnvState,
    params: agent.ActorCriticNetworkParams,
) -> Transition:
    # render the state
    observation = env.render(state=state)
    # choose action
    action_probs, value = net(p=params, x=observation)
    rng_action, rng = jax.random.split(rng)
    action = jax.random.choice(
        key=rng_action,
        a=action_probs.size,
        p=action_probs,
    )
    action_prob = action_probs[action]
    # apply the action
    next_state = env.step(state=state, action=action)
    reward = env.reward(prev_state=state, next_state=next_state)
    # reset if done
    done = env.done(next_state)
    rng_reset, rng = jax.random.split(rng)
    reset_state = env.reset(rng=rng_reset)
    next_state = jax.tree.map(
        lambda new_leaf, old_leaf: jnp.where(done, new_leaf, old_leaf),
        reset_state,
        next_state,
    )
    return Transition(
        state=state,
        observation=observation,
        value=value,
        action=action,
        action_prob=action_prob,
        reward=reward,
        done=done,
        next_state=next_state,
    )


@functools.partial(jax.jit, static_argnames=["env", "net", "num_steps"])
def collect_rollout(
    env: maze.MazeEnvironment,
    net: agent.ActorCriticNetwork,
    rng: PRNGKeyArray,
    params: agent.ActorCriticNetworkParams,
    num_steps: int,
) -> Rollout:
    # initialise the state
    reset_rng, rng = jax.random.split(rng)
    init_state = env.reset(rng=reset_rng)

    # scan a sequence of steps and collect transitions
    def _step(state_and_rng, _):
        state, rng = state_and_rng
        rng_step, rng = jax.random.split(rng)
        transition = step_and_reset(
            env=env,
            net=net,
            rng=rng_step,
            state=state,
            params=params,
        )
        next_state = transition.next_state
        return (next_state, rng), transition
    final_state_and_rng, transitions = jax.lax.scan(
        _step,
        (init_state, rng),
        length=num_steps,
    )

    # compute the final observation and value
    final_state, _ = final_state_and_rng
    final_observation = env.render(state=final_state)
    _, final_value = net(p=params, x=final_observation)

    return Rollout(
        transitions=transitions,
        final_observation=final_observation,
        final_value=final_value,
    )


@functools.partial(jax.jit, static_argnames=["env", "net"])
def vmap_step_and_reset(
    env: maze.MazeEnvironment,
    net: agent.ActorCriticNetwork,
    rng: PRNGKeyArray,
    states: maze.EnvState,                      # EnvState[num_envs]
    params: agent.ActorCriticNetworkParams,
) -> Transition:                                # Transition[num_envs]
    num_envs = jax.tree.leaves(states)[0].shape[0]
    vmapped = jax.vmap(
        step_and_reset,
        in_axes=(None, None, 0, 0, None),
    )
    return vmapped(
        env,
        net,
        jax.random.split(rng, num_envs),
        states,
        params,
    )


@functools.partial(
    jax.jit,
    static_argnames=["env", "net", "num_envs", "num_steps"],
)
def collect_rollouts(
    env: maze.MazeEnvironment,
    net: agent.ActorCriticNetwork,
    rng: PRNGKeyArray,
    params: agent.ActorCriticNetworkParams,
    num_envs: int,
    num_steps: int,
) -> Rollout:                                   # Rollout[num_envs]
    vmapped = jax.vmap(
        collect_rollout,
        in_axes=(None, None, 0, None, None),
    )
    return vmapped(
        env,
        net,
        jax.random.split(rng, num_envs),
        params,
        num_steps,
    )


# # # 
# Generalised advantage estimation


@jax.jit
def generalised_advantage_estimation(
    rewards: Float[Array, "num_steps"],
    dones: Bool[Array, "num_steps"],
    values: Float[Array, "num_steps"],
    final_value: float,
    eligibility_rate: float,
    discount_rate: float,
) -> Float[Array, "num_steps"]:
    # reverse scan through num_steps axis
    initial_gae_and_next_value = (0, final_value)
    transitions = (rewards, values, dones)
    def _gae_reverse_step(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        reward, this_value, done = transition
        gae = (
            reward
            - this_value
            + (1-done) * discount_rate * (next_value + eligibility_rate * gae)
        )
        return (gae, this_value), gae
    _final_carry, gaes = jax.lax.scan(
        _gae_reverse_step,
        initial_gae_and_next_value,
        transitions,
        reverse=True,
    )
    return gaes


@jax.jit
def vmap_generalised_advantage_estimation(
    rewards: Float[Array, "num_envs num_steps"],
    dones: Bool[Array, "num_envs num_steps"],
    values: Float[Array, "num_envs num_steps"],
    final_values: Float[Array, "num_envs"],
    eligibility_rate: float,
    discount_rate: float,
) -> Float[Array, "num_envs num_steps"]:
    vmapped = jax.vmap(
        generalised_advantage_estimation,
        in_axes=(0,0,0,0,None,None),
    )
    return vmapped(
        rewards,            # float[vmap(num_levels), num_steps]
        dones,              # bool[vmap(num_levels), num_steps]
        values,             # float[vmap(num_levels), num_steps]
        final_values,       # float[vmap(num_levels)]
        eligibility_rate,   # float
        discount_rate,      # float
    )                       # -> float[vmap(num_levels), num_steps]


# # # 
# Proximal policy optimisation


@functools.partial(jax.jit, static_argnames=["net"])
def ppo_loss_fn(
    params: agent.ActorCriticNetworkParams,
    net: agent.ActorCriticNetwork,
    transitions: Transition, # Transition[num_envs, num_steps]
    advantages: Float[Array, "num_envs num_steps"],
    discount_rate: float,
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> float:
    # reshape the data to have one batch dimension
    transitions, advantages = jax.tree.map(
        lambda x: einops.rearrange(
            x,
            "n_envs n_steps ... -> (n_envs n_steps) ...",
        ),
        (transitions, advantages),
    )
    batch_size = advantages.size

    # run network to get latest predictions
    action_probs, values = jax.vmap(net, in_axes=(None, 0))(
        params,
        transitions.observation,
    ) # -> float[batch_size, 4], float[batch_size]
    chosen_action_probs = action_probs[
        jnp.arange(batch_size),
        transitions.action,
    ]

    # actor loss
    chosen_action_prob_ratios = chosen_action_probs / transitions.action_prob
    chosen_action_prob_ratios_clipped = jnp.clip(
        chosen_action_prob_ratios,
        1-proximity_eps,
        1+proximity_eps,
    )
    std_advantages = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    )
    actor_loss = -jnp.minimum(
        std_advantages * chosen_action_prob_ratios,
        std_advantages * chosen_action_prob_ratios_clipped,
    ).mean()

    # critic loss
    value_diffs = values - transitions.value
    value_diffs_clipped = jnp.clip(
        value_diffs,
        -proximity_eps,
        proximity_eps,
    )
    values_proximal = transitions.value + value_diffs_clipped
    targets = transitions.value + advantages
    critic_loss = jnp.maximum(
        jnp.square(values - targets),
        jnp.square(values_proximal - targets),
    ).mean() / 2

    # entropy regularisation term
    per_step_entropy = - jnp.sum(
        action_probs * jnp.log(action_probs),
        axis=1,
    )
    average_entropy = jnp.mean(per_step_entropy)

    # total loss
    return (
        actor_loss
        + critic_coeff * critic_loss
        - entropy_coeff * average_entropy
    )


# # # 
# Evaluating rollouts


@jax.jit
def compute_average_return(
    rewards: Float[Array, "num_steps"],
    dones: Bool[Array, "num_steps"],
    discount_rate: float,
) -> float:
    # compute per-step returns
    def _accumulate_return(
        next_step_return,
        this_step_reward_and_done,
    ):
        reward, done = this_step_reward_and_done
        this_step_return = reward + (1-done) * discount_rate * next_step_return
        return this_step_return, this_step_return
    _, per_step_returns = jax.lax.scan(
        _accumulate_return,
        0,
        (rewards, dones),
        reverse=True,
    )

    # identify start of each episode
    first_steps = jnp.roll(dones, 1).at[0].set(True)
    
    # average returns at the start of each episode
    total_first_step_returns = jnp.sum(first_steps * per_step_returns)
    num_episodes = jnp.sum(first_steps)
    average_return = total_first_step_returns / num_episodes
    
    return average_return


@jax.jit
def vmap_compute_average_return(
    rewards: Float[Array, "num_envs num_steps"],
    dones: Bool[Array, "num_envs num_steps"],
    discount_rate: float,
) -> Float[Array, "num_envs"]:
    vmapped = jax.vmap(compute_average_return, in_axes=(0,0,None))
    return vmapped(
        rewards,
        dones,
        discount_rate,
    )


# # # 
# Visualisation


def vis_demo(
    env: maze.MazeEnvironment,
    states: maze.EnvState, # EnvState[num_envs]
) -> mp.plot:
    imgs = env.vmap_render(states)
    return mp.wrap(*[mp.image(img) for img in imgs])


def vis_metrics(
    metrics: dict[str, tuple[int, float]],
    total: int,
) -> mp.plot:
    plots = []
    for metric_name, metric_data in metrics.items():
        data = np.array(metric_data)
        xs = data[:,1]
        if metric_name == 'return':
            hi = 1
        elif metric_name == "loss":
            hi = 0.05
        else:
            hi = -np.inf
        description = (
            f"min: {xs.min():.3f} | max: {xs.max():.3f} | last: {xs[-1]:.3f}"
        )
        plot = mp.border(mp.vstack(
            mp.center(mp.text(metric_name), width=38),
            mp.scatter(
                data=data,
                xrange=(0, total-1),
                yrange=(0, max(hi, xs.max())),
                color=(0.2, 1.0, 0.8),
                width=38,
                height=9,
            ),
            mp.text(description),
        ))
        plots.append(plot)
    return mp.wrap(*plots, cols=2)


# # # 
# Entry point


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

