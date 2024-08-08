import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray as Key

class Attention(eqx.Module):
    W_query: Float[Array, "kq_dim e_dim"]
    W_key: Float[Array, "kq_dim e_dim"]
    W_value: Float[Array, "v_dim e_dim"]

    def __init__(self,
        entity_dim: int,
        keyquery_dim: int,
        value_dim: int,
        key: Key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        keyquery_lim = jnp.sqrt(6/(entity_dim + keyquery_dim))
        value_lim = jnp.sqrt(6/(entity_dim + value_dim))
        self.W_query = jax.random.uniform(key=k1, shape=(keyquery_dim, entity_dim), minval=-keyquery_lim, maxval=keyquery_lim)
        self.W_key   = jax.random.uniform(key=k2, shape=(keyquery_dim, entity_dim), minval=-keyquery_lim, maxval=keyquery_lim)
        self.W_value = jax.random.uniform(key=k3, shape=(value_dim, entity_dim), minval=-value_lim, maxval=value_lim)

    def forward(
        self,
        entities: Float[Array, "ent e_dim"]
    ) -> Float[Array, "ent v_dim"]:
        # i=kq_dim, j=e_dim, k=v_dim, e=ent
        queries = jnp.einsum('ij,ej->ei', self.W_query, entities)
        keys    = jnp.einsum('ij,ej->ei', self.W_key, entities)
        values  = jnp.einsum('kj,ej->ek', self.W_value, entities)

        # d=e=ent (e and d must be distinct labels for einsum)
        logits = jnp.einsum('ei,di->ed', queries, keys)
        alignment = jax.nn.softmax(logits)
        return jnp.einsum('ed,dk->ek', alignment, values)

class TransformerBlock(eqx.Module):
    Attn: Attention
    FF: eqx.nn.Linear
    LN: eqx.nn.LayerNorm

    def __init__(self,
        entity_dim: int,
        keyquery_dim: int,
        value_dim: int,
        key: Key,
    ):
        k1, k2 = jax.random.split(key)
        self.Attn = Attention(
            entity_dim=entity_dim,
            keyquery_dim=keyquery_dim,
            value_dim=value_dim,
            key=k1
        )
        self.FF = eqx.nn.Linear(
            in_features=value_dim,
            out_features=entity_dim,
            key=k2
        )
        self.LN = eqx.nn.LayerNorm(entity_dim)

    def forward(self, entities: Float[Array, "e n"]):
        refined_values = self.Attn.forward(entities)
        entity_deltas = jax.vmap(self.FF)(refined_values)
        refined_entities = jax.vmap(self.LN)(entities + entity_deltas)
        return refined_entities

if __name__ == "__main__":

    attn = Attention(
        entity_dim=4,
        keyquery_dim=2,
        value_dim=3,
        key = jax.random.key(0)
    )

    jax.tree.unflatten(jax.tree.structure(attn), (jnp.zeros((2,2)), jnp.zeros((2,2)), jnp.zeros((2,2))))

    attn = eqx.tree_at(lambda x: x.W_query, attn, jnp.array([
        [0.1, -0.1, 0.3, 0.2],
        [0.2, 0.1, -0.2, 0.1]
    ]))

    attn = eqx.tree_at(lambda x: x.W_key, attn, jnp.array([
        [0.2, 0.1, -0.2, 0.3],
        [-0.1, 0.3, 0.2, 0.1]
    ]))

    attn = eqx.tree_at(lambda x: x.W_value, attn, jnp.array([
        [0.1, -0.1, 0.2, 0.3],
        [0.2, 0.3, -0.1, 0.1],
        [-0.1, 0.2, 0.3, 0.2]
    ]))

    entities = jnp.array([
        [1.0, 0.5, -0.5, 0.8],
        [0.2, -0.3, 0.7, 0.1],
        [0.6, 0.4, 0.9, -0.2]
    ])
    
    transformer = TransformerBlock(
        entity_dim=4,
        keyquery_dim=50,
        value_dim=50,
        key = jax.random.key(0),
    )

    # print(attn.forward(entities))
    # print(jax.jit(Attention.forward)(attn, entities))
    # print(jax.jit(TransformerBlock.forward)(transformer, entities))
    print(TransformerBlock.forward(transformer, entities))