
import jax
import jax.numpy as jnp
import jax.random as random




def perturbate_y0_(y0, a,b,c,key):
    dim = jnp.size(y0)
    nzeros = (y0==0).sum()
    nonzeros = dim  - nzeros

    #prg keys 
    key1, subkey1 = random.split(key)
    nkey, subkey2 = random.split(key1)

    #random numbers
    u = jnp.exp (-a + (-a + b) * random.uniform(subkey1, (1,nzeros)))
    v = jnp.exp (-c + 2*c*random.uniform(subkey2, y0.shape))

    #perturbating
    y1 = y0.at[y0==0].set(u.squeeze())
    y1 = y1*v



    return nkey, y1


def perturbate_y0(y0,a,b,c,key, n_perturbations=1):
    y0 = jnp.stack([y0])
    y0 = jnp.repeat(y0, repeats=n_perturbations, axis=0)

    return perturbate_y0_(y0, a,b,c,key)












