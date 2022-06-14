import jax
import jax.numpy as jnp

def hertzian():
    return lambda x: x[0]*x[0]*x[1] + x[0]*x[1]

f = hertzian()

hess = jax.jacfwd(jax.jacrev(jax.grad(f)))

result = hess(jnp.array([1., 1.]))

print(result)