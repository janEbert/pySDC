import jax
import jax.numpy as jnp
from jax.experimental import stax


Real = stax.elementwise(jnp.real)


def _from_model_arch(model_arch, train):
    # For smaller intervals, this should be higher (e.g. 1e-3);
    # for larger intervals lower is better (e.g. 1e-7).
    scale = 1e-7
    glorot_normal = jax.nn.initializers.variance_scaling(
        scale, "fan_avg", "truncated_normal")
    normal = jax.nn.initializers.normal(scale)

    dropout_rate = 0.0
    mode = 'train' if train else 'test'
    dropout_keep_rate = 1 - dropout_rate

    model_arch_real = []
    for tup in model_arch:
        if not isinstance(tup, tuple):
            tup = (tup,)
        name = tup[0]
        if len(tup) > 1:
            args = tup[1]
        if len(tup) > 2:
            kwargs = tup[2]

        if name == 'Real':
            layer = Real
        else:
            layer = getattr(stax, name)

        if name == 'Dense':
            args = args + (glorot_normal, normal)
        elif name == 'Dropout':
            args = args + (dropout_keep_rate, mode)

        if len(tup) == 1:
            model_arch_real.append(layer)
        elif len(tup) == 2:
            model_arch_real.append(layer(*args))
        elif len(tup) == 3:
            model_arch_real.append(layer(*args, **kwargs))
        else:
            raise ValueError('error in model_arch syntax')
    (model_init, model_apply) = stax.serial(*model_arch_real)
    return (model_init, model_apply)


def load_model(path):
    with open(path, 'rb') as f:
        weights = jnp.load(f, allow_pickle=True)
    with open(str(path) + '.structure', 'rb') as f:
        model_arch = jnp.load(f, allow_pickle=True)
    model = _from_model_arch(model_arch, train=False)
    return weights, model
