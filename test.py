import jax.numpy as jnp
from dreamerv3 import jaxutils
from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="test")

import jax


def test_compatibility():
    """Test xem code có chạy được không"""

    # Test simnorm
    x = jnp.ones((2, 32))
    z = jaxutils.simnorm(x, groups=8)
    log.info(f"✓ simnorm: {z.shape}, sum per group = {z.reshape(2, 8, 4).sum(-1)[0]}")

    # Test VICRegLoss
    pred = jnp.ones((4, 16))
    target = jnp.zeros((4, 16))
    loss_fn = jaxutils.VICRegLoss()
    loss, metrics = loss_fn(pred, target)
    log.info(f"✓ VICRegLoss: {loss}")

    # Test JEPALoss
    loss_fn2 = jaxutils.JEPALoss()
    loss2, metrics2 = loss_fn2(pred, target)
    log.info(f"✓ JEPALoss: {metrics2}")

    # Test CollapseMetrics
    z = jnp.ones((10, 64))
    metrics = jaxutils.CollapseMetrics.compute(z, prefix="test_")
    log.info(f"✓ CollapseMetrics: {list(metrics.keys())}")

    # Test compute_collapse_metrics (wrapper)
    metrics2 = jaxutils.compute_collapse_metrics(z, prefix="test2_")
    log.info(f"✓ compute_collapse_metrics: {list(metrics2.keys())}")

    # Test ema_update
    params1 = {"w": jnp.ones((3, 3))}
    params2 = {"w": jnp.zeros((3, 3))}
    new_params = jaxutils.ema_update(params1, params2, decay=0.9)
    log.info(f"✓ ema_update: {new_params['w'][0, 0]}")  # Should be 0.1

    log.info("\n✅ All tests passed!")


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "gpu")
    test_compatibility()
