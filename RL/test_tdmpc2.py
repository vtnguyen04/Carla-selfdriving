import pathlib
import sys
import numpy as np
import jax
import jax.numpy as jnp
import embodied
import ruamel.yaml as yaml

# Add project root to path
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from RL.tdmpc2.agent import TDMPC2Agent
from unittest.mock import MagicMock

# Mock necessary modules
sys.modules["carla"] = MagicMock()
sys.modules["cv2"] = MagicMock()

# Mock jax.config.update to allow suppressing strict guard for test
original_update = jax.config.update
def mock_update(name, value):
    if name == "jax_transfer_guard":
        return
    original_update(name, value)
jax.config.update = mock_update

def test_agent():
    print("Initializing TDMPC2 Agent with real config...")
    
    # Load Real Config from YAML to ensure all keys exist
    config_path = pathlib.Path(__file__).parent / "tdmpc2/tdmpc2.yaml"
    model_configs = yaml.YAML(typ="safe").load(config_path.read_text())
    
    # Use 'xxsmall' as base for speed, update with defaults
    config = embodied.Config(model_configs["defaults"])
    config = config.update(model_configs["xxsmall"])
    
    # Override for testing environment (CPU, no JIT for debugging, small batch)
    config = config.update({
        'batch_size': 4,
        'batch_length': 16,
        'num_samples': 16,
        'iterations': 2,
        'jax': {
            'jit': True, 
            'platform': 'cpu', 
            'prealloc': False,
            'logical_cpus': 0,
            'debug': False,
            'debug_nans': False,
            'precision': 'float32',
            'policy_devices': [0],
            'train_devices': [0],
            'metrics_every': 10
        }
    })

    # Mock Spaces
    obs_space = {
        'image': embodied.Space(np.uint8, (64, 64, 3)),
        'reward': embodied.Space(np.float32, ()),
        'is_first': embodied.Space(bool, ()),
        'is_terminal': embodied.Space(bool, ()),
    }
    act_space = {
        'action': embodied.Space(np.float32, (2,), low=-1.0, high=1.0),
    }
    
    step = embodied.Counter()
    agent = TDMPC2Agent(obs_space, act_space, step, config)
    
    # Mock Data
    print("Creating mock data...")
    batch_size = 4
    seq_len = 16
    data = {
        'image': np.random.randint(0, 255, (batch_size, seq_len, 64, 64, 3), dtype=np.uint8),
        'action': np.random.uniform(-1, 1, (batch_size, seq_len, 2)).astype(np.float32),
        'reward': np.random.uniform(0, 1, (batch_size, seq_len)).astype(np.float32),
        'is_first': np.zeros((batch_size, seq_len), dtype=bool),
        'is_terminal': np.zeros((batch_size, seq_len), dtype=bool),
    }
    
    # Test Policy
    print("Testing Policy...")
    obs = {k: v[:, 0] for k, v in data.items()}
    # Convert obs to device
    obs = agent._convert_inps(obs, agent.policy_devices)
    
    outs, next_state = agent.policy(obs, state=None, mode='train')
    print(f"Policy Output Action Shape: {outs['action'].shape}")
    assert outs['action'].shape == (batch_size, 2)
    
    # Test Train
    print("Testing Train...")
    # Convert data to device arrays
    data = agent._convert_inps(data, agent.train_devices)
    outs, next_train_state, metrics = agent.train(data, state=None)
    print("Train Metrics:", metrics.keys())
    
    # Verify key metrics exist
    assert 'model_opt_loss' in metrics
    assert 's_loss' in metrics # Check smoothness loss exists
    
    print("TDMPC2 Agent Test Passed!")

if __name__ == "__main__":
    test_agent()