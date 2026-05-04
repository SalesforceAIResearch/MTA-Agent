from verl.workers.reward_manager.registry import register, REWARD_MANAGER_REGISTRY
from pathlib import Path

error_loaded_reward_manager = {}
def get_reward_manager_cls(name):
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER_REGISTRY:
        if name in error_loaded_reward_manager:
            print("Error loading reward manager:", name, "Please check your dependencies.")
            raise error_loaded_reward_manager[name]
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]

# search current directory for reward manager classes
current_dir = Path(__file__).parent

# Import .py files directly in the current directory
for file in current_dir.glob("*.py"):
    if file.name == "__init__.py":
        continue
    try:
        # import
        module = __import__(f"verl_tool.workers.reward_manager.{file.stem}", fromlist=[file.stem])
    except ImportError as e:
        error_loaded_reward_manager[file.stem] = e
        pass

# Import reward managers from subdirectories
for subdir in current_dir.iterdir():
    if subdir.is_dir() and not subdir.name.startswith('_') and not subdir.name == '__pycache__':
        # Check if subdirectory has an __init__.py (is a package)
        init_file = subdir / "__init__.py"
        if init_file.exists():
            try:
                # Import the package
                module = __import__(f"verl_tool.workers.reward_manager.{subdir.name}", fromlist=[subdir.name])
            except ImportError as e:
                error_loaded_reward_manager[subdir.name] = e
                pass

import verl.workers.reward_manager.registry
verl.workers.reward_manager.registry.get_reward_manager_cls = get_reward_manager_cls