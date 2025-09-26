import importlib.util
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from reinvent.utils import config_parse, setup_logger

from navidiv.reinvent import run_staged_learning_2
from navidiv.reinvent.InputGenerator import InputGenerator

LOGLEVEL_CHOICES = tuple(
    level.lower() for level in logging._nameToLevel.keys()
)

logger = logging.getLogger(__name__)


def _loadInputGeneratorFromFile(file_path: str, class_name: str):
    """Load InputGenerator class from file path.

    Args:
        file_path: Path to Python file containing the class
        class_name: Name of the class to load

    Returns:
        InputGenerator class

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is not a Python file
        AttributeError: If class is not found in file
        ImportError: If module loading fails
    """
    try:
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path_obj.suffix == ".py":
            raise ValueError(f"File must be a Python file (.py): {file_path}")

        # Create module spec and load module
        spec = importlib.util.spec_from_file_location(
            f"custom_input_generator_{file_path_obj.stem}", file_path_obj
        )
        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules to enable relative imports
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return getattr(module, class_name)

    except (FileNotFoundError, ValueError):
        raise
    except AttributeError as e:
        raise AttributeError(
            f"Class '{class_name}' not found in file '{file_path}'"
        ) from e
    except Exception as e:
        raise ImportError(
            f"Failed to load module from file '{file_path}': {e}"
        ) from e


@hydra.main(
    config_path="/media/mohammed/Work/Navi_diversity/reinvent_runs/conf_folder",
    config_name="test",
    version_base="1.1",  # Explicitly specify the compatibility version
)
def main(cfg: DictConfig):
    """Main function to run REINVENT with the given configuration."""
    # in case the input generator is defined in the config file
    if "input_generator" in cfg:
       
        inputGeneratorClass = _loadInputGeneratorFromFile(
            cfg.input_generator.file_path,
            cfg.input_generator.class_name
        )
        input_generator = inputGeneratorClass(cfg)

    else:
        input_generator = InputGenerator(cfg)
    
    print(cfg.diversity_scorer)
    os.chdir(input_generator.wd)
    input_generator.generate_input()
    input_config = config_parse.read_toml(
        os.path.join(input_generator.wd, "stage1.toml"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(name="reinvent", filename="stage1.log")
    device = torch.device(device)
    seed = input_config.get("seed", None)
    tb_logdir = input_config.get("tb_logdir", None)

    if tb_logdir:
        tb_logdir = os.path.abspath(tb_logdir)
        logger.info(f"Writing TensorBoard summary to {tb_logdir}")
    if getattr(cfg, "generate_input_only", False):
        logger.info("Input generation only, exiting.")
        return
    run_staged_learning_2.run_staged_learning(
        extract_sections(input_config),
        device,
        tb_logdir=tb_logdir,
        write_config=None,
        responder_config=False,
        seed=seed,
        diversity_scorer=cfg.diversity_scorer,
    )


def extract_sections(config: dict) -> dict:
    """Extract the sections of a config file

    :param config: the config file
    :returns: the extracted sections
    """
    # FIXME: stages are a list of dicts in RL, may clash with global lists
    return {k: v for k, v in config.items() if isinstance(v, (dict, list))}


if __name__ == "__main__":
    main()
