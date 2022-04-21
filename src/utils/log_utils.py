# code is adapted from: https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/utils/info.py

from loguru import logger
import os
from omegaconf import OmegaConf

from pathlib import Path
from typing import Union, Optional
from git import Repo, InvalidGitRepositoryError


def get_repo_info(path: Union[str, Path]):
    """
    Parse repository information from path
    Args:
        path (str): path to repo. If path is not a repository it
        searches parent folders for a repository
    Returns:
        dict: contains the current hash, gitdir and active branch
    """

    def find_repo(findpath):
        p = Path(findpath).absolute()
        for p in [p, *p.parents]:
            try:
                repo = Repo(p)
                break
            except InvalidGitRepositoryError:
                pass
        else:
            raise InvalidGitRepositoryError
        return repo

    repo = find_repo(path)
    return {
        "hash": repo.head.commit.hexsha,
        "gitdir": repo.git_dir,
        "active_branch": repo.active_branch.name,
    }


def log_git(repo_path: Union[Path, str], repo_name: str = None):
    """
    Use python logging module to log git information
    Args:
        repo_path (Union[pathlib.Path, str]): path to repo or file inside repository (repository is recursively searched)
    """
    try:
        git_info = get_repo_info(repo_path)
        return git_info
    except Exception:
        logger.error(
            "Was not able to read git information, trying to continue without."
        )
        return {}


def save_config_to_tests(cfg, save_name):
    """Place this function inside of a main to extract the OmegaConf dictionary.

    Args:
        cfg (_type_): _description_
        save_name (_type_): _description_
    """
    from omegaconf import OmegaConf
    from utils.path_utils import test_data_folder
    import utils.io as io
    import os

    save_name = save_name.replace(".py", "")
    save_file = os.path.join(test_data_folder, save_name)
    io.save_omega_conf(cfg, save_file)
