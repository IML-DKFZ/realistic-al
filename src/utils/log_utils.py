# code is adapted from: https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/utils/info.py

import os
from optparse import Option
from pathlib import Path
from typing import Optional, Union

import __main__
from git import InvalidGitRepositoryError, Repo
from loguru import logger
from omegaconf import OmegaConf


def setup_logger(path: Optional[Union[Path, str]] = None):
    logger.info("Current Working Directory: {}".format(os.getcwd()))
    logger.info("Main: {}".format((__main__.__file__.split(".")[0] + ".log")))
    if path is None:
        path = os.getcwd()
    path = Path(path)
    logpath = path / (Path(__main__.__file__).name.split(".")[0].__str__() + ".log")
    logger.add(logpath)
    logger.info("Logging to file: {}".format(logpath))


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
    import os

    from omegaconf import OmegaConf

    import utils.io as io
    from utils.path_utils import test_data_folder

    save_name = save_name.replace(".py", "")
    save_file = os.path.join(test_data_folder, save_name)
    io.save_omega_conf(cfg, save_file)
