# code is adapted from: https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/utils/info.py

# from loguru import logger

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
        print("Was not able to read git information, trying to continue without.")
        # logger.error(
        #     "Was not able to read git information, trying to continue without."
        # )
        return {}
