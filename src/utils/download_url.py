import os
import requests
import zipfile


def download_url(url, save_path):
    """Chunk wise downloading to not overuse RAM.

    Parameters
    ----------
    url : str
        URL from where to download.
    save_path : str
        Path where to save file.
    """
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
