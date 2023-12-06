import requests
from tqdm import tqdm


def download_url(url: str, save_path: str):
    """Chunk wise downloading to not overuse RAM.


    Args:
        url (str): URL from where to download.
        save_path (str): Path where to save file.
    """
    r = requests.get(url, stream=True, allow_redirects=True)
    total_length = int(r.headers.get("content-length"))
    print("Downloading File of Size: {} GB".format(total_length / 1024**3))
    dl = 0
    with open(save_path, "wb") as f:
        if total_length is None:  # no content length header
            f.write(r.content)
        else:
            pbar = tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total_length,
            )
            pbar.clear()  #  clear 0% info

            for chunk in r.iter_content(chunk_size=500 * 1024):
                if chunk:
                    dl += len(chunk)
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))


if __name__ == "__main__":
    import os

    try:
        download_url(
            "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
            "test.zip",
        )
    finally:
        os.remove("test.zip")
