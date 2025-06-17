import os
import tarfile
import urllib.request

def download_and_extract_scripts(
    url="ftp://ftp.cv.nrao.edu/pub/casaguides/analysis_scripts.tar",
    dest_dir="./scripts"
):
    """
    Downloads the CASA analysis utilities tar file from the specified FTP URL
    and extracts its contents into the given destination directory.
    
    Parameters:
        url (str): The URL to download the tar file from.
        dest_dir (str): The directory where the scripts will be extracted.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Define the local tar file path
    tar_path = os.path.join(dest_dir, "analysis_scripts.tar")
    
    print(f"Downloading CASA analysis utilities from: {url}")
    urllib.request.urlretrieve(url, tar_path)
    print("Download complete.")

    print(f"Extracting contents to: {dest_dir}")
    with tarfile.open(tar_path, "r:") as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete.")

    # Clean up the downloaded tar file
    os.remove(tar_path)
    print("Temporary file removed. All done.")

if __name__ == "__main__":
    download_and_extract_scripts(dest_dir="./")
