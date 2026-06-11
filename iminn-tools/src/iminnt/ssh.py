import os
import re
from pathlib import Path
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass, field
from stat import S_ISDIR

import paramiko
from paramiko.agent import Agent
from paramiko.ssh_exception import SSHException


from .log_cfg import logger, set_log_file
from .tqdm import tqdm
from .utils import shell


def initialize_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    return ssh


def walk_sftp_directory(sftp: paramiko.SFTPClient, remote_path: str):
    """SFTP implementation of os.walk."""
    files, folders = [], []
    for f in sftp.listdir_attr(remote_path):
        if S_ISDIR(f.st_mode if f.st_mode else 0):
            folders.append(f.filename)
        else:
            files.append(f.filename)
    return files, folders


def walk_local_directory(local_path: Path):
    """SFTP implementation of os.walk."""
    files, folders = [], []
    for f in local_path.iterdir():
        if f.is_dir():
            folders.append(f)
        else:
            files.append(f)
    return files, folders


def calculate_remote_folder_size(client: paramiko.SSHClient, folder_path: str) -> int:
    """Calculate the total size of the folder before compression."""
    stdin, stdout, stderr = client.exec_command(f"du -sb {folder_path}")
    size_info = stdout.read().decode().strip().split("\t")[0]
    return int(size_info)


def calculate_local_folder_size(folder_path: str) -> int:
    """Calculate the total size of the folder before compression."""
    fp = folder_path if isinstance(folder_path, Path) else Path(folder_path)
    res = shell(f"du -sb {fp}")
    size_info = res.strip().split("\t")[0]
    return int(size_info)


def download_file_with_progress(sftp: paramiko.SFTPClient, remote_path, local_path, total_bytes):
    """Download a file and emit progress using FileProgress."""
    try:
        progress: tqdm = tqdm(total=total_bytes, unit="bytes", desc=f"{remote_path}")

        def download_progress_callback(transferred, total):
            progress.update(transferred)

        # Perform the download
        sftp.get(remote_path, str(local_path), callback=download_progress_callback)
        progress.update(close=True)
    except OSError as e:
        logger.error(f"Error downloading file {remote_path} to {local_path}: {str(e)}")
        raise


def upload_file_with_progress(sftp: paramiko.SFTPClient, remote_path, local_path, total_bytes):
    """Upload a file and emit progress using FileProgress."""
    try:
        progress: tqdm = tqdm(total=total_bytes, unit="bytes", desc=f"{local_path}")

        def upload_progress_callback(transferred, total):
            progress.update(transferred)

        # Perform the upload
        sftp.put(str(local_path), remote_path, callback=upload_progress_callback)
        progress.update(close=True)
    except OSError as e:
        logger.error(f"Error uploading file {local_path} to {remote_path}: {str(e)}")
        raise


def is_excluded(file_path, exclude_patterns):
    """Check if the file matches any exclusion pattern."""
    return any(re.search(pattern, file_path) for pattern in exclude_patterns)


# Helpful resources demonstrating paramiko usage:
# - https://github.com/Tencent/TenDBCluster-TenDB/blob/tendb/storage/ndb/mcc/remote_clusterhost.py
# - https://github.com/tenstorrent/ttnn-visualizer/blob/dev/backend/ttnn_visualizer/sftp_operations.py
# - (relevant info for setting up ssh-agent): https://github.com/tenstorrent/ttnn-visualizer/blob/dev/docs/docker.md?plain=1
# TODO: Incorporate password into connection setup
@dataclass
class RemoteConn:
    name: str
    uname: str
    host: str
    port: int = field(default=22)
    password: Optional[str] = field(default=None)
    timeout: Optional[int] = field(default=5)
    client: Optional[paramiko.SSHClient] = None
    use_agent: bool = field(default=False)

    def __post_init__(self):
        assert self.port >= 1 and self.port <= 65535, (
            f"{self.port} is an invalid port value. Must be between 1 and 65535"
        )

    def get_connection_args(self) -> dict:
        use_agent = os.getenv("USE_SSH_AGENT", "false").lower() == "true" or self.use_agent
        ssh_config_path = Path(os.getenv("SSH_CONFIG_PATH", "~/.ssh/config")).expanduser()
        base_args = {"timeout": self.timeout}
        if use_agent:
            agent = Agent()
            keys = agent.get_keys()
            if not keys:
                logger.warning("No keys found in agent")
                return {}
            return {"look_for_keys": True, **base_args}

        config = paramiko.SSHConfig.from_path(ssh_config_path).lookup(self.host)
        if not config:
            raise SSHException(f"Host not found in SSH config {self.host}")
        key_file = config["identityfile"].pop()
        return {"key_filename": key_file, "look_for_keys": False, **base_args}  # type: ignore

    def get_client(self) -> paramiko.SSHClient:
        # TODO: May need to add better exception handling here
        try:
            ssh = initialize_ssh_client()
            connection_args = self.get_connection_args()
            ssh.connect(
                self.host,
                port=self.port,
                username=self.uname,
                **connection_args,
            )
        except Exception as e:
            raise
        return ssh

    def transfer_host_to_remote(
        self, host_path: Path, remote_path: Path, exclude_patterns=None, stream_stdout: bool = False
    ):
        exclude_patterns = exclude_patterns or []
        client = self.get_client()
        with client:
            self._run_cmd_no_stream(f"mkdir -P {remote_path}", client)
            with client.open_sftp() as sftp:
                finished_files = 0  # Initialize finished files counter

                # Recursively handle files and folders in the current directory
                def upload_directory_contents(remote_dir, local_dir):
                    # Ensure the local directory exists
                    self._run_cmd_no_stream(f"mkdir -P {remote_dir}", client)

                    # Get files and folders in the remote directory
                    files, folders = walk_local_directory(local_dir)
                    total_files = len(files)
                    progress_bar = tqdm(total=total_files, unit="files")

                    # Function to download a file with progress reporting
                    def upload_file(remote_file_path, local_file_path, index):
                        nonlocal finished_files
                        # Download file with progress callback
                        logger.info(f"Uploading {local_file_path}")
                        total_bytes = calculate_local_folder_size(local_file_path)
                        upload_file_with_progress(sftp, remote_file_path, local_file_path, total_bytes)
                        logger.info(f"Finished uploading {local_file_path}")
                        finished_files += 1
                        progress_bar.update(finished_files)

                    # Download all files in the current directory
                    for index, file in enumerate(files, start=1):
                        remote_file_path = f"{remote_dir}/{file}"
                        local_file_path = Path(local_dir, file)

                        # Skip files that match any exclusion pattern
                        if is_excluded(local_file_path, exclude_patterns):
                            logger.info(f"Skipping {local_file_path} (excluded by pattern)")
                            continue

                        upload_file(remote_file_path, local_file_path, index)

                    # Recursively handle subdirectories
                    for folder in folders:
                        remote_subdir = f"{remote_dir}/{folder}"
                        local_subdir = local_dir / folder
                        if is_excluded(local_subdir, exclude_patterns):
                            logger.info(f"Skipping directory {local_subdir} (excluded by pattern)")
                            continue
                        upload_directory_contents(remote_subdir, local_subdir)

                # Start downloading from the root folder
                upload_directory_contents(remote_path, host_path)
                logger.info("All files uploaded. Final progress emitted.")

    def transfer_remote_to_host(
        self, host_path: Path, remote_path: Path, exclude_patterns=None, stream_stdout: bool = False
    ):
        exclude_patterns = exclude_patterns or []
        client = self.get_client()
        with client:
            with client.open_sftp() as sftp:
                host_path.mkdir(parents=True, exist_ok=True)
                finished_files = 0  # Initialize finished files counter

                # Recursively handle files and folders in the current directory
                def download_directory_contents(remote_dir, local_dir):
                    # Ensure the local directory exists
                    local_dir.mkdir(parents=True, exist_ok=True)

                    # Get files and folders in the remote directory
                    files, folders = walk_sftp_directory(sftp, remote_dir)
                    total_files = len(files)
                    progress_bar = tqdm(total=total_files, unit="files")

                    # Function to download a file with progress reporting
                    def download_file(remote_file_path, local_file_path, index):
                        nonlocal finished_files
                        # Download file with progress callback
                        logger.info(f"Downloading {remote_file_path}")
                        total_bytes = calculate_remote_folder_size(client, remote_file_path)
                        download_file_with_progress(sftp, remote_file_path, local_file_path, total_bytes)
                        logger.info(f"Finished downloading {remote_file_path}")
                        finished_files += 1
                        progress_bar.update(finished_files)

                    # Download all files in the current directory
                    for index, file in enumerate(files, start=1):
                        remote_file_path = f"{remote_dir}/{file}"
                        local_file_path = Path(local_dir, file)

                        # Skip files that match any exclusion pattern
                        if is_excluded(remote_file_path, exclude_patterns):
                            logger.info(f"Skipping {remote_file_path} (excluded by pattern)")
                            continue

                        download_file(remote_file_path, local_file_path, index)

                    # Recursively handle subdirectories
                    for folder in folders:
                        remote_subdir = f"{remote_dir}/{folder}"
                        local_subdir = local_dir / folder
                        if is_excluded(remote_subdir, exclude_patterns):
                            logger.info(f"Skipping directory {remote_subdir} (excluded by pattern)")
                            continue
                        download_directory_contents(remote_subdir, local_subdir)

                # Start downloading from the root folder
                download_directory_contents(remote_path, host_path)
                logger.info("All files downloaded. Final progress emitted.")

    def copy_to_remote(self, local_path: Path, remote_path: Path, permissions: int, dry_run: bool = False):
        client = self.get_client()
        logger.info(f"Copying local file {local_path} to remote path: {remote_path}")
        with client:
            with client.open_sftp() as sftp:
                sftp.put(str(local_path), str(remote_path))
                sftp.chmod(str(remote_path), permissions)
        logger.info(f"Finished copying local file {local_path} to remote path: {remote_path}")


    def copy_from_remote(self, remote_path: Path, local_path: Path, permissions: int, dry_run: bool = False):
        client = self.get_client()
        logger.info(f"Copying remote file {remote_path} to local path: {local_path}")
        with client:
            with client.open_sftp() as sftp:
                sftp.chmod(str(remote_path), permissions)
                sftp.get(str(remote_path), str(local_path))
        logger.info(f"Finished copying remote file {remote_path} to local path: {local_path}")


    def _run_cmd(self, cmd: str, client: paramiko.SSHClient, dry_run: bool = False) -> tuple[str, str, int]:
        if dry_run:
            logger.info(f"Running dry-run command {cmd}")
            return
        logger.info(f"Running remote command {cmd}")
        _, stdout, stderr = client.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        return (
            stdout.read().decode("utf-8").strip(),
            stderr.read().decode("utf-8").strip(),
            exit_status,
        )

    def run_cmd(self, cmd: Union[str, List[str]],  dry_run: bool = False):
        cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
        if dry_run:
            logger.info(f"Dry running remote command: {cmd}")
            return
        client = self.get_client()
        with client:
            stdout, stderr, exit_code = self._run_cmd(cmd, client, dry_run=dry_run)
            msg = f"{stdout}\n\n{stderr}"
            if exit_code != 0:
                raise RuntimeError(f"Unable to run command {cmd}, exit_code={exit_code}:\n{msg}")
            logger.info(f"{msg}")
        return {"exit_code": exit_code, "msg": msg}

    def file_exists(self, fpath: str, dry_run: bool = False) -> bool:
        test_cmd = f"test -e {fpath} && echo exists"
        if dry_run:
            logger.info(f"Dry running remote command: {cmd}")
            return
        client = self.get_client()
        with client:
            stdout, stderr, exit_code = self._run_cmd(test_cmd, client, dry_run=dry_run)
            msg = f"{stdout}\n\n{stderr}"
            logger.info(f"{msg}")
        return exit_code == 0

    def check_dir_exists(self, remote_dir: Path, fail: bool = True):
        assert isinstance(remote_dir, Path)
        test_cmd = f"test -e {remote_dir} && echo exists"
        mkdir_cmd = f"mkdir -p {remote_dir}"
        client = self.get_client()
        with client:
            stdout, stderr, exit_code = self._run_cmd(test_cmd, client)
            if exit_code != 0 and not fail:
                self._run_cmd(mkdir_cmd, client)
            
        if exit_code != 0 and fail:
            raise RuntimeError(f"Remote directory {remote_dir} does not exist!")



