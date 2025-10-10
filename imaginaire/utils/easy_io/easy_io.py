# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import warnings
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import IO, Any

from imaginaire.utils.easy_io.backends import backends, prefix_to_backends
from imaginaire.utils.easy_io.file_client import FileClient
from imaginaire.utils.easy_io.handlers import file_handlers

backend_instances: dict = {}


def is_filepath(filepath):
    return isinstance(filepath, (str, Path))


def _parse_uri_prefix(uri: str | Path) -> str:
    """Parse the prefix of uri.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.

    Examples:
        >>> _parse_uri_prefix('/home/path/of/your/file')
        ''
        >>> _parse_uri_prefix('http://path/of/your/file')
        'http'

    Returns:
        str: Return the prefix of uri if the uri contains '://'. Otherwise,
        return ''.
    """
    assert is_filepath(uri)
    uri = str(uri)
    # if uri does not contains '://', the uri will be handled by
    # LocalBackend by default
    if "://" not in uri:
        return ""
    else:
        prefix, _ = uri.split("://")
        if ":" in prefix:
            _, prefix = prefix.split(":")
        return prefix


def _get_file_backend(prefix: str, backend_args: dict):
    """Return a file backend based on the prefix or backend_args.

    Args:
        prefix (str): Prefix of uri.
        backend_args (dict): Arguments to instantiate the corresponding
            backend.
    """
    # backend name has a higher priority
    if "backend" in backend_args:
        # backend_args should not be modified
        backend_args_bak = backend_args.copy()
        backend_name = backend_args_bak.pop("backend")
        backend = backends[backend_name](**backend_args_bak)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend


def get_file_backend(
    uri: str | Path | None = None,
    *,
    backend_args: dict | None = None,
    enable_singleton: bool = False,
    backend_key: str | None = None,
):
    """Return a file backend based on the prefix of uri or backend_args.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        enable_singleton (bool): Whether to enable the singleton pattern.
            If it is True, the backend created will be reused if the
            signature is same with the previous one. Defaults to False.
        backend_key: str: The key to register the backend. Defaults to None.

    Returns:
        BaseStorageBackend: Instantiated Backend object.

    Examples:
        >>> # get file backend based on the prefix of uri
        >>> uri = 'http://path/of/your/file'
        >>> backend = get_file_backend(uri)
        >>> # get file backend based on the backend_args
        >>> backend = get_file_backend(backend_args={'backend': 'http'})
        >>> # backend name has a higher priority if 'backend' in backend_args
        >>> backend = get_file_backend(uri, backend_args={'backend': 'http'})
    """
    global backend_instances
    if backend_key is not None:
        if backend_key in backend_instances:
            return backend_instances[backend_key]

    if backend_args is None:
        backend_args = {}

    if uri is None and "backend" not in backend_args and backend_key is None:
        raise ValueError('uri should not be None when "backend" does not exist in backend_args and backend_key is None')

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ""

    if enable_singleton:
        unique_key = f"{prefix}:{json.dumps(backend_args)}"
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        if backend_key is not None:
            backend_instances[backend_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend


def get(
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> bytes:
    """Read bytes from a given ``filepath`` with 'rb' mode.

    Args:
        filepath (str or Path): Path to read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        bytes: Expected bytes object.

    Examples:
        >>> filepath = '/path/of/file'
        >>> get(filepath)
        b'hello world'
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    return backend.get(filepath)


def get_text(
    filepath: str | Path,
    encoding="utf-8",
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str:
    """Read text from a given ``filepath`` with 'r' mode.

    Args:
        filepath (str or Path): Path to read data.
        encoding (str): The encoding format used to open the ``filepath``.
            Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        str: Expected text reading from ``filepath``.

    Examples:
        >>> filepath = '/path/of/file'
        >>> get_text(filepath)
        'hello world'
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    return backend.get_text(filepath, encoding)


def put(
    obj: bytes,
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> None:
    """Write bytes to a given ``filepath`` with 'wb' mode.

    Note:
        ``put`` should create a directory if the directory of
        ``filepath`` does not exist.

    Args:
        obj (bytes): Data to be written.
        filepath (str or Path): Path to write data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Examples:
        >>> filepath = '/path/of/file'
        >>> put(b'hello world', filepath)
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    backend.put(obj, filepath)


def put_text(
    obj: str,
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> None:
    """Write text to a given ``filepath`` with 'w' mode.

    Note:
        ``put_text`` should create a directory if the directory of
        ``filepath`` does not exist.

    Args:
        obj (str): Data to be written.
        filepath (str or Path): Path to write data.
        encoding (str, optional): The encoding format used to open the
            ``filepath``. Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Examples:
        >>> filepath = '/path/of/file'
        >>> put_text('hello world', filepath)
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    backend.put_text(obj, filepath)


def exists(
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> bool:
    """Check whether a file path exists.

    Args:
        filepath (str or Path): Path to be checked whether exists.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

    Examples:
        >>> filepath = '/path/of/file'
        >>> exists(filepath)
        True
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    return backend.exists(filepath)


def isdir(
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> bool:
    """Check whether a file path is a directory.

    Args:
        filepath (str or Path): Path to be checked whether it is a
            directory.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        bool: Return ``True`` if ``filepath`` points to a directory,
        ``False`` otherwise.

    Examples:
        >>> filepath = '/path/of/dir'
        >>> isdir(filepath)
        True
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    return backend.isdir(filepath)


def isfile(
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> bool:
    """Check whether a file path is a file.

    Args:
        filepath (str or Path): Path to be checked whether it is a file.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        bool: Return ``True`` if ``filepath`` points to a file, ``False``
        otherwise.

    Examples:
        >>> filepath = '/path/of/file'
        >>> isfile(filepath)
        True
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    return backend.isfile(filepath)


def join_path(
    filepath: str | Path,
    *filepaths: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    r"""Concatenate all file paths.

    Join one or more filepath components intelligently. The return value
    is the concatenation of filepath and any members of \*filepaths.

    Args:
        filepath (str or Path): Path to be concatenated.
        *filepaths (str or Path): Other paths to be concatenated.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        str: The result of concatenation.

    Examples:
        >>> filepath1 = '/path/of/dir1'
        >>> filepath2 = 'dir2'
        >>> filepath3 = 'path/of/file'
        >>> join_path(filepath1, filepath2, filepath3)
        '/path/of/dir/dir2/path/of/file'
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    return backend.join_path(filepath, *filepaths)


@contextmanager
def get_local_path(
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> Generator[str | Path, None, None]:
    """Download data from ``filepath`` and write the data to local path.

    ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
    can be called with ``with`` statement, and when exists from the
    ``with`` statement, the temporary path will be released.

    Note:
        If the ``filepath`` is a local path, just return itself and it will
        not be released (removed).

    Args:
        filepath (str or Path): Path to be read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: Only yield one path.

    Examples:
        >>> with get_local_path('http://example.com/file.jpg') as path:
        ...     # do something here
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    with backend.get_local_path(str(filepath)) as local_path:
        yield local_path


def copyfile(
    src: str | Path,
    dst: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    """Copy a file src to dst and return the destination file.

    src and dst should have the same prefix. If dst specifies a directory,
    the file will be copied into dst using the base filename from src. If
    dst specifies a file that already exists, it will be replaced.

    Args:
        src (str or Path): A file to be copied.
        dst (str or Path): Copy file to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination file.

    Raises:
        SameFileError: If src and dst are the same file, a SameFileError will
            be raised.

    Examples:
        >>> # dst is a file
        >>> src = '/path/of/file'
        >>> dst = '/path1/of/file1'
        >>> # src will be copied to '/path1/of/file1'
        >>> copyfile(src, dst)
        '/path1/of/file1'

        >>> # dst is a directory
        >>> dst = '/path1/of/dir'
        >>> # src will be copied to '/path1/of/dir/file'
        >>> copyfile(src, dst)
        '/path1/of/dir/file'
    """
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copyfile(src, dst)


def copytree(
    src: str | Path,
    dst: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    """Recursively copy an entire directory tree rooted at src to a directory
    named dst and return the destination directory.

    src and dst should have the same prefix and dst must not already exist.

    Args:
        src (str or Path): A directory to be copied.
        dst (str or Path): Copy directory to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        backend_key (str, optional): The key to get the backend from register.

    Returns:
        str: The destination directory.

    Raises:
        FileExistsError: If dst had already existed, a FileExistsError will be
            raised.

    Examples:
        >>> src = '/path/of/dir1'
        >>> dst = '/path/of/dir2'
        >>> copytree(src, dst)
        '/path/of/dir2'
    """
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copytree(src, dst)


def copyfile_from_local(
    src: str | Path,
    dst: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    """Copy a local file src to dst and return the destination file.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copyfile`.

    Args:
        src (str or Path): A local file to be copied.
        dst (str or Path): Copy file to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: If dst specifies a directory, the file will be copied into dst
        using the base filename from src.

    Examples:
        >>> # dst is a file
        >>> src = '/path/of/file'
        >>> dst = 'http://example.com/file1'
        >>> # src will be copied to 'http://example.com/file1'
        >>> copyfile_from_local(src, dst)
        http://example.com/file1

        >>> # dst is a directory
        >>> dst = 'http://example.com/dir'
        >>> # src will be copied to 'http://example.com/dir/file''
        >>> copyfile_from_local(src, dst)
        'http://example.com/dir/file'
    """
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copyfile_from_local(src, dst)


def copytree_from_local(
    src: str | Path,
    dst: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    """Recursively copy an entire directory tree rooted at src to a directory
    named dst and return the destination directory.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copytree`.

    Args:
        src (str or Path): A local directory to be copied.
        dst (str or Path): Copy directory to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.

    Examples:
        >>> src = '/path/of/dir'
        >>> dst = 'http://example.com/dir'
        >>> copyfile_from_local(src, dst)
        'http://example.com/dir'
    """
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copytree_from_local(src, dst)


def copyfile_to_local(
    src: str | Path,
    dst: str | Path,
    dst_type: str,  # Choose from ["file", "dir"]
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    """Copy the file src to local dst and return the destination file.

    If dst specifies a directory, the file will be copied into dst using
    the base filename from src. If dst specifies a file that already
    exists, it will be replaced.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copyfile`.

    Args:
        src (str or Path): A file to be copied.
        dst (str or Path): Copy file to to local dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: If dst specifies a directory, the file will be copied into dst
        using the base filename from src.

    Examples:
        >>> # dst is a file
        >>> src = 'http://example.com/file'
        >>> dst = '/path/of/file'
        >>> # src will be copied to '/path/of/file'
        >>> copyfile_to_local(src, dst)
        '/path/of/file'

        >>> # dst is a directory
        >>> dst = '/path/of/dir'
        >>> # src will be copied to '/path/of/dir/file'
        >>> copyfile_to_local(src, dst)
        '/path/of/dir/file'
    """
    assert dst_type in ["file", "dir"]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copyfile_to_local(src, dst, dst_type=dst_type)


def copytree_to_local(
    src: str | Path,
    dst: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> str | Path:
    """Recursively copy an entire directory tree rooted at src to a local
    directory named dst and return the destination directory.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copytree`.

    Args:
        src (str or Path): A directory to be copied.
        dst (str or Path): Copy directory to local dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.

    Examples:
        >>> src = 'http://example.com/dir'
        >>> dst = '/path/of/dir'
        >>> copytree_to_local(src, dst)
        '/path/of/dir'
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copytree_to_local(src, dst)


def remove(
    filepath: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> None:
    """Remove a file.

    Args:
        filepath (str, Path): Path to be removed.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Raises:
        FileNotFoundError: If filepath does not exist, an FileNotFoundError
            will be raised.
        IsADirectoryError: If filepath is a directory, an IsADirectoryError
            will be raised.

    Examples:
        >>> filepath = '/path/of/file'
        >>> remove(filepath)
    """
    backend = get_file_backend(
        filepath,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    backend.remove(filepath)


def rmtree(
    dir_path: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> None:
    """Recursively delete a directory tree.

    Args:
        dir_path (str or Path): A directory to be removed.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Examples:
        >>> dir_path = '/path/of/dir'
        >>> rmtree(dir_path)
    """
    backend = get_file_backend(
        dir_path,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    backend.rmtree(dir_path)


def copy_if_symlink_fails(
    src: str | Path,
    dst: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> bool:
    """Create a symbolic link pointing to src named dst.

    If failed to create a symbolic link pointing to src, directory copy src to
    dst instead.

    Args:
        src (str or Path): Create a symbolic link pointing to src.
        dst (str or Path): Create a symbolic link named dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return True if successfully create a symbolic link pointing to
        src. Otherwise, return False.

    Examples:
        >>> src = '/path/of/file'
        >>> dst = '/path1/of/file1'
        >>> copy_if_symlink_fails(src, dst)
        True
        >>> src = '/path/of/dir'
        >>> dst = '/path1/of/dir1'
        >>> copy_if_symlink_fails(src, dst)
        True
    """
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True, backend_key=backend_key)
    return backend.copy_if_symlink_fails(src, dst)


def list_dir(
    dir_path: str | Path,
    backend_args: dict | None = None,
    backend_key: str | None = None,
):
    """List all folders in a directory with a given path.

    Args:
        dir_path (str | Path): Path of the directory.

    Examples:
        >>> dir_path = '/path/of/dir'
        >>> for file_path in list_dir(dir_path):
        ...     print(file_path)
    """
    if not dir_path.endswith("/"):
        dir_path += "/"
    backend = get_file_backend(
        dir_path,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )

    return backend.list_dir(dir_path)


def list_dir_or_file(
    dir_path: str | Path,
    list_dir: bool = True,
    list_file: bool = True,
    suffix: str | tuple[str] | None = None,
    recursive: bool = False,
    backend_args: dict | None = None,
    backend_key: str | None = None,
) -> Iterator[str]:
    """Scan a directory to find the interested directories or files in
    arbitrary order.

    Note:
        :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

    Args:
        dir_path (str or Path): Path of the directory.
        list_dir (bool): List the directories. Defaults to True.
        list_file (bool): List the path of files. Defaults to True.
        suffix (str or tuple[str], optional): File suffix that we are
            interested in. Defaults to None.
        recursive (bool): If set to True, recursively scan the directory.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: A relative path to ``dir_path``.

    Examples:
        >>> dir_path = '/path/of/dir'
        >>> for file_path in list_dir_or_file(dir_path):
        ...     print(file_path)
        >>> # list those files and directories in current directory
        >>> for file_path in list_dir_or_file(dir_path):
        ...     print(file_path)
        >>> # only list files
        >>> for file_path in list_dir_or_file(dir_path, list_dir=False):
        ...     print(file_path)
        >>> # only list directories
        >>> for file_path in list_dir_or_file(dir_path, list_file=False):
        ...     print(file_path)
        >>> # only list files ending with specified suffixes
        >>> for file_path in list_dir_or_file(dir_path, suffix='.txt'):
        ...     print(file_path)
        >>> # list all files and directory recursively
        >>> for file_path in list_dir_or_file(dir_path, recursive=True):
        ...     print(file_path)
    """
    backend = get_file_backend(
        dir_path,
        backend_args=backend_args,
        enable_singleton=True,
        backend_key=backend_key,
    )
    yield from backend.list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)


def load(
    file: str | Path | IO[Any],
    file_format: str | None = None,
    file_client_args: dict | None = None,
    fast_backend: bool = False,
    backend_args: dict | None = None,
    backend_key: str | None = None,
    **kwargs,
):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    ``load`` supports loading data from serialized files those can be storaged
    in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        fast_backend: bool: Whether to use multiprocess. Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split(".")[-1]
    # convert file_format to lower case
    file_format = file_format.lower()
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    if file_client_args is not None:
        warnings.warn(  # noqa: B028
            '"file_client_args" will be deprecated in future. Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError('"file_client_args and "backend_args" cannot be set at the same time.')

    handler = file_handlers[file_format]
    if isinstance(file, str):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(
                file,
                backend_args=backend_args,
                backend_key=backend_key,
                enable_singleton=True,
            )

        if handler.str_like:
            with StringIO(file_backend.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            if fast_backend:
                if hasattr(file_backend, "fast_get"):
                    with BytesIO(file_backend.fast_get(file)) as f:
                        obj = handler.load_from_fileobj(f, **kwargs)
                else:
                    warnings.warn(  # noqa: B028
                        f"fast_backend is not supported by the backend, type {type(file_backend)} fallback to normal get"
                    )
                    with BytesIO(file_backend.get(file)) as f:
                        obj = handler.load_from_fileobj(f, **kwargs)
            else:
                with BytesIO(file_backend.get(file)) as f:
                    obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(
    obj: Any,
    file: str | Path | IO[Any] | None = None,
    file_format: str | None = None,
    file_client_args: dict | None = None,
    fast_backend: bool = False,
    backend_args: dict | None = None,
    backend_key: str | None = None,
    **kwargs,
):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping data as strings or to files which is saved to
    different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        fast_backend: bool: Whether to use multiprocess. Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.
        backend_key: str: The key to register the backend. Defaults to None.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 'http://path/of/your/file')  # http

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    # convert file_format to lower case
    file_format = file_format.lower()
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    if file_client_args is not None:
        warnings.warn(  # noqa: B028
            '"file_client_args" will be deprecated in future. Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError('"file_client_args" and "backend_args" cannot be set at the same time.')

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(
                file,
                backend_args=backend_args,
                backend_key=backend_key,
                enable_singleton=True,
            )

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                if fast_backend:
                    if hasattr(file_backend, "fast_put"):
                        file_backend.fast_put(f, file)
                    else:
                        warnings.warn("fast_backend is not supported by the backend, fallback to normal put")  # noqa: B028
                        file_backend.put(f, file)
                else:
                    file_backend.put(f, file)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
