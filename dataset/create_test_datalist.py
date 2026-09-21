import argparse
import os
from pathlib import Path


def resolve_path(path_value):
    return Path(os.path.expandvars(os.path.expanduser(str(path_value)))).resolve()


def make_datalist(data_dir, data_list, relative_to=None):
    data_dir = resolve_path(data_dir)
    data_list = resolve_path(data_list)
    path_root = resolve_path(relative_to) if relative_to else None

    data_paths = sorted(data_dir.glob('*.npz'))
    if not data_paths:
        raise FileNotFoundError(f'No .npz files found in {data_dir}')

    def format_path(path):
        return os.path.relpath(path, path_root) if path_root else str(path)

    data_list.parent.mkdir(parents=True, exist_ok=True)
    data_list.write_text('\n'.join(format_path(path) for path in data_paths) + '\n', encoding='utf-8')


def get_arguments():
    parser = argparse.ArgumentParser(description='Create a test list from .npz files.')
    parser.add_argument('--data-dir', required=True, help='Directory containing test .npz files.')
    parser.add_argument('--data-list', required=True, help='Output list path.')
    parser.add_argument('--relative-to', default=None,
                        help='Optional base directory used to write portable relative paths.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    make_datalist(args.data_dir, args.data_list, args.relative_to)
