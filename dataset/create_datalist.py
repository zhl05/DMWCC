import argparse
import os
from pathlib import Path


def resolve_path(path_value):
    return Path(os.path.expandvars(os.path.expanduser(str(path_value)))).resolve()


def make_datalist(data_dir, image_list, label_list, label_dir, relative_to=None):
    data_dir = resolve_path(data_dir)
    label_dir = resolve_path(label_dir)
    image_list = resolve_path(image_list)
    label_list = resolve_path(label_list)
    path_root = resolve_path(relative_to) if relative_to else None

    image_paths = sorted(data_dir.glob('*.npy'))
    if not image_paths:
        raise FileNotFoundError(f'No .npy files found in {data_dir}')

    label_paths = [label_dir / f'{path.stem}_gt.npy' for path in image_paths]
    missing = [str(path) for path in label_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} label files; first missing file: {missing[0]}')

    def format_path(path):
        return os.path.relpath(path, path_root) if path_root else str(path)

    image_list.parent.mkdir(parents=True, exist_ok=True)
    label_list.parent.mkdir(parents=True, exist_ok=True)
    image_list.write_text('\n'.join(format_path(path) for path in image_paths) + '\n', encoding='utf-8')
    label_list.write_text('\n'.join(format_path(path) for path in label_paths) + '\n', encoding='utf-8')


def get_arguments():
    parser = argparse.ArgumentParser(description='Create paired .npy image and label lists.')
    parser.add_argument('--data-dir', required=True, help='Directory containing image .npy files.')
    parser.add_argument('--label-dir', required=True, help='Directory containing <stem>_gt.npy labels.')
    parser.add_argument('--image-list', required=True, help='Output image-list path.')
    parser.add_argument('--label-list', required=True, help='Output label-list path.')
    parser.add_argument('--relative-to', default=None,
                        help='Optional base directory used to write portable relative paths.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    make_datalist(args.data_dir, args.image_list, args.label_list, args.label_dir, args.relative_to)
