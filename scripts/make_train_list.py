from argparse import ArgumentParser
from pathlib import Path
import random

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    parser = ArgumentParser(description='Create a shuffled txt/list file for identity-stage training.')
    parser.add_argument('--image_dir', required=True, help='Directory containing training images.')
    parser.add_argument('--num', type=int, required=True, help='Number of images to sample.')
    parser.add_argument('--out', required=True, help='Output txt/list path.')
    parser.add_argument('--seed', type=int, default=231)
    args = parser.parse_args()

    root = Path(args.image_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    images = [p.resolve() for p in root.rglob('*') if p.suffix.lower() in EXTS and not p.name.startswith('._')]
    if not images:
        raise RuntimeError(f'No images found under {root}')
    random.seed(args.seed)
    random.shuffle(images)
    images = images[:min(args.num, len(images))]

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(str(p) for p in images) + '\n')
    print(f'wrote {len(images)} images to {out}')


if __name__ == '__main__':
    main()
