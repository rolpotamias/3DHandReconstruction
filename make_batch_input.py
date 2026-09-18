"""Build a SageMaker batch-transform JSONL input for the 3DHandReconstruction model package.

Each output line is a complete, self-contained /invocations request:

    {"image": "<b64>", "mano_npz": "<b64>", "options": {...}}

Batch transform (SplitType=Line, BatchStrategy=SingleRecord) processes lines
independently with no ordering or instance-affinity guarantee, so a request
that needs 3D reconstruction must carry the MANO file in EVERY line (the
endpoint caches it by sha256, so repeated attachment costs payload bytes
only, ~4.7 MB/line base64). Detection-only jobs need no MANO at all.

    python make_batch_input.py --detect_only --out batch_input.jsonl samples/images/*.jpg
    python make_batch_input.py --mano_npz mano_right.npz --options '{"mesh": false}' \
        --out batch_input.jsonl my_photos/*.jpg
"""
import argparse
import base64
import json
from pathlib import Path


def build(images, out, mano_npz=None, options=None):
    """Write one JSONL line per image; returns the number of lines."""
    mano_b64 = None
    if mano_npz:
        mano_b64 = base64.b64encode(Path(mano_npz).read_bytes()).decode()
    with open(out, 'w') as f:
        for img in images:
            body = {'image': base64.b64encode(Path(img).read_bytes()).decode()}
            if mano_b64:
                body['mano_npz'] = mano_b64
            if options:
                body['options'] = options
            f.write(json.dumps(body) + '\n')
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mano_npz', default=None,
                        help='converted MANO file; required for 3D reconstruction lines')
    parser.add_argument('--options', default=None,
                        help='JSON options object copied into every request, '
                             'e.g. \'{"mesh": false, "conf": 0.3}\'')
    parser.add_argument('--detect_only', action='store_true',
                        help='detection-only job (boxes, handedness, 2D keypoints); no MANO needed')
    args = parser.parse_args()

    options = json.loads(args.options) if args.options else {}
    if args.detect_only:
        options['detect_only'] = True
    if not args.detect_only and not args.mano_npz:
        parser.error('3D reconstruction lines need --mano_npz (or pass --detect_only)')
    n = build(args.images, args.out, args.mano_npz, options or None)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f'{args.out}: {n} lines, {size_mb:.1f} MB (output lines align 1:1, same order)')


if __name__ == '__main__':
    main()
