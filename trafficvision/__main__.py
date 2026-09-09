import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description='Learn vehicle detection: prepare, train, evaluate, predict')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Convert the original DETRAC folder into a separate YOLO dataset')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--stride', type=int, default=1, help='1 uses every frame; larger values create a documented subset')
    p.add_argument('--limit', type=int, default=0, help='Maximum frames per sequence; 0 means all')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse verified prepared images with matching inputs/settings')
    p = sub.add_parser('train')
    p.add_argument('--config', default='configs/train.yaml')
    p.add_argument('--data')
    p.add_argument('--weights')
    p.add_argument('--epochs', type=int)
    p.add_argument('--device')
    p.add_argument('--batch', type=int)
    p = sub.add_parser('resume')
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('evaluate')
    p.add_argument('--weights', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--output', default='runs/evaluation')
    p.add_argument('--device', default='auto')
    p = sub.add_parser('compare', help='Conditional car/bus precision-recall comparison, same images and thresholds')
    p.add_argument('--baseline', required=True)
    p.add_argument('--trained', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('predict')
    p.add_argument('--image', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--device', default='auto')
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        if command == 'prepare':
            from .data import prepare
            result = prepare(**args)
        elif command == 'compare':
            from .comparison import compare
            result = compare(**args)
        else:
            from . import model
            if command == 'train':
                args['config_path'] = args.pop('config')
            result = getattr(model, command)(**args)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()  # Windows needs this guard when a training library starts worker processes.
