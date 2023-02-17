import sys
import yaml
import argparse


def parse_arguments() -> argparse.Namespace:
    '''
    Read the values in config.yaml
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', 
        type=argparse.FileType(mode='r'),
        default='config.yaml', 
        help='The config file to use. Default: Must be placed in the root folder.',
    )

    args = parser.parse_args()
    arg_dict = vars(args)
    if args.config:
        arg_dict.update(yaml.load(args.config, Loader=yaml.FullLoader))

    return args


def main():
    args = parse_arguments()
    print('Es muss sein.')


if __name__ == '__main__':
    sys.exit(main())
