#!/usr/bin/env python

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signwriting', required=True, type=str, help='path to input pose file')
    parser.add_argument('--pose', required=True, type=str, help='path to output pose file')
    return parser.parse_args()


def main():
    # pylint: disable=unused-variable
    args = get_args()


if __name__ == '__main__':
    main()
