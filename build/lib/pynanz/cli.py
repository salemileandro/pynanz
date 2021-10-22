#!/usr/bin/env python3
"""
Command line interface for the pynanz python library
"""

import pynanz
import argparse
from datetime import date
import sys
import pandas as pd


def main(command=None):
    parser = argparse.ArgumentParser(description='Pynanz library.')
    subparsers = parser.add_subparsers(help='sub-command help')

    download_sp = subparsers.add_parser("download")
    download_sp.add_argument('--from', help='Starting date "YYYY-MM-DD"', type=pd.Timestamp)
    download_sp.add_argument('--to', help='Ending date "YYYY-MM-DD"', type=pd.Timestamp)

    upload_sp = subparsers.add_parser("upload")
    upload_sp.add_argument('--from', help='Starting date "YYYY-MM-DD"', type=pd.Timestamp)
    upload_sp.add_argument('--to', help='Ending date "YYYY-MM-DD"', type=pd.Timestamp)

    args = parser.parse_args(command)

    print(args, type(args))

    for k, v in vars(args).items():
        print(k, v)

if __name__ == "__main__":
    main(sys.argv)
