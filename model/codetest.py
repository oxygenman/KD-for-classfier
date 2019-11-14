import argparse
parser = argparse.ArgumentParser()
parser.add_argument('file_path', help='enter file path')
parser.add_argument('file_name', help='enter file name')
parser.add_argument('--read', help='read or write')
# args = parser.parse_args()
args = parser.parse_args(['test\image', 'file_path', '--read','true'])
print(args)
import array
a=[1,2,3]
b=a@a.reverse()
print(b)