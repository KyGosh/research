import os
import sys
sys.path.append(os.path.join('d:\\', 'Project', 'Research'))
from scripts.ml.datasets import list_name_dirs, list_single_for_name, list_pairs_for_name
'''
确认root目录下文件夹
查看键盘数据和鼠标数据的个数
'''
def main():
    root = os.path.join('d:\\', 'Project', 'Research', 'processed_data')
    nd = list_name_dirs(root)
    print('names', len(nd))
    names = sorted(nd.keys())
    for n in names[:5]:
        kb = list_single_for_name(root, n, 'keyboard')
        ms = list_single_for_name(root, n, 'mouse')
        pairs = list_pairs_for_name(root, n)
        print(n, 'kb', len(kb), 'ms', len(ms), 'pairs', len(pairs))

if __name__ == '__main__':
    main()
