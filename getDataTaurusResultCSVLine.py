'''
Created on Mar 11, 2023

@author: Miguel

USE:
    Script to give the output file from Taurus and get an exportable line
    
    Linux: from one or several files
    
    $ python3 getDataTaurusResultCSVLine out1.OUT out2.OUT ...
    > exported_results.txt
        DataTaurus(out1.OUT) as csv line
        DataTaurus(out2.OUT) as csv line
        ...

'''

from sys import argv
from tools.data import DataTaurus

if __name__ == '__main__':
    
    files2print = argv[1:]
    assert len(argv) > 1, "[ERROR] Give at least one output file from Taurus"
    exportable = []
    for file_ in files2print:
        res = DataTaurus(0,0, file_)
        exportable.append(res.getAttributesDictLike)
    
    with open('exported_results.txt', 'w+') as f:
        f.writelines(exportable)
    
    print("[OK] Exporting of files finished")
    
