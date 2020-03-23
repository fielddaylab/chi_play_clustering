import os,sys
def init_path():
    os.chdir('..')
    print('CDW:', os.getcwd())
    sys.path.append('.')
