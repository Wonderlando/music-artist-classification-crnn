import os

def list_dir_test():
    return os.listdir('./artists')

if __name__ == '__main__':
    print(list_dir_test())