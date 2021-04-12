import os

# 需要的文件的路径
path = r'datas\tf_records\meta_data_path_ann'

# 以读模式打开该文件，这里只是为了说明该文件存在
# file_open = open(path, mode='r')

# 打印绝对路径
print(os.path.abspath(path))