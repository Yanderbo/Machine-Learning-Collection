import os

for root, dirs, files in os.walk("dataset"):
    dirs.sort()

    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list

    # #遍历文件
    for f in files:
        print(os.path.join(root, f))

    # 遍历所有的文件夹
    for d in dirs:
        print(os.path.join(root, d))

    print(len(files))
