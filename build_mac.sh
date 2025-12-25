#!/bin/bash
# Mac上打包命令
# 使用方法：在终端运行 ./build_mac.sh

pyinstaller --onefile --noconsole \
    --add-data "image:image" \
    --name "ChristmasTree" \
    main.py \
    --clean

echo "打包完成！可执行文件在 dist 文件夹中"

