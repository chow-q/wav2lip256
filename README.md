# wav2lip256
1080p高清模型，直接推理，无需高清修复

wav2lip-onnx模式：

这是我修改过的最低配置要求 wav2lip 版本。

不需要GPU也可以推理。

使用转换后的 wav2lip onnx 模型，推理在 CPU 上运行速度相当快。速度在1:3左右。

用 Nvidia GPU 推理更快，速度在1:0.8左右。

跑onnx之前需先对要处理的视频先执行python generate_avatar.py生成视频avatar缓存文件。

推理效果：

https://github.com/user-attachments/assets/3d6fe403-48dd-4589-aed1-49b12bf46cb7

扫码添加微信获取模型：


![6102d342f57aa5bfe5f90478dd12ded](https://github.com/user-attachments/assets/302f43f8-834d-4a52-8993-7afdcd297b3e)
