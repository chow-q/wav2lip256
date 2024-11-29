# wav2lip256
1080p高清模型，直接推理，无需高清修复

wav2lip-onnx模式：

这是我修改过的最低配置要求 wav2lip 版本。

不需要GPU也可以推理。

使用转换后的 wav2lip onnx 模型，推理在 CPU 上运行速度相当快。速度在1:3左右。

用 Nvidia GPU 推理更快，速度在1:0.8左右。

跑onnx之前需先对要处理的视频先执行python generate_avatar.py生成视频avatar缓存文件。

增加虚拟摄像头：如果有新的视频文件：首先播放新的视频文件。如果没有新的视频文件：播放默认的 test.mp4。实现循环播放视频。

推理效果：

https://github.com/user-attachments/assets/3d6fe403-48dd-4589-aed1-49b12bf46cb7

you can contact me to get the model by:

email: 313733727@qq.com

Telegram: https://t.me/chow_dong

或者扫码添加微信获取模型：


![32651](https://github.com/user-attachments/assets/e3db3eb4-24a9-4226-876d-d07f6e0d519f)

