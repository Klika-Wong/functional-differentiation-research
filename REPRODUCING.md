# 复现说明

原运行环境：Windows、Python 3.12.14、NumPy 2.4.6、PyTorch 2.13.0+cpu；这些是记录值，不表示已验证所有平台。请安装适合自己平台的NumPy与PyTorch。本发布包只做静态与内容检查，未在清洁环境重新执行训练。

在仓库根目录运行，限制OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。脚本内部使用单线程和确定性Torch设置：

```text
python 03_实验代码/pilot44_fixed.py run
python 03_实验代码/pilot44_fixed.py audit
python accept_p0.py
python 03_实验代码/pilot44_learning.py run
python 03_实验代码/pilot44_learning.py audit
```

P0是零训练构造校准；仅P0科学通过且审计通过才能生成接受记录并运行P1。P1按固定预算完成主网与冻结递归对照。脚本沿历史候选标识写入work/pilot44-fixed-20261110；目录存在拒绝覆盖。日期是候选标识，真实执行时间记录在产物中。

运行会产生数据、权重及预测，须保留足够磁盘空间，按协议CPU/磁盘上限停止。不要删除失败输出后悄悄重试至成功。P2尚无发布实现，不能从本包宣称已做因果干预。协议摘录保留原登记与执行追加的时间顺序，历史NOT_RUN以之后明确追加的结果状态为准。
