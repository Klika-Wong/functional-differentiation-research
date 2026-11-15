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

运行会产生数据、权重及预测，须保留足够磁盘空间，按协议CPU/磁盘上限停止。不要删除失败输出后悄悄重试至成功。本次增加P2/P2B参考实现和历史证据，但未完成从清洁环境贯通全流程的复现验证。协议摘录保留原登记与执行追加的时间顺序，历史NOT_RUN以之后明确追加的结果状态为准。

## P2/P2B实现与复现限制（2026-11-15）

本次发布是源码与证据更新，没有重新训练或重跑历史确认。新增脚本保留原实验逻辑；仅做语法、依赖文件与证据绑定检查，不称一键可复现。

处理顺序为：P1完成及独立接受 → pilot45发现 → discovery审计 → lock → confirm → confirmation审计 → accept；然后pilot46 run → audit →独立接受；最后pilot47 run → audit →独立接受。pilot45对应的审计入口在pilot45_joint_audit.py，其他两者各自提供run/audit入口。禁止直接运行后续阶段跳过依赖。

具体缺口：原流程中P1的p1-acceptance.json和pilot46的acceptance.json由Codex审核后另行写入，本包未提供自动审核工具。需要复现者先复核自己新运行的完整数据、权重、产物清单及result/audit哈希，再生成自己的接受记录；不可复制公开历史接受记录去放行新运行，更不能仅把状态字符串改成通过。P1接受要求ACCEPTED_P1_LEARNING_PASS及result_sha256/audit_sha256绑定；校准接受要求ACCEPTED_PAIRED_CALIBRATION_PASS及同样的双哈希绑定。这些字段只是脚本接口，不替代协议规定的科学与工程验收。

`evidence/`是历史证据目录，不是运行依赖目录。P2/P2B脚本从固定的work/pilot44-*、work/pilot45-*、work/pilot46-*路径读取重新生成的完整产物；本包省略权重和原始轨迹，不能仅凭下载本包就直接执行历史audit。新生成的数据与快照须完整保留，源文件在依赖锁定后不得随意改动。

公开协议摘录与内部历史全协议不是同一字节文件。新复现应从当前公开源码/协议开始建立自己的快照和哈希链；不得用它替换历史封存文件。各阶段CPU/磁盘停止规则见协议16.66–16.69，不提高预算或删除失败输出来凑通过。
