Update: 2026/3/3

调整时间窗大小 从30s(1920tick) ==> 20s(1280tick) 情况没有改善
样本量不充分？ authentication任务：90pos + 90neg 严重的过拟合

有以下解决方案：

把每个 1280 tick 序列，提取：
* forward_ratio
* backward_ratio
* left_ratio
* right_ratio
* movement_switch_count
* 平均连续 forward 长度
* 最大连续 forward 长度
用 sklearn 训练一个 Logistic。
切换模型，数据量不足以支持LSTM模型的训练

如果继续保持LSTM模型：
1. 增加样本量，滑动窗口+多场比赛
2. 调整数据集，LOMO(Leave one match out)
3. 修改模型参数，hidden_dim = 64
4. 调整时间窗 20s --> 10s

数据集规模：
2026/2/9     3-1    IEM Kraków 2026
2025/11/13   1-2    BLAST Rivals 2025 Season 2
2025/11/09   0-3    IEM Chengdu 2025

共10张地图
采用LOMO方式，重复10轮
8:1:1 = train:valid:test

先选定地图类型(train/valid/test)
在不同类型内部，按地图处理数据，滑动窗口，640tick
pos样本数=neg样本数

事先处理数据==>
--processed_data
    --map1
        --apEX
            --keyboard
            --mouse
        --...
    --map2
    ...
    --map10

实现思路：
🔁 外层循环（10轮）
for k in 1..10:
    Test map = map_k
    Valid map = map_(k+1)  (mod 10)
    Train maps = 剩余8张

** Train阶段
pos = apEX 在 8个 Train maps 全部数据
neg_pool = 其他9名玩家 在 8个 Train maps 的全部数据
从每名neg玩家中采样：
    每人 = len(pos) / 9，样本随机从这8张地图中选取
合并后：
neg_total = len(pos)
（负样本处理需要考虑：下采样/修改正样本权重/引入“硬负采样”/etc.）

** Valid阶段
pos = apEX 在 Valid map 的全部数据
neg = 其他9名玩家 在 Valid map 的全部数据
保持真实比例（可能1:9）。
用AUC/EER表现结果，同时给出acc便于直观观察结果

** Test阶段
pos = apEX 在 Test map 的全部数据
neg = 其他9名玩家 在 Test map 的全部数据

实验分为四种情况：
1. 只有键盘
2. 只有鼠标
3. 键盘+鼠标
4. fusion model


Update: 03/06

处理鼠标数据：
pitch 俯仰角度  -90 ~ 90     -90 ~ 0  抬头 |  0 ~ 90  低头
yaw   水平转向  -180 ~ 180   -180 ~ 0 右转 |  0 ~ 180 左转

1. 直接使用原始数据 -- 不太可行
2. 通过原始数据，确定偏移角度，百分比？
3. 反向构造鼠标轨迹，使用1d-CNN进行学习，是否可以视为坐标

Update: 3.11

存在neg样本数小于pos的情况，需要排查原因
破案：部分样本存在规模小于预期tick数的情况被删除