---
name: aigc-drama-style-lock
description: AIGC 短剧"统一画风锁定包"通用 Skill。只要用户提到统一画风、风格锁定、角色一致性、首帧封面、视觉锚点、色板、镜头语法、电影感、韩剧感、参考风格、提示词风格块、Seedream 参考图、跨集画风漂移、服装漂移，就必须触发本 Skill。它会产出可复利的 style-lock：风格定位、色板与光线、镜头语法、角色视觉锚点、封面构图模板与大字文案原则，并给出可直接用于 Seedance/Seedream 的风格锁定块。
---

# Style Lock · 统一画风锁定包

> 短剧最痛的事：每集换一个 Seedance prompt，画风就漂一次。Style Lock 是把"全剧视觉语言"显性化的统一文档，所有 Gate D-G 的产出都必须套用它。
>
> 配套：[`aigc-drama-pipeline`](../aigc-drama-pipeline/SKILL.md) Gate D + Gate F；外部依赖 [`~/.cursor/skills/seedream-image/`](../../../../.cursor/skills/seedream-image/) 出参考图。

---

## 一、Style Lock 包的 6 个组成部分

每部新剧都要写完整一份 `_meta/style-lock.md`，含 6 节：

```
1. 风格定位        ← 一句话 + 气质关键词
2. 色板与光线      ← 常态 + 情绪切换
3. 镜头语法        ← 景别 / 运动 / 节奏
4. 角色视觉锚点    ← 每位主角 ≥5 项
5. 封面构图模板    ← ≥3 套
6. Prompt 风格锁定块 ← 正向 + 负面词，可直接复制
```

---

## 二、各节产出标准

### 2.1 风格定位

一句话 + 3-5 个气质关键词。

**示例（`$5B BOYFRIEND`）**：
```
冷峻韩剧式商业霸总剧 + 资本爽感 + 私密欲感张力
气质关键词：冷峻 / 清冷 / 奢华 / 隐忍 / 锐利
```

**通过标准**：
- 一句话 ≤25 字
- 气质关键词 3-5 个
- 关键词必须能直接映射到 Seedance prompt 的"氛围词"

### 2.2 色板与光线

**常态色板**（全剧默认）：
- 主色：石板灰 / 黑曜 / 暗木色（占画面 60%+）
- 副色：暖琥珀 / 冷蓝（窗光、台灯、屏幕反射）
- 点缀色：金属铜（戒指、纽扣、装饰）

**情绪切换色板**：
| 情绪 | 主光 | 辅光 | 色调 |
|------|------|------|------|
| 冲突/对峙 | 冷蓝顶光 | 暖橙边光 | 高对比 |
| 暧昧/亲密 | 琥珀暖光 | 镜面反射 | 低饱和 |
| 危机/崩溃 | 红色应急光 | 黑色背景 | 高反差 |
| 圆满/收官 | 金色逆光 | 柔光 | 暖调饱和 |

### 2.3 镜头语法

| 维度 | 选择 |
|------|------|
| 主要景别 | MCU 中近景 + 特写并用 |
| 镜头运动 | 缓慢推进 / 跟拍 / 固定（不用手持）|
| 切换节奏 | 3 秒一个 panel，节拍按 BGM 鼓点 |
| 比例 | 9:16 竖屏 + 内画面 2.39:1 宽幅信箱 |
| 帧率 | 24fps 电影感 |

### 2.4 角色视觉锚点

每位主角 ≥ 5 项，按 [`aigc-drama-pipeline/references/storyboard-industrialization.md`](../aigc-drama-pipeline/references/storyboard-industrialization.md) A.1 写入 A1-characters.json。

**示例（CHAR01 = Kace）**：
1. 32 岁
2. 灰蓝色眼睛
3. 黑色短发往后梳
4. 1.88m 健身体型
5. 左眉尾有一道浅疤
6. 公众形象：冷峻商业霸总
7. 私下：占有欲极强的偏执隐粉

### 2.5 封面构图模板（≥3 套）

| 模板 | 构图 | 用途 |
|------|------|------|
| **A 双人对峙** | 两人左右半身，中间留白文字位 | 冲突/反转集 |
| **B 单人特写** | 一人特写偏左，右侧大字标题 | 金句集 |
| **C 暧昧距离** | 两人侧脸近距离，留唇间气息感 | 欲感集 |

**大字文案原则**：
- 字号占封面高度 1/4 以上
- 颜色：暖白 / 烫金 / 冷蓝（与色板呼应）
- 字数 ≤8（金句优先），不写剧情摘要

### 2.6 Prompt 风格锁定块（最重要，可直接复制）

每段 Seedance 提示词的 "总述" 部分套这一段：

```
【风格锁定：电影 2.39:1 宽幅信箱构图（竖屏 9:16 内显示），24fps 电影感，胶片颗粒感，4K 高清；
主色 [石板灰+暗木色]，辅色 [暖琥珀+冷蓝]，点缀 [金属铜]；
光线 [侧顶光为主，强调下颌轮廓，眼神高光突出]；
氛围 [冷峻 / 隐忍 / 奢华]。】
```

**负面词**（追加到末尾或另起一行）：
```
负面：不要卡通画风、不要油画质感、不要过曝、不要塑料感皮肤、不要异形手指、不要字幕、不要水印、不要 logo
```

---

## 三、跨集一致性 SOP

### 3.1 角色一致性（避免每集脸漂）

- A1-characters.json 的 `ref_image` 必须固定（推荐 Seedream 出 1-2 张参考图，全剧用同一张）
- Seedance prompt 每段都用 `@图片N` 引用，**不允许文字描述脸**
- 服装锁定按 [`seedance-prompt-rules.md`](../aigc-drama-pipeline/references/seedance-prompt-rules.md) E 节写法

### 3.2 场景一致性

- A2-locations.json 的 `lighting_default` 全剧固定
- 同一场景不同集，光线方向/色板必须一致
- 若需要情绪切换，按 2.2 的"情绪切换色板"明确标注

### 3.3 风格漂的紧急处理

如果某集生成出来"画风不对"：
1. 先检查 prompt 是否套了 2.6 的风格锁定块
2. 检查是否有"额外的风格描述词"覆盖了锁定块（例如某段写了"水彩风"）
3. 检查 `@图片N` 是否齐全
4. 三项都没问题但仍漂 → 把当前 prompt 加进 `_meta/risk-register.md`，更新 [`safety-wordlist.md`](../aigc-drama-pipeline/references/safety-wordlist.md) C 节

---

## 四、新剧启动 Style Lock 的 4 步

1. **定调**（半天）：与用户确认风格定位 + 气质关键词
2. **出参考图**（半天）：用 `~/.cursor/skills/seedream-image/` 出 2 张主角参考图 + 1 张代表性场景图，写入 A1/A2
3. **写完整 6 节**（半天）：按上述 2.1-2.6 写 `_meta/style-lock.md`
4. **首集试拍验收**（1 天）：EP01 出来后，对比 style-lock，发现漂移立即修

---

## 五、与其他 Skill 的关系

- 本 Skill = "全剧视觉语言定义"
- [`aigc-drama-pipeline`](../aigc-drama-pipeline/SKILL.md) Gate D = 用本 Skill 出角色包
- [`aigc-drama-pipeline`](../aigc-drama-pipeline/SKILL.md) Gate F = 用本 Skill 出每集首帧
- [`aigc-drama-pipeline/references/seedance-prompt-rules.md`](../aigc-drama-pipeline/references/seedance-prompt-rules.md) D.2 = Gate G 套用本 Skill 2.6 的风格锁定块
- `~/.cursor/skills/seedream-image/` = 出参考图

---

## 六、参考

完整版项目化 style-lock 写入：`aigc-video/<series>/_meta/style-lock.md`，含本剧定制的：
- 实际色值（HEX）
- 实际服装锁定列表
- 实际参考图文件路径
