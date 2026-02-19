---
name: seedream-image
description: |
  Generate high-quality image prompts for Seedream 5.0 (即梦AI) by ByteDance.
  Covers text-to-image, image editing, multi-image fusion, character consistency,
  knowledge cards, posters, PPT backgrounds, e-commerce, avatars, and more.

  为即梦AI（Seedream 5.0）生成高质量图片提示词。支持文生图、图生图、多图融合、
  多轮编辑、知识卡片、海报、角色一致性、PPT背景、电商产品图等各类场景。

  Triggers: "seedream", "即梦", "jimeng", "seedream prompt", "seedream 提示词",
  "AI image prompt", "AI生图", "生成图片", "写提示词", "image generation",
  "text to image", "文生图", "图生图", "character consistency", "角色一致性",
  "knowledge card", "知识卡片", "poster design", "海报设计", "avatar",
  "头像生成", "e-commerce photo", "电商图", "PPT background", "PPT背景",
  "multi-image fusion", "多图融合", "AI art", "AI绘画", "doubao image",
  "豆包生图", "volcengine image", "火山方舟生图", "capcut AI image",
  "jianying AI", "剪映AI绘画", "seedream 5", "seedream 4.5"
license: MIT
metadata:
  author: AI Visuals
  version: "1.0.0"
  homepage: https://github.com/ppdbxdawj/seedream-image-skill
  tags:
    - ai-image-generation
    - prompt-engineering
    - seedream
    - bytedance
    - jimeng
    - text-to-image
    - image-editing
    - character-consistency
    - chinese-ai
    - creative-tools
---

# Seedream Image Prompt Assistant | Seedream 生图提示词助手

Seedream 5.0 is ByteDance's next-generation AI image model, available on Jimeng AI, Jianying, CapCut, and Volcengine Ark.

Seedream 5.0 是字节跳动推出的新一代 AI 图像生成模型，已在即梦AI、剪映、CapCut、火山方舟上线。

## Core Capabilities | 核心能力

| Capability | Description |
|------------|-------------|
| **Real-time Web Search** | Auto-fetches trending info when prompt contains timely keywords |
| **Multi-step Reasoning** | Interprets abstract concepts (e.g. "serene tech feel" → desaturated + clean lines + cold light) |
| **Multi-round Editing** | Iterative refinement: local edits, style transfer, element add/remove, text rendering |
| **High Resolution** | Native 2K, AI-enhanced 4K, 2-5 second generation |
| **Character Consistency** | Maintains face, clothing, pose across multiple images (storyboard-ready) |
| **Text Rendering** | 99%+ accuracy for Chinese/English text, use quotes for best results |

## 提示词结构

### 基础结构（文生图）

```
[主体描述] + [行为/动作] + [环境/背景] + [材质/质感] + [光影效果] + [构图要求] + [风格关键词]
```

- 主体+行为+环境用**自然语言**描述
- 风格/色彩/光影/构图用**短词**点缀
- 文字内容用引号标注，如：`"Hello World"`

### 四段式结构（进阶）

```
主体 → 环境 → 材质/质感 → 光影
```

### 编辑提示词公式

```
变化动作 + 变化对象 + 变化特征
示例："将骑士的头盔变为金色"
```

## 风格词汇库

### 写实摄影
- `写实电影剧照` `商业摄影` `纪实摄影` `超写实` `RAW 原片质感`
- 镜头：`85mm定焦` `35mm广角` `长焦压缩感` `鱼眼镜头`
- 光线：`伦勃朗光` `环形光` `分割光` `黄金时刻暖光` `蓝调时刻冷光` `霓虹光`

### 动漫/插画
- 日漫：`吉卜力动画风格` `新海诚风格` `日系少女漫画` `赛璐璐质感`
- 欧美：`美漫风格` `DC漫画风格` `欧美写实人物` `Pop Art波普艺术`
- 中国：`国潮插画` `水墨画风格` `中式工笔画` `赛博国风`
- 其他：`像素风格` `低多边形` `扁平插画` `厚涂油画` `水彩手绘`

### 设计/商业
- `极简主义` `包豪斯风格` `磨砂玻璃质感` `高质感金属` `赛博朋克`
- `电影海报级别` `品牌VI视觉` `信息图Infographic` `知识卡片`

### 光影修饰词
- `戏剧性侧光` `柔和漫射光` `高对比度` `低饱和度` `莫兰迪色调`
- `赛博霓虹` `暖橙调` `冷蓝调` `胶片颗粒感`

## 常用提示词模板

### 人物写实
```
[性别年龄外貌]，[服装描述]，[表情神态]，[环境背景]，85mm定焦，自然光，写实电影剧照风格，超高清，细节丰富
```

### 风景/场景
```
[场景描述]，[时间/天气]，[光线描述]，[构图]，[风格词]，电影感构图，8K超清
```

### 知识卡片（完整模板）
```
生成一张[格式/载体]风格的图像，向[目标受众]解释/展示"[核心概念]"。
图像需具备[风格特征A]、[风格特征B]和[排版要求C]，整体感觉类似于[熟悉参照物]。
```

### 品牌/海报（留白模板）
```
[视觉主体描述]，[材质描述]，[光影效果]，
所有视觉主体集中在画面[左/右]侧，为[右/左]侧留出大面积干净的背景区域，方便后期排版添加文字。
背景：[背景描述]
```

### 连续分镜（角色一致性）
```
参考[图1]的面部和发型，将其更改为[场景风格]装束，
生成N张连续的[场景描述]分镜图，[风格]，需要在一个场景中，连续动作。
```

### 电商产品
```
为这件[产品]创建[平台]风格的展示图，风格类似于[品牌参照]，
背景简洁，突出产品质感，专业商业摄影
```

## 场景速查

| 场景 | 提示词关键词 | 注意事项 |
|------|------------|---------|
| 头像 | `头像图标` `正方形构图` `纯色背景` | 指定风格参考图效果更好 |
| 知识卡片 | `信息图` `知识图谱` `排版清晰` | 说明目标受众和核心概念 |
| PPT背景 | `留白构图` `偏向[左/右]侧` `哑光背景` | 强调一侧留白供排版 |
| 角色Cos | `保持人脸不变` `写实质感服饰` `相同姿势` | 上传原图+目标角色图 |
| 手帐日记 | `手写字体` `纸张纹理` `拼贴风格` `米黄底色` | 告知日期和天气增加氛围 |
| 玻璃图标 | `磨砂玻璃质感` `渐变色` `C4D` `OC渲染` | 纯白背景+简洁构图 |
| 海报设计 | `电影海报级别` `戏剧光` `大面积留白` | 明确文字内容和位置 |
| 护身符/国潮 | `山海经` `国潮票据` `水墨` `篆刻印章` | 可加入"愿望"文字增加情感 |

## 进阶技巧

### 1. 联网触发
提示词中含时效词时自动联网：`2026年流行色` `最新款XX` `今年XX趋势` `米兰冬奥会`

### 2. 图像编辑
- **指定区域**："将图中[区域]替换成..."
- **风格迁移**："保持内容不变，改成[风格]"
- **元素控制**："为画面增加/移除[元素]"
- **光影调整**："将画面光影改为[光线名称]"
- **滤镜添加**："为画面添加[滤镜名]滤镜"
- **妆容修改**："为角色添加[妆容描述]"

### 3. 文字渲染
将需要生成的文字放入引号：`图片中央写着"创意无界"`

### 4. 构图控制
- 黄金分割：`三分法构图` `黄金螺旋`
- 视角：`俯视鸟瞰` `仰视` `正面平视` `45度斜角`
- 留白：`大量留白` `简洁背景` `主体偏[方向]`

### 5. 多图融合
最多支持 14 张参考图，融合时说明参考哪张图的哪个元素：
`参考图1的风格，图2的色调，图3的人物姿势`

### 6. 组图生成
触发词：`一系列` `组图` `生成N张连续的` `分镜图`

## 负向提示词写法

明确说明不需要的元素，放在提示词末尾：
- `背景简洁，不要杂乱元素`
- `保持人脸，不要改变面部特征`
- `不要文字水印`
- `不要过度曝光`

## 平台入口

- **即梦AI**（灰度测试）：https://jimeng.jianying.com/
- **火山方舟**（企业/4K直出）：https://console.volcengine.com/ark
- **剪映 App**：AI绘画功能选 Seedream 5.0
- **CapCut**（海外版）：AI Image 功能

每天免费 20 次生成额度（2K 清晰度免费，4K 增强可能收费）。

## Supported Platforms | 可用平台

| Platform | URL | Notes |
|----------|-----|-------|
| **Jimeng AI** 即梦AI | https://jimeng.jianying.com/ | Primary web app, 20 free/day |
| **Volcengine Ark** 火山方舟 | https://console.volcengine.com/ark | Enterprise API, 4K direct |
| **Jianying** 剪映 | App Store | AI Drawing → Seedream 5.0 |
| **CapCut** (Global) | App Store | AI Image feature |

## References | 参考文件

- Detailed examples & use cases → [examples.md](examples.md)
- Official docs, API params, size chart, full style dictionary → [reference.md](reference.md)
