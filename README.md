# Seedream Image Prompt Skill

**Generate high-quality AI image prompts for [Seedream 5.0](https://jimeng.jianying.com/) by ByteDance.**

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

### Features

- **Text-to-Image Prompts** — Structured prompt templates for portraits, landscapes, posters, cards, and more
- **Image Editing** — Style transfer, element add/remove, lighting adjustment, portrait retouching
- **API Image Generation** — One-command image generation via Seedream 4.0 API (`generate.py`)
- **Character Consistency** — Keep face, hair, clothing identical across multiple generated images
- **Knowledge Cards** — Infographic, tarot, blueprint, game skill card, and ancient scroll styles
- **E-commerce** — Product photography prompts for online stores
- **Multi-image Fusion** — Combine up to 10 reference images with precise element control
- **Group Image Generation** — Generate up to 10 related images in one call (storyboards, variations)
- **Text Rendering** — 99%+ accuracy for Chinese/English text in images (wrap text in quotes)
- **Style Dictionary** — 200+ curated style keywords organized by category
- **API Reference** — Seedream 4.0 visual API docs, size charts, and Python examples

### Quick Install

**Cursor / Claude Code / Cline / Other Agents:**

```bash
npx skills add ppdbxdawj/seedream-image-skill@seedream-image
```

**Manual:**

```bash
git clone https://github.com/ppdbxdawj/seedream-image-skill.git
cp -r seedream-image-skill/seedream-image ~/.cursor/skills/
```

### Supported Platforms

| Platform | Access | Notes |
|----------|--------|-------|
| [Jimeng AI](https://jimeng.jianying.com/) | Web | 20 free generations/day |
| [Volcengine Ark](https://console.volcengine.com/ark) | API | Enterprise, 4K direct |
| [Jianying](https://www.capcut.cn/) | App | AI Drawing feature |
| [CapCut](https://www.capcut.com/) | App | Global version |

### Example Prompts

**Portrait Photography**
```
Young Asian man, clean features, black oversized sweater,
sitting by library window reading, sunlight on side face,
Tyndall light, cool tones, shallow depth of field, 85mm prime lens,
cinematic realism style, ultra HD
```

**Knowledge Card (Fantasy RPG Style)**
```
Design a fantasy RPG game skill card explaining "[CONCEPT]".
Include a cool glowing icon, skill description in game language,
energy cost and cooldown timer. Dark fantasy style with luminous
magical rune borders.
```

**E-commerce Product Shot**
```
Create a product showcase for this [PRODUCT] in [BRAND] style,
clean white background, highlighting texture and material quality,
professional commercial photography, high contrast.
```

### API Image Generation

Generate images directly from the command line using the Seedream 4.0 API:

```bash
# Setup
pip install -r requirements.txt
export VOLC_ACCESSKEY="your_access_key"
export VOLC_SECRETKEY="your_secret_key"

# Text-to-image
python generate.py --prompt "A cute cat in a garden, watercolor style"

# Image editing (with reference image)
python generate.py --prompt "Change background to beach" --image-urls "https://example.com/photo.jpg"

# Specify resolution + single image
python generate.py --prompt "Product hero shot" --width 2560 --height 1440 --force-single

# Group image generation
python generate.py --prompt "Generate 4 blind box figures: crow, rabbit, dog, cat"
```

Get your API credentials at [Volcengine IAM](https://console.volcengine.com/iam/keymanage/) and enable the service at [Jimeng 4.0](https://console.volcengine.com/ai/ability/detail/10).

### Use Cases

| Scenario | Key Prompts |
|----------|-------------|
| Social media avatar | `square composition` `solid background` `icon style` |
| Knowledge card | `infographic` `clean layout` `target audience: ...` |
| PPT background | `for PPT cover background` `whitespace` `matte background` |
| E-commerce hero | `product hero shot` `commercial photography` `clean background` |
| Poster design | `movie poster level` `dramatic lighting` `large whitespace` |
| Brand design | `brand VI visual` `include logo/packaging/cards` |
| Character cosplay | `keep face unchanged` `realistic costume` `same pose` |
| Storyboard / comic | `generate a set of comics` `storyboard` `group image` |
| Education | `comparison chart` `info poster` `knowledge explainer` |

### File Structure

```
seedream-image/
├── SKILL.md          # Prompt structures, style vocabulary, templates, API usage guide
├── generate.py       # Seedream 4.0 API generation script (submit + poll + download)
├── requirements.txt  # Python dependencies (volcengine SDK)
├── examples.md       # 30+ ready-to-use prompts across 8 categories
├── reference.md      # Official docs, API params, size charts, 200+ style keywords
└── marketplace.json  # Skill metadata for search indexing
```

---

<a id="中文"></a>

## 中文

### 功能特性

- **文生图提示词** — 人物写真、风景、海报、卡片等结构化提示词模板
- **图像编辑** — 风格迁移、元素增删、光影调整、人像美化
- **API 一键生图** — 通过即梦 4.0 API 一键生成图片（`generate.py` 脚本）
- **角色一致性** — 跨图保持面部、发型、服装高度一致（连续分镜可用）
- **知识卡片** — 信息图、塔罗牌、蓝图、游戏技能卡、仙侠古籍等风格
- **电商产品图** — 商品展示图、试穿效果、品牌风格匹配
- **多图融合** — 最多 10 张参考图，精确控制各图元素组合
- **组图生成** — 一次生成最多 10 张关联图像（分镜、变体、系列图）
- **文字渲染** — 中英文 99%+ 准确率，引号包裹文字即可精准生成
- **风格词典** — 200+ 风格关键词，按写实摄影/动漫插画/设计商业/光影分类整理
- **API 参考** — 即梦 4.0 Visual API 文档、尺寸速查表、Python 调用示例

### 快速安装

**Cursor / Claude Code / Cline / 其他 Agent：**

```bash
npx skills add ppdbxdawj/seedream-image-skill@seedream-image
```

**手动安装：**

```bash
git clone https://github.com/ppdbxdawj/seedream-image-skill.git
cp -r seedream-image-skill/seedream-image ~/.cursor/skills/
```

### 支持平台

| 平台 | 访问方式 | 说明 |
|------|----------|------|
| [即梦AI](https://jimeng.jianying.com/) | 网页 | 每天 20 次免费生成 |
| [火山方舟](https://console.volcengine.com/ark) | API | 企业版，4K 直出 |
| [剪映](https://www.capcut.cn/) | App | AI 绘画功能 |
| [CapCut](https://www.capcut.com/) | App | 海外版 |

### 提示词示例

**人物写真**
```
清秀亚洲青年男性，黑色宽松毛衣，坐在图书馆靠窗位置低头看书，
窗外阳光透进来打在侧脸，丁达尔光，冷色调，浅景深背景虚化，
85mm定焦，写实电影剧照风格，超高清
```

**知识卡片（游戏技能卡风格）**
```
设计一张幻想RPG游戏中的技能卡片，用文字和图像解释"[概念]"。
有游戏化的技能名称，卡片上有酷炫图标、技能描述（用游戏化语言解释概念）、
消耗的"精力值"和冷却时间。整体是暗黑奇幻风格，带有发光的魔法符文边框。
```

**电商产品图**
```
为这件[产品]创建网店展示图，风格类似于[GUCCI/PRADA/Nike]，
简洁白色背景，突出产品质感，专业商业摄影风格，高对比度。
```

### API 一键生图

通过命令行直接调用即梦 4.0 API 生成图片：

```bash
# 环境准备
pip install -r requirements.txt
export VOLC_ACCESSKEY="你的AccessKey"
export VOLC_SECRETKEY="你的SecretKey"

# 文生图
python generate.py --prompt "一只猫在花园里玩耍，水彩风格"

# 图像编辑（传入参考图）
python generate.py --prompt "将背景换成海滩" --image-urls "https://example.com/photo.jpg"

# 指定分辨率 + 强制单图
python generate.py --prompt "电商主图，产品特写" --width 2560 --height 1440 --force-single

# 组图生成
python generate.py --prompt "生成4张分别关于春夏秋冬的盲盒组图"
```

API 凭证获取：[火山引擎密钥管理](https://console.volcengine.com/iam/keymanage/)，服务开通：[即梦 4.0](https://console.volcengine.com/ai/ability/detail/10)

### 使用场景速查

| 场景 | 关键提示词 |
|------|-----------|
| 社交头像 | `正方形构图` `纯色背景` `图标风格` |
| 知识卡片 | `信息图` `排版清晰` `目标受众：...` |
| PPT 背景 | `用于PPT封面背景` `留白构图` `哑光背景` |
| 电商主图 | `电商主图` `产品展示` `专业商业摄影` |
| 海报设计 | `电影海报级别` `戏剧光` `大面积留白` |
| 品牌设计 | `品牌VI视觉` `包含logo/包装/卡片` |
| 角色 Cos | `保持人脸不变` `写实质感服饰` `相同姿势` |
| 分镜漫画 | `生成一组漫画` `分镜图` `组图` |
| 教育培训 | `对比图` `信息海报` `知识科普` |

### 文件结构

```
seedream-image/
├── SKILL.md          # 提示词结构、风格词汇库、常用模板、API 使用指南
├── generate.py       # 即梦 4.0 API 生图脚本（提交 + 轮询 + 下载）
├── requirements.txt  # Python 依赖（volcengine SDK）
├── examples.md       # 30+ 即用提示词，覆盖 8 大类场景
├── reference.md      # 官方文档、API 参数、尺寸表、200+ 风格词
└── marketplace.json  # 搜索索引元数据
```

---

## Keywords

`seedream` `seedream 4.0` `即梦` `jimeng` `bytedance` `doubao` `豆包` `volcengine` `火山方舟`
`AI image generation` `AI生图` `text-to-image` `文生图` `image editing` `图生图`
`prompt engineering` `提示词` `character consistency` `角色一致性`
`knowledge card` `知识卡片` `poster design` `海报设计` `e-commerce` `电商`
`group image` `组图` `storyboard` `分镜` `multi-image fusion` `多图融合`
`capcut` `jianying` `剪映` `AI art` `AI绘画` `creative tools` `API generation` `一键生图`

## License

MIT
