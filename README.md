# Seedream Image Prompt Skill

**Generate high-quality prompts for [Seedream 5.0/4.0](https://jimeng.jianying.com/) (即梦AI) by ByteDance — describe your idea, review the prompt, confirm, and get images auto-downloaded to your machine.**

**为字节即梦AI 生成高质量提示词：描述想法 → 审核提示词 → 确认后自动生图并下载到本地。**

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

### Features

- **Text-to-Image Prompts** — Structured prompt templates for portraits, landscapes, posters, cards, and more
- **Image Editing** — Style transfer, element add/remove, lighting adjustment, portrait retouching
- **API Image Generation** — One-command generation via Seedream 4.0 API; images are auto-downloaded to `output/`
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

### How It Works

You don't need to run any commands yourself. The workflow is simple:

1. **Describe what you want** — Tell the agent your image idea in natural language
2. **Review the prompt** — The agent generates an optimized Seedream prompt for you to review
3. **Confirm and generate** — Before sending, the agent will confirm: *default is 1 image; do you need multiple (group)?* If you need a set, the agent uses `--no-force-single` or adds group trigger words. Once you approve, the agent runs the API and downloads the image(s) to the `output/` folder

That's it! The agent handles prompt engineering, the single/group choice, API calls, polling, and auto-download.

> **One-time setup**: Install deps and set Volcengine API credentials. Get them at [Volcengine IAM](https://console.volcengine.com/iam/keymanage/) and enable [Jimeng 4.0](https://console.volcengine.com/ai/ability/detail/10).
>
> **Option A — .env (recommended)**  
> Create a `.env` file in the **same folder as `generate.py`** (that folder is the skill’s inner directory, e.g. after `npx skills add` it may be under `~/.cursor/skills/` or your project’s skills path). The script loads it automatically; works the same for manual install or `npx skills add`. Do not commit `.env`.
> ```bash
> pip install -r requirements.txt
> # In the directory that contains generate.py, create .env:
> VOLC_ACCESSKEY=your_access_key
> VOLC_SECRETKEY=your_secret_key
> ```
>
> **Option B — shell**  
> ```bash
> export VOLC_ACCESSKEY="your_access_key"
> export VOLC_SECRETKEY="your_secret_key"
> ```

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

<details>
<summary><b>Advanced: Manual CLI Usage</b></summary>

```bash
# Text-to-image (default: 1 image)
python generate.py --prompt "A cute cat in a garden, watercolor style"

# Image editing (with reference image)
python generate.py --prompt "Change background to beach" --image-urls "https://example.com/photo.jpg"

# Specify resolution (still 1 image by default)
python generate.py --prompt "Product hero shot" --width 2560 --height 1440

# Group image generation (use --no-force-single)
python generate.py --prompt "Generate 4 blind box figures: crow, rabbit, dog, cat" --no-force-single
```

</details>

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
- **API 一键生图** — 通过即梦 4.0 API 一键生成，图片自动下载到 `output/` 目录
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

### 使用流程

你不需要手动执行任何命令，流程非常简单：

1. **描述你的想法** — 用自然语言告诉 Agent 你想生成什么图
2. **审核提示词** — Agent 会生成优化后的 Seedream 提示词供你审核
3. **确认并生成** — 发起请求前 Agent 会软提示：默认生成 1 张，需要多张（组图）吗？需要则加 `--no-force-single` 或保留组图触发词。确认后 Agent 调用 API 并将图片自动下载到 `output/` 目录

就这么简单！Agent 会自动处理提示词优化、单张/组图确认、API 调用、轮询和图片下载。

> **首次使用需配置**：安装依赖并设置火山引擎 API 凭证。前往 [火山引擎密钥管理](https://console.volcengine.com/iam/keymanage/) 获取密钥，并开通 [即梦 4.0 服务](https://console.volcengine.com/ai/ability/detail/10)。
>
> **方式一 — .env（推荐）**  
> 在 **与 `generate.py` 同一目录**下新建 `.env` 文件（用 `npx skills add` 安装时，该目录一般在 `~/.cursor/skills/` 或项目 skills 路径下）。脚本会自动读取，手动安装或 npx 安装均适用。不要提交 `.env`。
> ```bash
> pip install -r requirements.txt
> # 在 generate.py 所在目录创建 .env，写入：
> VOLC_ACCESSKEY=你的AccessKey
> VOLC_SECRETKEY=你的SecretKey
> ```
>
> **方式二 — 终端环境变量**  
> ```bash
> export VOLC_ACCESSKEY="你的AccessKey"
> export VOLC_SECRETKEY="你的SecretKey"
> ```

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

<details>
<summary><b>进阶：手动命令行用法</b></summary>

```bash
# 文生图（默认 1 张）
python generate.py --prompt "一只猫在花园里玩耍，水彩风格"

# 图像编辑（传入参考图）
python generate.py --prompt "将背景换成海滩" --image-urls "https://example.com/photo.jpg"

# 指定分辨率（默认 1 张）
python generate.py --prompt "电商主图，产品特写" --width 2560 --height 1440

# 组图生成（加 --no-force-single）
python generate.py --prompt "生成4张分别关于春夏秋冬的盲盒组图" --no-force-single
```

</details>

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
