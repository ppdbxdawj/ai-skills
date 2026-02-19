# Seedream Image Prompt Skill

**Generate high-quality AI image prompts for [Seedream 5.0](https://jimeng.jianying.com/) by ByteDance.**

为字节跳动 [即梦AI（Seedream 5.0）](https://jimeng.jianying.com/) 生成高质量图片提示词的 Agent Skill。

---

## Features

- **Text-to-Image Prompts** — Structured prompt templates for portraits, landscapes, posters, cards, and more
- **Image Editing** — Style transfer, element add/remove, lighting adjustment, makeup changes
- **Character Consistency** — Keep face, hair, clothing identical across multiple generated images
- **Knowledge Cards** — Infographic, tarot, blueprint, game skill card, and ancient scroll styles
- **E-commerce** — Product photography prompts for online stores
- **Multi-image Fusion** — Combine up to 14 reference images with precise element control
- **Text Rendering** — 99%+ accuracy for Chinese/English text in images
- **Style Dictionary** — 200+ curated style keywords organized by category
- **API Reference** — Complete Volcengine Ark API params, size charts, and Python examples

## Quick Install

### Cursor / Claude Code / Cline / Other Agents

```bash
npx skills add ppdbxdawj/seedream-image-skill@seedream-image
```

### Manual Install

```bash
git clone https://github.com/ppdbxdawj/seedream-image-skill.git
cp -r seedream-image-skill/seedream-image ~/.cursor/skills/
```

## File Structure

```
seedream-image/
├── SKILL.md          # Main skill document — prompt structures, style vocabulary, templates
├── examples.md       # 30+ ready-to-use prompt examples across 8 categories
├── reference.md      # Official docs, API params, size charts, 200+ style keywords
├── marketplace.json  # Skill metadata for search indexing
└── README.md         # This file
```

## Supported Platforms

| Platform | Access | Notes |
|----------|--------|-------|
| [Jimeng AI](https://jimeng.jianying.com/) | Web | 20 free generations/day |
| [Volcengine Ark](https://console.volcengine.com/ark) | API | Enterprise, 4K direct |
| [Jianying](https://www.capcut.cn/) | App | AI Drawing feature |
| [CapCut](https://www.capcut.com/) | App | Global version |

## Example Prompts

### Portrait Photography
```
Young Asian man, clean features, black oversized sweater,
sitting by library window reading, sunlight on side face,
Tyndall light, cool tones, shallow depth of field, 85mm prime lens,
cinematic realism style, ultra HD
```

### Knowledge Card (Fantasy RPG Style)
```
Design a fantasy RPG game skill card explaining "[CONCEPT]".
Include a cool glowing icon, skill description in game language,
energy cost and cooldown timer. Dark fantasy style with luminous
magical rune borders.
```

### E-commerce Product Shot
```
Create a product showcase for this [PRODUCT] in [BRAND] style,
clean white background, highlighting texture and material quality,
professional commercial photography, high contrast.
```

## Use Cases

| Scenario | Key Prompts |
|----------|-------------|
| Social media avatar | `square composition` `solid background` `icon style` |
| Knowledge card | `infographic` `clean layout` `target audience: ...` |
| PPT background | `whitespace composition` `matte background` `offset to [side]` |
| Character cosplay | `keep face unchanged` `realistic costume` `same pose` |
| Poster design | `movie poster level` `dramatic lighting` `large whitespace` |
| Product photo | `commercial photography` `clean background` `texture highlight` |

## Keywords

`seedream` `即梦` `jimeng` `bytedance` `doubao` `豆包` `volcengine` `火山方舟`
`AI image generation` `AI生图` `text-to-image` `文生图` `image editing` `图生图`
`prompt engineering` `提示词` `character consistency` `角色一致性`
`knowledge card` `知识卡片` `poster design` `海报设计` `e-commerce` `电商`
`capcut` `jianying` `剪映` `AI art` `AI绘画` `creative tools`

## License

MIT
