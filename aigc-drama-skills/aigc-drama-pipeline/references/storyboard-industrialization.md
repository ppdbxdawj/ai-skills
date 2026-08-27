# Storyboard Industrialization · Gate E 完整规范

> Gate E 的工作分两步：先建全剧资产注册表（A1/A2，一次性），再每集出 B-clips（3 段）+ CD-storyboard（每段 5 panel）。
>
> **跳过 CD-storyboard 直接写 Seedance 提示词 = 失去 80% 细节**，这是 13 集 EP09 用真金白银付的学费。
>
> 配套文件：
> - [`../templates/B-clips.json`](../templates/B-clips.json)
> - [`../templates/CD-storyboard.json`](../templates/CD-storyboard.json)

---

## A · 资产注册表（全剧只做一次）

### A.1 `storyboard/A1-characters.json`

```json
{
  "CHAR01": {
    "name_cn": "凯斯·诺克斯",
    "name_en": "Kace Knox",
    "ref_image": "kace.png",
    "visual_anchor": [
      "32 岁",
      "灰蓝色眼睛",
      "黑色短发往后梳",
      "1.88m 健身体型",
      "左眉尾有一道浅疤"
    ],
    "voice_anchor": "低沉胸腔音 + 缓慢半拍",
    "public_persona": "冷峻商业霸总",
    "private_persona": "占有欲极强的偏执隐粉"
  },
  "CHAR02": {
    "name_cn": "陈泽川",
    "name_en": "Zen Chen",
    "ref_image": "zen.png",
    "visual_anchor": [...],
    "voice_anchor": "...",
    "public_persona": "清冷国民爱豆",
    "private_persona": "社恐软萌"
  }
}
```

**字段标准**：每个 CHAR 至少含 6 字段（name_cn / name_en / ref_image / visual_anchor / voice_anchor / public_persona），主角必含 `private_persona`。

### A.2 `storyboard/A2-locations.json`

```json
{
  "LOC01": {
    "name": "Kace 顶层公寓客厅",
    "lighting_default": "暖色琥珀 + 冷蓝窗光",
    "color_palette": "深木色 + 黑色大理石 + 暖金属",
    "atmosphere": "压抑奢华",
    "camera_notes": "落地窗作为常用前景，俯视城市夜景"
  },
  "LOC02": { "name": "ZK Entertainment 总部走廊", ... }
}
```

**字段标准**：每个 LOC 至少 5 字段（name / lighting_default / color_palette / atmosphere / camera_notes）。

---

## B · B-clips：一集切 3 段 × 15s

### B.1 文件位置

`storyboard/EPxx-B-clips.json`

### B.2 标准结构

```json
[
  {
    "clip_id": "EP10-C1",
    "start": "本段从哪个画面开始（具体到镜头）",
    "end": "本段在哪个画面结束（必须有钩子感或情绪落点）",
    "summary": "本段核心剧情一句话（含[爽]/[欲]/[反转]标签）",
    "location": "LOC02",
    "characters": ["CHAR01", "CHAR02"],
    "duration_target": "15s",
    "segment": "Seg1",
    "pace_note": "本段节奏建议（多少 beat、节奏曲线、视觉重点）",
    "energy_curve": "冷峻"
  },
  { "clip_id": "EP10-C2", "segment": "Seg2", "energy_curve": "缠绵", ... },
  { "clip_id": "EP10-C3", "segment": "Seg3", "energy_curve": "骤变", ... }
]
```

### B.3 通过标准（Gate E 一票否决）

- [ ] 每集恰好 3 段
- [ ] 每段 `duration_target` = 15s
- [ ] `clip_id` 命名规范 `EPxx-Cy`
- [ ] `location` 引用 A2-locations.json 的真实 LOC ID
- [ ] `characters` 引用 A1-characters.json 的真实 CHAR ID
- [ ] **三段的 `energy_curve` 必须有对比**（典型：冷峻 → 缠绵 → 骤变）
- [ ] **Seg3 的 `end` 必须是截断式 cliffhanger，明确指向下集**
- [ ] `summary` 必须含 `[爽]` / `[欲]` / `[反转]` 至少一个标签

---

## C · CD-storyboard：每段切 5 panel × 3s

### C.1 文件位置

`storyboard/EPxx-CD-storyboard.json`

### C.2 标准结构

```json
{
  "EP10": [
    {
      "clip_id": "EP10-C1",
      "segment": "Seg1",
      "screenplay": {
        "title": "办公室宣战",
        "content": [
          "0-3s：Kace 推开 Zen 的更衣室门，手里拿着收购协议。",
          "3-7s：Kace 走到 Zen 面前，把协议拍在桌上。",
          "...完整剧本流"
        ]
      },
      "panels": [
        {
          "panel_number": 1,
          "shot_type": "近景",
          "camera_move": "固定",
          "duration": "3s",
          "description": "Kace 推开门，门把转动声 + 急促脚步停在画面左侧",
          "photographyPlan": {
            "lighting": "门外冷白逆光 + 室内暖黄面光",
            "colorPalette": "冷蓝 + 暖琥珀对比",
            "atmosphere": "压迫感",
            "technicalNotes": "人物剪影优先，眼神高光突出"
          },
          "actingNotes": [
            { "name": "凯斯·诺克斯", "acting": "右手按门框，下颌微抬，眼神锁定" }
          ],
          "audio": {
            "dialogue": [],
            "sfx": "门轴吱呀 + 衣料摩擦",
            "bgm": "低音鼓点起拍"
          }
        },
        { "panel_number": 2, "shot_type": "中景", ... },
        { "panel_number": 3, "shot_type": "特写", ... },
        { "panel_number": 4, "shot_type": "中景", ... },
        { "panel_number": 5, "shot_type": "特写截断", ... }
      ]
    },
    { "clip_id": "EP10-C2", ... },
    { "clip_id": "EP10-C3", ... }
  ]
}
```

### C.3 通过标准（Gate E 一票否决）

- [ ] 每集 3 段，每段 5 panel
- [ ] 每段都有完整的 `screenplay.content[]`
- [ ] 每个 panel 7 字段齐全：`panel_number / shot_type / camera_move / duration / description / photographyPlan / actingNotes / audio`
- [ ] `photographyPlan` 4 子字段齐全：`lighting / colorPalette / atmosphere / technicalNotes`
- [ ] `audio` 3 子字段齐全：`dialogue / sfx / bgm`（可空数组但必须存在）
- [ ] 5 panel 的 `duration` 加起来 = 15s（一般是 3+3+3+3+3 或 2+3+4+3+3 等灵活分配）

---

## D · 为什么必须做完整的 CD-storyboard

直接从 B-clips 跳到 Seedance 提示词会丢：

| 丢失内容 | 后果 |
|---------|------|
| 每帧的灯光方向 | AI 自己脑补光线，全段光线不连贯 |
| 镜头景别变化 | 整段一个景别，节奏死掉 |
| 表演细节 | 角色站着不动只动嘴，没有戏 |
| 音频卡点 | 没有 SFX/BGM 节奏支撑，画面空 |
| 5 panel 之间的过渡 | 突兀的硬切，无视觉引导 |

**EP09 教训**：跳了 CD-storyboard 直接写提示词 → 钩子弱、对白和原剧本对不上、视觉无层次 → 重做一次 → 浪费 1 次模型生成额度 + 半天工时。

---

## E · 与 Gate G 的对接

CD-storyboard 写完后，Gate G 视频 prompt 团队的工作就是：

```
对每个 clip：
  对每个 panel：
    把 description + photographyPlan + actingNotes + audio
    转写成 0-3s/3-7s/... 时间块的 Seedance 提示词文本
  追加服装锁定声明（开头）+ 禁止句（末尾）
  跑八角笼 8 项自检
```

实际上**有了 CD-storyboard，Gate G 几乎是机械翻译**，所以这一步千万不要省。
