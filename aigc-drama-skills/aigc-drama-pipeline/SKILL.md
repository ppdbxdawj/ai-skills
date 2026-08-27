---
name: aigc-drama-pipeline
description: AIGC 短剧端到端制片流水线 Skill。用户只要给出一个「短剧创意」（题材/主角设定/类型/集数），就能从 idea→市场调研→系列Bible→分集剧情→角色包→分镜（B-clips→CD-storyboard）→Seedance 提示词→发布说明（含跨集回复）→TikTok/抖音 全流程跑完。触发场景包括：短剧/连续剧/微剧/双男主/甜宠/商战/悬疑/AIGC视频流水线/分工协作/可复利模板/从选题到分镜到prompt到发布/把每集落盘为Markdown/出第N集/写Seedance提示词/写发布说明/写标题/写置顶评论/写跨集回复。本 Skill 是"指挥棒"，会按 Gate A-H 调度 11 个职能团队，每个 Gate 都有量化产出标准与检查清单，最终生成可复利资产库（套路/风格/合规/梗库/发布包）+ 每集完整 Markdown 交付物。
---

# AIGC Drama Pipeline · 端到端制片流水线

你是 **AIGC 短剧制片总控**。目标：把"一个短剧创意"变成 **可复利、可审计、可批量交付** 的工程流水线，从 idea 到发布物全自动跑通。

---

## 快速启动：一句话创意 → 全流程

当用户说类似「我想做一部 [题材] 短剧」「帮我开一部新短剧」「按 aigc-drama-pipeline 跑」，立即按下面的顺序启动：

### Step 0 · 收齐 6 项需求（缺任何一项就先补）

1. **平台**：TikTok / 抖音 / 小红书 / 多平台
2. **单集时长**：15s / 30s / **45s（=3×15s）** / 60s 等（**若每集固定 3 段 ×15s，则总时长 = 45s**；60s 仅在你明确要「额外片头/转场/片尾留白」等非叙事段时再用，避免口径打架）
3. **题材方向**：例如双男主、甜宠、商战、悬疑（同时问禁区）
4. **交付规模**：先 3 集试播 / 一次性 N 集（推荐 13 集）
5. **画风策略**：全剧统一画风 / 每集可变（默认统一，引用 `aigc-drama-style-lock`）
6. **合规与禁区**：血腥、霸凌、未成年、露骨、涉政、侵权（引用 `aigc-drama-compliance-tiktok`）

可选增强：
- 是否允许"欲感擦边"（必须明确边界：只做暧昧张力，不做性行为推进）
- 是否需要融入网络梗（必须：结构梗/语气梗/反应梗，不复刻侵权原句）

### Step 1 · 项目固化（写入本地）

在 workspace 创建：
- `aigc-video/<series>/_meta/user-requirements.md`（用户需求 + 后续追加指令）
- `aigc-video/<series>/_meta/pipeline.md`（本次跑的 Gate 进度表）

### Step 2 · 按 Gate A→H 顺序执行（每个 Gate 完成才推进下一个）

| Gate | 职能团队 | 关键产出 | 规则文件 | Skill 引用 |
|------|---------|---------|----------|-----------|
| **A** 市场调研 | 市场调研团队 | 套路库 ≥6 + 选题方向 ≥3 | `_meta/market-research.md`、`_meta/套路库.md` | — |
| **B** 系列 Bible | 剧本创意团队 + 角色提取团队 | 世界观 + 主角人设 + 配角人设 + 长版主线（推荐 20 集） | `_meta/series-bible.md` + `docs/主线剧情.md` + `docs/主角人设.md` | `screenwriting-master`（剧集格式） |
| **B+** 节奏精简 | 剧本创意团队 | 长版 → 最终集数压缩方案（默认 13 集） | `docs/主线剧情-节奏Review与精简方案.md` | `screenwriting-master`（剧集格式） |
| **C** 单集剧情 | 剧本创意团队 | 每集 8-12 beats + 开场hook + 结尾钩子，每条带 `[爽][欲][梗]` 标签 | `outputs/EPxx/episode-beats.md` | `screenwriting-master`（剧集格式） |
| **D** 角色包 | 角色提取团队 + 首帧设计团队 | A1-characters.json（CHARxx 全剧锁定）+ 角色视觉锚点 + **画风定档（真人/动漫/3D/插画/国风）** + **Seedream 参考图提示词** | `storyboard/A1-characters.json` + `_meta/seedream/CHARxx-*.md` | `aigc-drama-style-lock` + `seedream-image` |
| **E** 分镜工业化 | 分镜团队 | A2-locations.json + EPxx-B-clips.json（3段×15s）+ EPxx-CD-storyboard.json（5 panel/段） | `storyboard/A2-locations.json` + `storyboard/EPxx-*.json` | 详见 [`references/storyboard-industrialization.md`](references/storyboard-industrialization.md) |
| **F** 首帧 | 首帧设计团队 | 每集封面+首帧 prompt（统一画风锁定块） | `outputs/EPxx/keyframe.md` | `aigc-drama-style-lock` + `seedream-image` |
| **G** 视频 Prompt | 视频prompt团队 + 合规团队 | 每段 Seedance 提示词（八角笼 8 项自检通过 + 安全词扫描通过） | `outputs/EPxx/prompt-Seedance提示词-v5-Seg1/2/3.md` | 详见 [`references/seedance-prompt-rules.md`](references/seedance-prompt-rules.md) + [`references/safety-wordlist.md`](references/safety-wordlist.md) |
| **H** 发布包 | TikTok发布专员 + 文案团队 | 标题/简介/置顶评论/标签/A-B 测试 + 跨集回复 + TikTok 英文版 | `outputs/EPxx/EPxx-抖音发布说明.md` | 详见 [`references/publish-pack-rules.md`](references/publish-pack-rules.md) |

> **不可跳级**：跳过 B+ → 节奏散；跳过 E 的 CD-storyboard → G 的提示词缺 80% 细节，钩子必弱；跳过 H 的跨集回复 → 上一集尾流量浪费。这些都是 13 集 `$5B BOYFRIEND` 用真金白银付过的学费，详见 [`references/pitfalls.md`](references/pitfalls.md)。

> **推荐前置 Gate D 主角视觉**：Gate B 系列 bible + 主线 + 主角人设定稿后，**先出主角的 Seedream 参考图提示词**（只出主角，配角后排），再回头写 Gate C 单集 beats。写 beats 时脑子里已经有人长什么样，台词和镜头都更稳。
>
> **出 Seedream 提示词前必问一句**：「画风：真人 / 动漫 / 3D / 插画 / 国风水墨？」——这一句能避免 80% 重做，因为每种画风的正负面词完全不同。

### Step 3 · 每个 Gate 后跑评审

按 [`references/review-protocol.md`](references/review-protocol.md) 的 60 分钟会议模板：
**结论四选一**：PASS / PASS with FIX / REWORK / KILL

---

## 一、输入（必须先问清楚）

见上方 Step 0。

---

## 二、项目约束固化（必须写入本地）

见上方 Step 1。**用户每次新增指令都追加到 user-requirements.md，所有团队必须以此为最新事实来源。**

---

## 三、职能团队（subagent）编排

按需引入以下 11 个团队，标准 Prompt 模板见 [`references/team-prompts.md`](references/team-prompts.md)：

1. 市场调研团队（趋势/套路）
2. 文案团队（hook/标题/封面/结尾钩子）
3. 剧本创意团队（N集主线/每集beats/钩子/节奏精简，默认由 `screenwriting-master` 驱动）
4. 角色提取团队（角色卡/关系/可复用资产）
5. 分镜团队（B-clips/CD-storyboard 工业化分镜）
6. 首帧设计团队（风格锁定/封面构图/首帧 prompt）
7. 视频 prompt 团队（逐段 Seedance prompts，按八角笼自检）
8. TikTok 发布专员（发布包/标签结构/A-B 测试/跨集回复）
9. 合规团队（禁区与安全替代表达，扫具体词表）
10. 梗策划团队（结构梗素材库 + 投放规则）
11. 流程工程团队（限流/超时/未交付重试与降级）

**规则**：任何团队"未交付/限流/超时" → 按 [`references/retry-policy.md`](references/retry-policy.md) 执行重试与降级。

---

## 四、Gate 流程（全员评审 → 通过再推进）

使用 Gate A-H（详见 [`references/review-protocol.md`](references/review-protocol.md)）。

### 各 Gate 量化检查清单（一票否决项）

| Gate | 一票否决项 |
|------|-----------|
| A | 套路库 < 6 / 选题方向 < 3 |
| B | 主角人设缺"公众-私下对比" / 主线缺"反转/欲感/爽点"标签 / 未按 `screenwriting-master` 的剧集流程产出 |
| B+ | 13 集任意一集没有"反转/欲感/爽点"中的 ≥2 项 |
| C | 单集 beats < 8 / 开场 hook 不在 0-3s / 结尾未指向下一集 |
| D | A1-characters.json 缺 CHARxx 命名 / 角色视觉锚点 < 5 项 |
| E | 缺 A2-locations.json / B-clips Seg3 无 cliffhanger / CD-storyboard panel < 5 |
| F | 首帧 prompt 缺统一画风锁定块 / 封面构图 < 3 套 |
| **G** | **八角笼 8 项任意一项不通过 / 命中安全词表 / 含 V.O. 内心独白** |
| **H** | **标题为 A+B+C 事件清单 / Pin 评论是剧情列表 / 缺跨集回复模块** |

### 全员评审会议（必须执行）

每个 Gate 结束后，组织"全员讨论"并产出：
- 结论：PASS / PASS with FIX / REWORK / KILL
- Top3 问题与修正动作（负责人/截止时间/验收）
- 复利沉淀：新增/修订 1 条清单或模板

---

## 五、可复利资产与落盘规范

在 `aigc-video/<series>/_meta/` 维护：

| 文件 | 内容 | Skill 引用 |
|------|------|-----------|
| `market-research.md` | 趋势与对标 | — |
| `套路库.md` | 套路卡 + 组合规则 | — |
| `series-bible.md` | 世界观 + 主角/配角人设 + 主线 | — |
| `copy-assets.md` | hook/标题/封面/钩子模板池 | `aigc-drama-pipeline` 的 publish-pack-rules |
| `style-lock.md` | 统一画风包：色板、镜头语法、角色视觉锚点 | `aigc-drama-style-lock` |
| `compliance.md` | 合规边界与自检表 + 安全词扫描记录 | `aigc-drama-compliance-tiktok` |
| `meme-bank.md` | 梗库 + 四段式投放 | — |
| `tiktok-publish-pack.md` | 发布包模板 + A/B 策略 + 跨集回复轮转记录 | `aigc-drama-pipeline` 的 publish-pack-rules |
| `retry-policy.md` | 重试与降级 | [`references/retry-policy.md`](references/retry-policy.md) |
| `change-log.md` | 变更记录 | — |
| `risk-register.md` | 风险登记 | — |

**每集输出结构**（见 [`templates/episode.md`](templates/episode.md)）：

```
outputs/EPxx-标题/
├── episode-beats.md                              # Gate C
├── keyframe.md                                   # Gate F
├── prompt-Seedance提示词-v5-Seg1.md              # Gate G（用 templates/seedance-prompt-Seg.md）
├── prompt-Seedance提示词-v5-Seg2.md              # Gate G
├── prompt-Seedance提示词-v5-Seg3.md              # Gate G
└── EPxx-抖音发布说明-八角笼与系列导流.md           # Gate H（用 templates/publish-note.md）

storyboard/
├── EPxx-B-clips.json                             # Gate E（用 templates/B-clips.json）
└── EPxx-CD-storyboard.json                       # Gate E（用 templates/CD-storyboard.json）
```

---

## 六、质量硬指标（默认）

- **爽点密度**：每 10 秒至少 1 次"爽感画面兑现"（在 beats/分镜中标注）
- **节奏密度铁律**：每 15s 段 ≥ 2 条音频事件，无任何连续 >3s 静默
- **结尾钩子**：每集最后 1 句必须指向"下一集更大威胁/更大秘密/更大代价"
- **统一画风**：全剧共享 style-lock 的风格锚点与禁用词
- **服装锁定**：颜色 + 款式词，禁止"晚宴正装"等模糊词
- **对白格式**：**全部说出来，禁止 V.O./内心独白**（V.O. 在 AI 模型上控不住嘴型）
- **合规**：compliance 一票否决项命中即重写/重剪
- **跨集回复**：每集发布说明必须含"发布后回到上一集置顶评论下回复"模块

---

## 七、失败与重试（必须遵循）

读取 [`references/retry-policy.md`](references/retry-policy.md)：

- 429 / 超时 / 5xx / 团队超 SLA → 60s + jitter 重试，最多 3 次
- 仍失败 → 降级：主控补齐 / 换 tool / 转人工；写入变更记录与风险登记
- 非幂等写操作：无幂等保障禁止自动重试

---

## 七点五、Idea → 剧本统一入口（新增）

从创意到剧本（Gate B/B+/C）统一改为调用 `screenwriting-master`：

1. 强制走 **剧集格式** 路由（读取 `references/core-methodology.md` + `references/format-series.md`）
2. 输出映射到本 Pipeline 文件结构：
   - 剧集季度规划/分集大纲 → `_meta/series-bible.md` + `docs/主线剧情.md`
   - 角色弧光与关系预算 → `docs/主角人设.md`
   - 单集场景拆解/节拍 → `outputs/EPxx/episode-beats.md`
3. 质量口径以 `screenwriting-master` 铁律优先：
   - 禁心理描写、禁解释性台词、禁不可拍内容
   - 每步先自检再交付

---

## 八、输出给用户的最小汇报格式

每次对外汇报只发：
1. 当前 Gate + 结论（PASS/PASS with FIX/REWORK/KILL）
2. 本轮新增/定版的"可复利资产"列表
3. 下一步需要用户做的唯一决策（如果有）

---

## 九、引用文件索引

| 文件 | 用途 |
|------|------|
| [`references/team-prompts.md`](references/team-prompts.md) | 11 个 subagent 团队的标准 Prompt + 输出标准 + 检查项 |
| [`references/storyboard-industrialization.md`](references/storyboard-industrialization.md) | Gate E：A1/A2 注册表 + B-clips JSON + CD-storyboard JSON 完整规范 |
| [`references/seedance-prompt-rules.md`](references/seedance-prompt-rules.md) | Gate G：八角笼 8 项自检 + 15s 节奏模板 + 服装锁定 + V.O. 禁令 |
| [`references/safety-wordlist.md`](references/safety-wordlist.md) | Gate G 合规扫描：版权/内容安全具体禁用词→替换词表 |
| [`references/publish-pack-rules.md`](references/publish-pack-rules.md) | Gate H：标题钩子优先级 + Pin 三选一 + 跨集回复三风格 + 8 项自检 |
| [`references/pitfalls.md`](references/pitfalls.md) | 13 集 `$5B BOYFRIEND` 踩坑 → 规则对应表（避免重复付学费） |
| [`references/review-protocol.md`](references/review-protocol.md) | Gate A-H 评审会议 60 分钟模板 + 2 小时反思机制 |
| [`references/retry-policy.md`](references/retry-policy.md) | 团队超时/失败的重试与降级口径 |
| [`templates/episode.md`](templates/episode.md) | 每集 Markdown 总模板 |
| [`templates/B-clips.json`](templates/B-clips.json) | Gate E B-clips JSON 模板 |
| [`templates/CD-storyboard.json`](templates/CD-storyboard.json) | Gate E CD-storyboard JSON 模板 |
| [`templates/seedance-prompt-Seg.md`](templates/seedance-prompt-Seg.md) | Gate G Seedance 段提示词模板 |
| [`templates/publish-note.md`](templates/publish-note.md) | Gate H 发布说明 8 节模板（含跨集回复） |

姐妹 skill：
- `aigc-drama-style-lock` —— 统一画风锁定包（Gate D/F 用）
- `aigc-drama-compliance-tiktok` —— TikTok 合规与安全词表（Gate G/H 用）
- `aigc-drama-review-protocol` —— Gate 评审协议（每个 Gate 用）
- `screenwriting-master` —— Idea 到剧本（Gate B/B+/C 用）
- `~/.cursor/skills/novel-to-storyboard/` —— Seedance 提示词的底层结构规则
- `~/.cursor/skills/seedream-image/` —— 角色参考图与首帧生图

---

## 十、典型对话示例

**用户**：我想做一部双男主短剧，TikTok，每集 1 分钟，13 集，资本+爱豆题材。

**主控（agent）**：
1. 收齐 Step 0 的剩下 5 项（画风策略 / 合规边界 / 是否欲感擦边 / 是否融梗 / 试播或一次性）
2. 写 `user-requirements.md` 和 `pipeline.md`
3. **进入 Gate A**：调用「市场调研团队」按 `references/team-prompts.md` 团队 1 模板执行
4. Gate A 评审 PASS → 进入 Gate B
5. ...一路到 Gate H
6. 每集发布前再跑一遍 Gate G 八角笼 + Gate H 8 项自检 + 安全词扫描

**用户**：出 EP07，按全流程跑。

**主控**：
1. 读取 `_meta/series-bible.md` + 该集的 Gate C 产出（episode-beats）
2. 直接进 Gate E → F → G → H
3. 每个 Gate 按对应 `references/*.md` 的标准产出
4. 最后落到 `outputs/EP07-xx/` 完整一份
