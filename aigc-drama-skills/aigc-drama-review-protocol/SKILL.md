---
name: aigc-drama-review-protocol
description: AIGC 短剧项目的"全员评审 + 质量门槛 + 2 小时反思"通用 Skill。只要用户提到评审机制、质量把控、Gate 流程、全员讨论、复盘、反思、可复利流程标准化、八角笼自检、发布说明 8 项自检，就必须触发本 Skill。它会给出 Gate A-H 的量化检查清单（含一票否决项）、Gate G 八角笼 8 项 + Gate H 发布说明 8 项的具体清单、常见失败模式与修正动作，并提供 60 分钟评审会议模板与 2 小时反思模板。
---

# 全员评审 / 质量门槛协议

## 使用方式

- 作为 [`aigc-drama-pipeline`](../aigc-drama-pipeline/SKILL.md) 的质量中枢：每个 Gate 结束都按此协议开评审会
- 任何争议按"事实→补证据；审美→A/B 小样；成本→难度分级替代"收敛

---

## 一、Gate A-H 检查清单（一票否决项 = 必须 REWORK，不能 PASS with FIX）

### Gate A · 市场调研

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | 套路库 ≥ 6 张 | ✅ |
| 2 | 选题方向 ≥ 3 个 | ✅ |
| 3 | 每方向能映射到 ≥1 个对标剧 | |
| 4 | 不虚构数据 | ✅ |

### Gate B · 系列 Bible

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | 世界观 4 行内说清 | |
| 2 | 长版主线集数 ≥ 推荐 N（默认 20） | ✅ |
| 3 | 主角必含"公众-私下对比"段 | ✅ |
| 4 | 配角分类 ≥ 6 类 | |
| 5 | 主角视觉/声音/行为锚点齐全 | ✅ |

### Gate B+ · 节奏精简（**最容易跳的步骤**）

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | 长版每集"反转/欲感/爽点"列表完整 | ✅ |
| 2 | 压缩映射明确（原 N → 最终集数）| ✅ |
| 3 | 13 集（或最终集数）任一集都有 ≥2 项核心 beat | ✅ |
| 4 | 标注绝不能丢的高保留度点 | ✅ |
| 5 | 收官集含"全剧情绪闭环" | ✅ |

### Gate C · 单集剧情

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | 一句话 logline 写明 | |
| 2 | beats 8-12 条 | ✅ |
| 3 | 每条 beat 带 `[爽][欲][梗]` 标签 | |
| 4 | 开场 hook 在 0-3s | ✅ |
| 5 | 结尾钩子指向下一集 | ✅ |
| 6 | 每集至少 3 条 `[爽]` 标签 | |

### Gate D · 角色包

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | A1-characters.json 字段齐全（≥6 字段）| ✅ |
| 2 | 主角必含 `private_persona` | ✅ |
| 3 | 角色视觉锚点 ≥ 5 项 | ✅ |
| 4 | Seedream 参考图就位 | ✅ |
| 5 | 关系图/三角清晰 | |

### Gate E · 分镜（**最关键的一步**）

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | A2-locations.json 5 字段齐全 | ✅ |
| 2 | EPxx-B-clips.json 三段 × 15s | ✅ |
| 3 | 三段 energy_curve 有对比 | ✅ |
| 4 | Seg3 是截断式 cliffhanger | ✅ |
| 5 | EPxx-CD-storyboard.json 每段 5 panel | ✅ |
| 6 | 每 panel 7 字段齐全 | ✅ |
| 7 | 每段 screenplay.content[] 完整 | ✅ |

### Gate F · 首帧

| # | 检查项 | 一票否决？ |
|---|--------|:---------:|
| 1 | 全剧风格锁定块只有一份 | ✅ |
| 2 | 角色视觉锚点 ≥ 5 项 | ✅ |
| 3 | 封面构图 ≥ 3 套 | |
| 4 | 每集首帧 prompt 套用统一锁定块 | ✅ |

### Gate G · 视频 Prompt（八角笼 8 项 = 8 个一票否决项）

| # | 八角笼检查项 |
|---|------|
| 1 | 开头 3s 钩子（首帧人脸 + 3s 内冲突/接触） |
| 2 | 节奏密度（≥2 条音频事件，无 >3s 静默） |
| 3 | 情绪顶点（每段至少 1 个明确情绪 beat） |
| 4 | 服装具体（颜色+款式，无模糊词） |
| 5 | 服装锁定声明（总述里有） |
| 6 | 时长正确（15s + 时间戳一致 + 段标题对） |
| 7 | 引用完整（每段每角色都有 `@图片N`） |
| 8 | 禁止句（末尾"禁止：任何字幕、水印、logo、品牌标志"） |

**额外铁律**：
- 全部说出来，禁止 V.O.（ ✅ 一票否决）
- 安全词扫描通过（[`safety-wordlist.md`](../aigc-drama-pipeline/references/safety-wordlist.md) ✅ 一票否决）

详细规则：[`aigc-drama-pipeline/references/seedance-prompt-rules.md`](../aigc-drama-pipeline/references/seedance-prompt-rules.md)

### Gate H · 发布包（8 项自检 = 8 个一票否决项）

| # | 自检项 |
|---|------|
| 1 | 标题是一句钩子（不叠加，不剧情清单）|
| 2 | 标题字数 ≤18 字（含系列名 ≤25）|
| 3 | 备选标题换维度 |
| 4 | One-liner 独立可立 |
| 5 | Pin 用三选一（悬念/情绪/导流），不叠加 |
| 6 | Pin 含导流元素 |
| 7 | 跨集回复模块存在 |
| 8 | 跨集回复换风格 |

详细规则：[`aigc-drama-pipeline/references/publish-pack-rules.md`](../aigc-drama-pipeline/references/publish-pack-rules.md)

---

## 二、评审结论口径（四选一）

| 结论 | 含义 | 后续动作 |
|------|------|---------|
| **PASS** | 全部检查项通过 | 直接进下一 Gate |
| **PASS with FIX** | 非一票否决项有问题，但不阻塞 | 边推进边修，限时回头补 |
| **REWORK** | 一票否决项命中 | 不进下一 Gate，本 Gate 重做 |
| **KILL** | 方向性错误（题材/合规/版权根本问题） | 退回 Gate A-B 重新立项 |

---

## 三、60 分钟评审会议模板

| 时段 | 内容 |
|------|------|
| 3' | 目标对齐（过哪道 Gate，门槛是什么） |
| 7' | 静默阅读（所有人独立读完产出物，不打断）|
| 20' | 找问题（**只提问题不提方案**；每人 ≤3 条；必须指向上面的 Gate 检查清单条目） |
| 20' | 出方案（只解决 Top3；动作/负责人/截止/验收） |
| 7' | 定结论（PASS / PASS with FIX / REWORK / KILL） |
| 3' | 复利沉淀（新增/修订 1 条清单或模板，更新到对应 references/*.md） |

---

## 四、2 小时反思机制

每 2 小时一个产出周期：
- **90' 产出**
- **10' 自检**（用上面的 Gate 检查清单跑一遍当前 Gate）
- **20' 反思会**

反思会必须产出：
1. **Top 1-3 问题的根因**（5 Why 分析，不是表象）
2. **修正动作**（具体到文件/行号/责任人）
3. **下一周期计划**

---

## 五、争议收敛口径

| 争议类型 | 收敛办法 |
|---------|---------|
| 事实争议（数据/规则）| 补证据，引用 references/*.md 条款 |
| 审美争议（哪个版本好）| A/B 出小样，用数据决（点击率/完播率）|
| 成本争议（要不要做）| 难度分级替代（A 方案做不了→B方案，B 做不了→C方案）|
| 节奏争议（要不要现在做）| 拉到 Gate 流程里确认是否阻塞下一 Gate |

---

## 六、与其他 Skill 的关系

- 本 Skill 是 [`aigc-drama-pipeline`](../aigc-drama-pipeline/SKILL.md) 的"质量中枢"
- Gate G 八角笼细则在 [`seedance-prompt-rules.md`](../aigc-drama-pipeline/references/seedance-prompt-rules.md)
- Gate H 8 项自检细则在 [`publish-pack-rules.md`](../aigc-drama-pipeline/references/publish-pack-rules.md)
- 安全合规规则在 [`aigc-drama-compliance-tiktok`](../aigc-drama-compliance-tiktok/SKILL.md) + [`safety-wordlist.md`](../aigc-drama-pipeline/references/safety-wordlist.md)

---

## 七、参考

完整版项目化协议（含本剧定制的检查项）写入：`aigc-video/<series>/_meta/review-protocol.md`。
