# Team Prompts · 11 个职能团队的标准 Prompt + 输出标准 + 检查项

> 用法：主控根据用户需求把 `{{...}}` 占位符替换后，用 `sessions_spawn(mode=run)` 启动 subagent。
> 每个团队的"通过标准"由 [`review-protocol.md`](review-protocol.md) 的 Gate 检查清单兜底。

---

## 1) 市场调研团队（Gate A）

**目标**：为【{{平台}}】【{{时长}}】【{{题材}}】提供可复利套路库与选题方向。

**输出（Markdown，写入 `_meta/market-research.md` 和 `_meta/套路库.md`）**：
- 趋势概括（不虚构数据，引用对标账号/对标剧名）
- 套路卡 ≥ 6（每张：套路名 / 触发心理 / 典型场景 / 被滥用程度 / 适配题材）
- 系列方向 ≥ 3（每个 10 集一句话走向）

**通过标准**：
- 套路卡 ≥ 6 ✅
- 选题方向 ≥ 3 ✅
- 每个方向能映射到至少 1 个对标剧 ✅

---

## 2) 文案团队（Gate H 复用，部分 Gate C 用）

**目标**：产出可复用的开场 hook、结尾钩子、封面大字、标题模板（中英可选）。

**输出（写入 `_meta/copy-assets.md`）**：
- 开场 hook 模板池 ≥ 10
- 结尾钩子模板池 ≥ 10
- 封面大字句式池 ≥ 8
- 标题模板池 ≥ 8（按 [`publish-pack-rules.md`](publish-pack-rules.md) 的优先级阶梯：金句 > 冲突 > 悬念 > POV）

**通过标准**：
- 所有模板都是可填空的句式，不是具体台词 ✅
- 短狠口语，平台适配 ✅
- 中英两语都有 ✅

---

## 3) 剧本创意团队（Gate B / B+ / C）

**目标**：完整的系列 Bible 和单集 beats。

**统一执行引擎**：本团队默认调用 `screenwriting-master`（剧集格式），先读：
- `screenwriting-master/references/core-methodology.md`
- `screenwriting-master/references/format-series.md`

**硬要求**（沿用 `screenwriting-master` 铁律）：
- 禁心理描写、禁解释性台词、禁不可拍内容
- 先内部自检再交付

### 3.1 Gate B 输出（`_meta/series-bible.md` + `docs/主线剧情.md`）

- 世界观（4 行内：时间/空间/规则/冲突源）
- 长版主线 N 集（默认 20，按四阶段分：伪装→欲感→危机→圆满）
- 每集列三件事：**反转钩子**（这一集"等等什么"）/ **欲感名场面**（视觉记忆点）/ **炸裂爽点**（情感/资本/护短）

**通过标准**：
- 每集"反转/欲感/爽点"中至少 2 项 ✅
- 集间情绪曲线有对比，不是平铺 ✅

### 3.2 Gate B+ 输出（`docs/主线剧情-节奏Review与精简方案.md`）

- 长版每集"反转/欲感/爽点"列表
- 每个 beat 的"独立性"标注（能否合并到隔壁集）
- 压缩映射：原 N 集 → 最终集数（推荐 13）
- **绝不能丢的高保留度点**：第一次撞见 / 第一次吻 / 第一次告白 / 求婚 / 婚礼 / 收官反转

**通过标准**：
- 13 集任意一集都有 ≥2 项核心 beat ✅
- 收官集 EP-final 必含"全剧情绪闭环" ✅

### 3.3 Gate C 输出（`outputs/EPxx/episode-beats.md`）

- 一句话 logline
- 8-12 条 beats，每条带 `[爽][欲][梗]` 标签（可多选）
- 开场 0-3s hook（必须有）
- 结尾钩子（必须指向下一集更大威胁/秘密/代价）

**通过标准**：
- beats 8-12 条 ✅
- 开场 hook 在 0-3s ✅
- 结尾钩子指向下一集 ✅
- 每集有至少 3 条带 `[爽]` 标签 ✅

**附加通过标准**：
- 每条 beat 都能被摄影机拍出来（不可出现纯心理状态句） ✅

---

## 4) 角色提取团队（Gate D）

**目标**：把剧情角色结构化为可复利资产。

**输出**：
- `storyboard/A1-characters.json`（CHAR01/CHAR02/... 全剧 ID 锁定，按 [`storyboard-industrialization.md`](storyboard-industrialization.md) 规范）
- `docs/主角人设.md`：每位主角含 视觉/声音/**公众-私下对比**/行为细节/AI 生成关键词
- `docs/配角人设.md`：6 类配角 × 出场节奏 / 功能 / 视觉关键词
- 关系图（双主角关系演进 + 关键三角）
- 可复用道具/场景表

**通过标准**：
- A1-characters.json 字段齐全（name_cn / name_en / ref_image / visual_anchor 5 项）✅
- 主角必须有"公众-私下对比"段（这是后续所有亲密戏的磁性核心）✅

---

## 5) 分镜团队（Gate E）

**目标**：把每集 beats 转为工业化分镜。

**这是流水线最关键的一步，跳过 100% 后期返工。** 完整规则见 [`storyboard-industrialization.md`](storyboard-industrialization.md)。

**输出**：
- `storyboard/A2-locations.json`（LOC01/LOC02/... 全剧场景注册）
- `storyboard/EPxx-B-clips.json`（每集固定 3 段 × 15s）
- `storyboard/EPxx-CD-storyboard.json`（每段 5 panel × 3s）

**通过标准**：
- B-clips 三段必须有情绪曲线对比（冷峻→缠绵→骤变 等）✅
- Seg3 必须有截断式 cliffhanger，明确指向下集 ✅
- CD-storyboard 每个 panel 字段齐全（shot_type/camera_move/description/photographyPlan/actingNotes/audio）✅
- 每段 screenplay.content[] 完整（5 panel 的对白+动作汇总）✅

---

## 6) 首帧设计团队（Gate F）

**目标**：统一画风锁定包 + 每集首帧/封面 prompt。

**输出（`outputs/EPxx/keyframe.md` + `_meta/style-lock.md`）**：
- 风格锁定块（正向 + 负面词，全剧通用）
- 色板（常态 + 情绪切换）
- 镜头语法（景别/运动/节奏）
- 角色视觉锚点（每位主角 ≥5 项）
- 封面构图模板 ≥ 3 套
- 每集首帧 prompt（套用统一锁定块）

**通过标准**：
- 全剧风格锁定块只有一份，不允许每集独立 ✅
- 角色视觉锚点 ≥ 5 项 ✅
- 封面构图 ≥ 3 套 ✅
- 引用 `aigc-drama-style-lock` skill ✅

---

## 7) 视频 prompt 团队（Gate G，最容易出问题）

**目标**：把 CD-storyboard 转成可直接喂 Seedance 的提示词。

**完整规则**：[`seedance-prompt-rules.md`](seedance-prompt-rules.md)（八角笼 8 项 + 15s 节奏模板 + 服装锁定 + V.O. 禁令）

**输出（每集 3 个文件）**：
- `outputs/EPxx/prompt-Seedance提示词-v5-Seg1.md`
- `outputs/EPxx/prompt-Seedance提示词-v5-Seg2.md`
- `outputs/EPxx/prompt-Seedance提示词-v5-Seg3.md`

每个文件包含：
1. 图片上传顺序表（`@图片1=CHAR01=Kace`）
2. 服装锁定声明（颜色+款式，禁止"晚宴正装"等模糊词）
3. **代码块（提交给 Seedance 的纯净提示词）**
4. 末尾八角笼自检结果

**通过标准（八角笼 8/8）**：
1. 开头 3s 钩子（首帧人脸 + 3s 内冲突/接触）
2. 节奏密度（≥2 条音频事件，无 >3s 静默）
3. 情绪顶点（每段至少 1 个明确情绪 beat）
4. 服装具体（颜色+款式，无模糊词）
5. 服装锁定声明（总述里有）
6. 时长正确（15s + 时间戳一致 + 段标题对，Seg2 = 15-30s）
7. 引用完整（每段每角色都有 `@图片N`）
8. 禁止句（末尾"禁止：任何字幕、水印、logo、品牌标志"）

**铁律**：
- **全部说出来，禁止 V.O./内心独白**（V.O. 在 AI 模型上控不住嘴型）
- 提交前必跑安全词扫描（[`safety-wordlist.md`](safety-wordlist.md)）

---

## 8) TikTok 发布专员（Gate H）

**目标**：每集发布包 + 跨集导流闭环。

**完整规则**：[`publish-pack-rules.md`](publish-pack-rules.md)（标题钩子优先级 + Pin 三选一 + 跨集回复三风格 + 8 项自检）

**输出**：`outputs/EPxx/EPxx-抖音发布说明-八角笼与系列导流.md`，包含 8 节：

1. 一句话剧情 + 4 条本集看点
2. 八角笼策略表（5 心理动机 × 本集如何踩 × 发布时如何强化）
3. 下一集钩子（简介末段 + 评论区文字）
4. 原创 + 关注追更（系列名/封面/导流）
5. 标题/简介/置顶评论/标签（ready-to-paste）
6. 6 项发布前 checklist
7. TikTok 英文版（One-liner + Key beats + Caption + Pinned + Hashtags）
8. **🔥 跨集回复文案（必备模块）** —— 给上一集置顶评论下的 1-2 条候选回复

**通过标准（8/8 自检）**：
1. 标题是一句钩子（不叠加，不剧情清单）
2. 标题字数 ≤18 字（含系列名 ≤25）
3. 备选标题换维度
4. One-liner 独立可立
5. Pin 用三选一（悬念/情绪/导流）不叠加
6. Pin 含导流元素
7. 跨集回复模块存在
8. 跨集回复换风格（与最近几集不重复）

---

## 9) 合规团队（贯穿 Gate D-H）

**目标**：禁区清单 + 安全替代 + 发布前自检。

**完整规则**：
- 框架：[`~/.openclaw/skills/aigc-drama-compliance-tiktok/SKILL.md`](../../aigc-drama-compliance-tiktok/SKILL.md)
- 具体词表：[`safety-wordlist.md`](safety-wordlist.md)

**输出**（写入 `_meta/compliance.md`）：
- 本剧禁区清单（基于题材定制）
- 欲感擦边边界（暧昧张力 vs 性行为推进）
- 高风险场景安全处理（制服/权力关系/卧室/浴室）
- 发布前自检表（含一票否决项）
- **每集 Gate G 提交前的安全词扫描记录**（命中→替换→放行）

**通过标准**：
- 命中一票否决项 → REWORK，不是 PASS with FIX ✅
- 安全词扫描必跑且记录 ✅

---

## 10) 梗策划团队（贯穿 Gate C-G）

**目标**：可投放的结构梗素材库。

**输出**（写入 `_meta/meme-bank.md`）：
- 10 类结构梗框架（反转梗/对比梗/谐音梗/接梗梗/转场梗 等）
- 每类 5 条可直接用句式（共 ≥50）
- 60s 四段式投放位置映射（A段开场 / B段发展 / C段转折 / D段钩子）

**通过标准**：
- 不复刻任何侵权原句（只用结构/语气/反应模式）✅
- 每类 ≥5 条 ✅

---

## 11) 流程工程团队（一次性配置，全程兜底）

**目标**：失败兜底机制。

**输出**：[`retry-policy.md`](retry-policy.md) 的项目化副本（写入 `_meta/retry-policy.md`）+ `_meta/change-log.md` + `_meta/risk-register.md`

**通过标准**：
- 三份文件齐 ✅
- 每个 Gate 失败后都有变更记录 ✅
