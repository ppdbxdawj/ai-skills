# EP{{EP}}｜{{TITLE}}

> 单集总模板 · 含 Gate C/E/F/G/H 全部输出物的索引与摘要
>
> 完整产出物见各分文件，本文是入口。

---

## 0. 元信息

| 字段 | 值 |
|------|---|
| 集号 | EP{{EP}} |
| 标题 | {{TITLE}} |
| 集长 | 60s（3 段 × 15s） |
| 拍摄地（LOC）| {{LOCATIONS}} |
| 出场角色（CHAR）| {{CHARACTERS}} |
| 进度 | Gate C ✅ / Gate E ✅ / Gate F ✅ / Gate G ✅ / Gate H ⏳ |

---

## 1. Gate C · 单集 Beats

> 完整文件：`outputs/EP{{EP}}-{{TITLE}}/episode-beats.md`

### 1.1 一句话 Logline
{{ONE_LINE}}

### 1.2 开场 0-3 秒 Hook
{{HOOK}}

### 1.3 60s 四段式结构（时间轴）
- A 0-15s（Seg1）：{{A}}
- B 15-30s（Seg2）：{{B}}
- C 30-45s（Seg3）：{{C}}
- D 45-60s（如有）：{{D}}

### 1.4 Beat Sheet（8-12 条｜每条标注 `[爽][欲][梗]`）
1. {{BEAT_1}}
2. {{BEAT_2}}
3. ...

### 1.5 结尾钩子（指向下一集）
{{CLIFFHANGER}}

---

## 2. Gate D · 角色与场景引用

- 角色：从 `storyboard/A1-characters.json` 引用 → {{CHARACTERS}}
- 场景：从 `storyboard/A2-locations.json` 引用 → {{LOCATIONS}}
- 道具：{{PROPS}}

---

## 3. Gate E · 分镜（B-clips → CD-storyboard）

> 完整文件：
> - `storyboard/EP{{EP}}-B-clips.json`（3 段 × 15s）
> - `storyboard/EP{{EP}}-CD-storyboard.json`（每段 5 panel × 3s）

### 3.1 B-clips 摘要

| Seg | clip_id | summary | energy_curve |
|-----|---------|---------|---------------|
| Seg1 | EP{{EP}}-C1 | {{C1_SUMMARY}} | {{C1_ENERGY}} |
| Seg2 | EP{{EP}}-C2 | {{C2_SUMMARY}} | {{C2_ENERGY}} |
| Seg3 | EP{{EP}}-C3 | {{C3_SUMMARY}} | {{C3_ENERGY}} |

### 3.2 CD-storyboard 完成度
- [ ] 每段 5 panel 齐
- [ ] 每段 screenplay.content[] 齐
- [ ] 每 panel 7 字段齐

---

## 4. Gate F · 首帧 / 封面 Prompt

> 完整文件：`outputs/EP{{EP}}-{{TITLE}}/keyframe.md`

- 正向 prompt：{{KEYFRAME_PROMPT}}
- 负面 prompt：{{NEGATIVE_PROMPT}}
- 套用风格锁定块（见 `_meta/style-lock.md`）

---

## 5. Gate G · 视频 Prompts（逐段 Seedance）

> 完整文件：
> - `outputs/EP{{EP}}-{{TITLE}}/prompt-Seedance提示词-v5-Seg1.md`
> - `outputs/EP{{EP}}-{{TITLE}}/prompt-Seedance提示词-v5-Seg2.md`
> - `outputs/EP{{EP}}-{{TITLE}}/prompt-Seedance提示词-v5-Seg3.md`

### 5.1 八角笼自检结果

| Seg | 1 钩子 | 2 节奏 | 3 顶点 | 4 服装 | 5 锁定 | 6 时长 | 7 引用 | 8 禁止 | 安全词 |
|-----|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Seg1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Seg2 | | | | | | | | | |
| Seg3 | | | | | | | | | |

8/8 才能进 Gate H。

---

## 6. Gate H · TikTok / 抖音 发布包

> 完整文件：`outputs/EP{{EP}}-{{TITLE}}/EP{{EP}}-抖音发布说明-八角笼与系列导流.md`

### 6.1 标题
- 主标题：{{T1}}
- 备选标题（换维度）：{{T2}}

### 6.2 简介（Caption）
{{CAPTION}}

### 6.3 置顶评论（Pin）
{{PIN}}

### 6.4 标签
{{TAGS}}

### 6.5 跨集回复（发布后立即去 EP{{EP-1}} 置顶下回复）
- 候选 1：{{REPLY_1}}
- 候选 2：{{REPLY_2}}

### 6.6 8 项自检
- [ ] 1. 标题一句钩子
- [ ] 2. 标题字数 ≤25 字
- [ ] 3. 备选标题换维度
- [ ] 4. One-liner 独立可立
- [ ] 5. Pin 三选一
- [ ] 6. Pin 含导流
- [ ] 7. 跨集回复模块存在
- [ ] 8. 跨集回复换风格

8/8 才能发布。

---

## 7. 合规自检（一票否决）

- [ ] 未命中血腥/霸凌/未成年/权力胁迫换亲密等一票否决项
- [ ] 欲感仅为暧昧张力，不含性行为推进
- [ ] 任何亲密戏都有清晰的双方同意表达
- [ ] 标题/封面承诺与内容一致
- [ ] 安全词扫描（`safety-wordlist.md` 第 4.2 节）已跑且通过

---

## 8. 发布操作时序

```
T+0    视频发布
T+1min 自评 + 置顶 Pin
T+5min 去 EP{{EP-1}} 置顶下发跨集回复
T+30min 检查首批数据
T+1h   按数据决定是否补一条跨集回复
```

---

## 9. 引用文件清单

- Gate C: `outputs/EP{{EP}}-{{TITLE}}/episode-beats.md`
- Gate E: `storyboard/EP{{EP}}-B-clips.json` + `storyboard/EP{{EP}}-CD-storyboard.json`
- Gate F: `outputs/EP{{EP}}-{{TITLE}}/keyframe.md`
- Gate G: `outputs/EP{{EP}}-{{TITLE}}/prompt-Seedance提示词-v5-Seg{1,2,3}.md`
- Gate H: `outputs/EP{{EP}}-{{TITLE}}/EP{{EP}}-抖音发布说明-八角笼与系列导流.md`
