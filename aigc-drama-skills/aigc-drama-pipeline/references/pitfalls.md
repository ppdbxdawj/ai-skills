# Pitfalls · 13 集 `$5B BOYFRIEND` 踩坑 → 规则对应表

> 每一条都是被 Seedance 拒过 / 被审核拦过 / 被算法划走才形成的。下次新剧直接复用，不用再付学费。

---

## A · Gate G（视频 prompt）相关

| 踩过的坑 | 形成的规则 | 文件位置 |
|----------|-----------|----------|
| V.O. 内心独白 → AI 嘴型乱画，闭嘴说话穿帮 | 全部说出来，禁止 V.O. | [`seedance-prompt-rules.md`](seedance-prompt-rules.md) D.3 |
| 提示词写「高潮」被审核拦 | 替换为情绪顶点/爆发帧/关键帧 | [`safety-wordlist.md`](safety-wordlist.md) B.1 |
| 「婚礼/神父/誓言」组合被同性婚姻语义整体拒 | 婚礼词全替换为"庆典" + 删神父 | [`safety-wordlist.md`](safety-wordlist.md) B.2 |
| 应援视频截图触发版权拦截 | 应援场景全部去品牌化 | [`safety-wordlist.md`](safety-wordlist.md) A |
| 服装写「晚宴正装」AI 漂移每帧不一样 | 服装必须含颜色+款式 | [`seedance-prompt-rules.md`](seedance-prompt-rules.md) E |
| 段落首帧空镜，前 3s 没人脸划走 | 八角笼 #1 开头 3s 钩子 | [`seedance-prompt-rules.md`](seedance-prompt-rules.md) B |
| 提示词缺 @图片N → AI 用陌生人脸 | 八角笼 #7 引用完整 | [`seedance-prompt-rules.md`](seedance-prompt-rules.md) B |
| 末尾不禁字幕 → AI 幻觉出字幕和品牌 | 八角笼 #8 强制禁止句 | [`seedance-prompt-rules.md`](seedance-prompt-rules.md) D.5 |
| Seg2 时间戳写 0-15s（其实应该 15-30s） | 八角笼 #6 时长正确 | [`seedance-prompt-rules.md`](seedance-prompt-rules.md) B |

---

## B · Gate E（分镜）相关

| 踩过的坑 | 形成的规则 | 文件位置 |
|----------|-----------|----------|
| EP09 跳过 CD-storyboard 直接写提示词，钩子和原剧本对不上要返工 | Gate E 一票否决：必须出完整 CD-storyboard | [`storyboard-industrialization.md`](storyboard-industrialization.md) C/D |
| 三段情绪曲线没对比，看着平 | B-clips 三段必须有 energy_curve 对比 | [`storyboard-industrialization.md`](storyboard-industrialization.md) B.3 |
| Seg3 没 cliffhanger，下集没人想看 | B-clips Seg3 必须截断式 | [`storyboard-industrialization.md`](storyboard-industrialization.md) B.3 |
| 角色 ID 集间不一致（CHAR01 / CHAR1 / Kace 混用）| A1-characters.json 全剧锁定 | [`storyboard-industrialization.md`](storyboard-industrialization.md) A.1 |
| 场景描述每集不一样导致风格漂 | A2-locations.json 全剧锁定 | [`storyboard-industrialization.md`](storyboard-industrialization.md) A.2 |

---

## C · Gate H（发布包）相关

| 踩过的坑 | 形成的规则 | 文件位置 |
|----------|-----------|----------|
| 标题写「100亿收购+他只是为了保护我｜EP10」没人点 | 单钩规则 + 优先级阶梯 | [`publish-pack-rules.md`](publish-pack-rules.md) 一 |
| 置顶评论写 A→B→C→D 剧情清单像营销号 | Pin 三选一不叠加 | [`publish-pack-rules.md`](publish-pack-rules.md) 四 |
| 跨集回复忘写 → 上一集尾流量浪费 | Gate H 一票否决：跨集回复必填 | [`publish-pack-rules.md`](publish-pack-rules.md) 五 |
| 跨集回复每集都用同一个模板 → 像 bot | 三风格轮换（预言/情绪/细节）| [`publish-pack-rules.md`](publish-pack-rules.md) 5.4 |
| 跨集回复发太晚 → 错过算法热度窗口 | T+5min 必发 | [`publish-pack-rules.md`](publish-pack-rules.md) 八 |
| 「主页搜 EPxx」太像广告 | 改 `主页有 ep10👆` | [`publish-pack-rules.md`](publish-pack-rules.md) 5.4 |
| 标题字数超 25 字被截断 | 中文 ≤18 字（含系列名 ≤25）| [`publish-pack-rules.md`](publish-pack-rules.md) 1.3 |

---

## D · Gate B/B+（剧本）相关

| 踩过的坑 | 形成的规则 | 文件位置 |
|----------|-----------|----------|
| 20 集按传统短剧节奏做，1 分钟一集装不下 | Gate B+ 必做：20 → 13 集压缩 | [`team-prompts.md`](team-prompts.md) 3.2 |
| EP09 没严格按精简方案做，"官宣顺序"和原剧本对不上要重做 | Gate B+ 通过后所有后续不允许"灵感发挥" | [`team-prompts.md`](team-prompts.md) 3.2 |
| 主角人设缺"公众-私下对比"，亲密戏没张力 | Gate D 一票否决：主角必含 public_persona + private_persona | [`storyboard-industrialization.md`](storyboard-industrialization.md) A.1 |
| 单集只有"爽点"没"欲感"或"反转"，平 | Gate B/C 一票否决：每集 ≥2 项 | [`team-prompts.md`](team-prompts.md) 3.1 |

---

## E · 流程协作相关

| 踩过的坑 | 形成的规则 |
|----------|-----------|
| Gate 没评审就推进，下游发现问题 → 上游重做 | 全员评审会议必须开（[`review-protocol.md`](review-protocol.md)）|
| Gate 之间产出物路径不一致，下游找不到 | 落盘路径强制规范（见 SKILL.md 五）|
| 第 N 集发现规则不对，前 N-1 集要不要补？ | 写进 change-log，新规则只对未来集生效（除非用户明确要补）|
| 每次失败都口头说"以后注意"，重复踩 | 必须写进 risk-register + 更新对应 references/*.md |

---

## F · 时间线复盘（13 集真实经验）

| 时间节点 | 事件 | 教训 |
|---------|------|------|
| EP01-EP05 | 写得很快，规则少 | 没问题，但已埋雷 |
| EP06 | V.O. 内心独白嘴型穿帮 | → Rule 7 全部说出来 |
| EP07 Seg2 | `OutputVideoSensitiveContentDetected.PolicyViolation - copyright` | → safety-wordlist A 版权表 |
| EP08-EP09 | 跳 CD-storyboard 直接写提示词，钩子弱 | → Gate E 一票否决 CD 必做 |
| EP10-EP13 | 标题/Pin 写得不好，跨集回复忘写 | → publish-pack-rules 完整规则 + Gate H 一票否决 |
| EP13 Seg2 | `PolicyViolation` 同性婚礼语义触发 | → safety-wordlist B.2 婚礼语义重写 |

---

## G · 给新剧的建议

如果是开新短剧，**至少把这 5 件事的规则先读一遍**，能省 70% 的重做成本：

1. [`seedance-prompt-rules.md`](seedance-prompt-rules.md) D.3 V.O. 禁令 → 剧本阶段就不允许 V.O.
2. [`safety-wordlist.md`](safety-wordlist.md) B.1 + B.2 → 命名/选词阶段就避开
3. [`storyboard-industrialization.md`](storyboard-industrialization.md) C → CD-storyboard 必做
4. [`publish-pack-rules.md`](publish-pack-rules.md) 五 → 跨集回复必备
5. [`team-prompts.md`](team-prompts.md) 3.2 → Gate B+ 节奏精简必做
