# Review Protocol · 评审协议

> 简化版导引。完整协议（Gate A-H 检查清单 + 60 分钟会议模板 + 2 小时反思机制）请直接读：
>
> **[`~/.openclaw/skills/aigc-drama-review-protocol/SKILL.md`](../../aigc-drama-review-protocol/SKILL.md)**

---

## 速查

### 评审结论四选一
- **PASS** —— 全部检查项通过
- **PASS with FIX** —— 非一票否决项有问题，限时回头补
- **REWORK** —— 一票否决项命中，本 Gate 重做
- **KILL** —— 方向性错误，退回 Gate A-B

### 各 Gate 一票否决项汇总

| Gate | 一票否决（命中即 REWORK）|
|------|-------------------------|
| A | 套路库 < 6 / 选题方向 < 3 |
| B | 主角缺"公众-私下对比" / 主线缺"反转/欲感/爽点"标签 |
| B+ | 13 集任意一集没有"反转/欲感/爽点"中的 ≥2 项 |
| C | 单集 beats < 8 / 开场 hook 不在 0-3s / 结尾未指向下一集 |
| D | A1-characters.json 缺 CHARxx 命名 / 角色视觉锚点 < 5 项 |
| E | 缺 A2-locations.json / B-clips Seg3 无 cliffhanger / CD-storyboard panel < 5 |
| F | 首帧 prompt 缺统一画风锁定块 / 封面构图 < 3 套 |
| **G** | **八角笼 8 项任意一项不通过 / 命中安全词表 / 含 V.O. 内心独白** |
| **H** | **标题为 A+B+C 事件清单 / Pin 评论是剧情列表 / 缺跨集回复模块** |

### 60 分钟会议模板速查

```
3'  目标对齐
7'  静默阅读
20' 找问题（只提问题不提方案，每人≤3条，必指清单条目）
20' 出方案（只解 Top3）
7'  定结论
3'  复利沉淀（更新对应 references/*.md）
```

### 2 小时反思周期

```
90' 产出 → 10' 自检 → 20' 反思会
```

---

## 项目化副本

把以上内容定制后写入：`aigc-video/<series>/_meta/review-protocol.md`。
