# Skill 迁移对照表

**背景**：2026-03 把 4 件套（aigc-drama-pipeline + style-lock + compliance-tiktok + review-protocol）正式接入 aigc-drama agent 后，workspace-aigc-drama/skills/ 下的部分老 skill 与新四件套**功能重叠**，需要按下表治理。

观察期：**2026-03-06 → 2026-03-13**（7 天），到期未发现退路问题就物理删除。

---

## 治理矩阵

| 老 skill | 路径 | 处置 | 迁去哪里 | 备注 |
|---|---|:---:|---|---|
| `novel-to-drama-script` | `workspace-aigc-drama/skills/novel-to-drama-script/` | 🔴 **DEPRECATED** | `aigc-drama-pipeline` Gate A/B/B+ | 老 skill 只能输出"剧本草稿"，新 pipeline 还包含市场调研 → 节奏精简 → 主线锁定 |
| `storyboard-creation` | `workspace-aigc-drama/skills/storyboard-creation/` | 🔴 **DEPRECATED** | `aigc-drama-pipeline/references/storyboard-industrialization.md` | 老 skill 只懂通用分镜语法，不懂 B-clips（3×15s）+ CD-storyboard（5 panels）双层结构 |
| `video-production` | `workspace-aigc-drama/skills/video-production/` | 🟡 **改 scope** | 仅 `aigc-video/` 单条创作 | 短剧任务**禁用**，避免与 Gate G 冲突 |
| `video-edit` | `workspace-aigc-drama/skills/video-edit/` | 🟢 **保留** | — | 后期工具，正交不冲突 |
| `video-to-prompt-pipeline` | `workspace-aigc-drama/skills/video-to-prompt-pipeline/` | 🟢 **保留** | — | 视频反推 prompt，与 Gate G 是反向流程 |
| `whisper-transcriber` | `workspace-aigc-drama/skills/whisper-transcriber/` | 🟢 **保留** | — | 字幕/转写，正交 |

---

## 为什么要这么做

### 不治理的代价（真实风险）

agent 看到一句"帮我写 EP14 的分镜"，可能会**同时命中**：
- ✅ `aigc-drama-pipeline`（新四件套，Gate E 工业化分镜）
- ❌ `storyboard-creation`（老 skill，触发词 `storyboard / 分镜 / shot list` 全在里面）

skill 加载顺序不可预测，agent 可能：
- 用老 skill 出一份"通用分镜"，不带 B-clips/CD-storyboard 结构
- 没经过 A1-characters.json / A2-locations.json 锁定
- 直接进 Gate G → 又触发 13 集里那批"细节丢失 80%"的老问题

### DEPRECATED 标记的工作机制

把老 skill 的 `description` 字段改成 `[DEPRECATED ...]`+ 加 `deprecated: true` 字段后：
- 触发词描述明确说"短剧任务请勿触发本 skill"
- agent 即便误命中，读到 description 第一行也会立即放弃，转去读 replacement 路径
- 文件没删，可以 7 天内随时回滚

---

## 检查命令

```bash
# 看哪些 skill 还在 deprecated 状态
rg -l "deprecated: true" ~/.openclaw/workspace-aigc-drama/skills/

# 看 7 天观察期到没到
rg "remove_after:" ~/.openclaw/workspace-aigc-drama/skills/
```

观察期内每 2 天自检一次：有没有 session 误触发了 deprecated skill？如果没有，到期物理删除。

---

## 删除清单（2026-03-13 后执行）

```bash
trash ~/.openclaw/workspace-aigc-drama/skills/novel-to-drama-script
trash ~/.openclaw/workspace-aigc-drama/skills/storyboard-creation
```

`video-production` 只改 scope，不删。
