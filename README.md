# 弹珠游戏（本地运行）

一个使用 HTML5 Canvas 与 Matter.js 物理引擎实现的弹珠游戏示例。

- 开始：随机生成赔率（2/4/6/8/10）与红色目标出口（12 个出口中的一个）
- 击发：从底部向上发射弹珠，穿过顶部进入 5 层栅格（每层 10 个柱子）中随机碰撞下落，最终进入某个出口
- 命中红色出口：显示奖励提示

## 运行方式

无需构建，直接打开 `index.html` 即可。若浏览器安全策略阻止本地加载，可使用任意本地静态服务器。

### 用 Python 一行命令启动本地服务器（任选其一）

```bash
# Python 3
cd /workspace && python3 -m http.server 8000
# 访问： http://localhost:8000/
```

或使用 Node（若环境有）：

```bash
cd /workspace && npx serve . -l 8000 --single --yes
# 访问： http://localhost:8000/
```

## 操作说明

- 点击“开始”：生成本轮赔率并高亮目标出口（红色）
- 点击“击发”：发射弹珠；弹珠穿过顶部后进入栅格随机碰撞，再落入出口
- 命中红色出口：显示“命中红色出口！奖励 xN”

## 结构

- `index.html`：页面骨架、引入 Matter.js 与脚本
- `styles.css`：样式与布局
- `main.js`：游戏逻辑、物理世界、UI 交互

## 调参

在 `main.js` 中可按需调整：
- 重力：`world.gravity.y`
- 钉子大小、位置、层数：`NUM_ROWS`、`PEGS_PER_ROW`、半径等
- 发射速度、弹性与摩擦：`Body.setVelocity` 与球体属性

## 许可证

MIT