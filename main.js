(function () {
  const {
    Engine,
    Render,
    Runner,
    Composite,
    Composites,
    Constraint,
    Bodies,
    Body,
    Events,
    Vector,
    World,
  } = Matter;

  // 基础尺寸
  const STAGE_WIDTH = 840;
  const STAGE_HEIGHT = 1080;

  // 物理世界
  const engine = Engine.create();
  const world = engine.world;
  world.gravity.y = 1.05; // 更接近真实下落感

  // 渲染器
  const gameEl = document.getElementById('game');
  const render = Render.create({
    element: gameEl,
    engine,
    options: {
      width: STAGE_WIDTH,
      height: STAGE_HEIGHT,
      background: '#0c111b',
      wireframes: false,
      pixelRatio: window.devicePixelRatio || 1,
    },
  });
  Render.run(render);

  const runner = Runner.create();
  Runner.run(runner, engine);

  // UI 控件
  const startBtn = document.getElementById('startBtn');
  const fireBtn = document.getElementById('fireBtn');
  const payoutLabel = document.getElementById('payoutLabel');
  const targetExitLabel = document.getElementById('targetExitLabel');
  const resultLabel = document.getElementById('resultLabel');

  // 区域定义
  const MARGIN = 24;
  const BOARD_LEFT = MARGIN;
  const BOARD_RIGHT = STAGE_WIDTH - MARGIN;
  const BOARD_TOP = MARGIN;
  const BOARD_BOTTOM = STAGE_HEIGHT - MARGIN;

  const GRID_TOP_Y = 180; // 栅格顶部 y
  const GRID_BOTTOM_Y = STAGE_HEIGHT - 180; // 栅格底部 y（出口上方）
  const GRID_LEFT_X = BOARD_LEFT + 40;
  const GRID_RIGHT_X = BOARD_RIGHT - 40;
  const GRID_WIDTH = GRID_RIGHT_X - GRID_LEFT_X;

  const NUM_ROWS = 5;
  const PEGS_PER_ROW = 10;
  const NUM_EXITS = 12;

  // 发射通道（底部到顶部的竖直通道）
  const CHANNEL_WIDTH = 70;
  const CHANNEL_CENTER_X = STAGE_WIDTH / 2;
  const CHANNEL_LEFT_X = CHANNEL_CENTER_X - CHANNEL_WIDTH / 2;
  const CHANNEL_RIGHT_X = CHANNEL_CENTER_X + CHANNEL_WIDTH / 2;
  const CHANNEL_TOP_STOP = GRID_TOP_Y - 12; // 通道上边界（略低于栅格顶部，留出出口）

  // 状态
  const PAYOUT_CHOICES = [2, 4, 6, 8, 10];
  let currentPayout = null;
  let targetExitIndex = null; // 0..11
  let ballBody = null;
  let roundActive = false; // 按下开始后才允许击发

  // 边界墙
  const wallThickness = 40;
  const walls = [
    Bodies.rectangle(BOARD_LEFT - wallThickness / 2, STAGE_HEIGHT / 2, wallThickness, STAGE_HEIGHT, {
      isStatic: true,
      render: { fillStyle: '#0e1624' },
    }),
    Bodies.rectangle(BOARD_RIGHT + wallThickness / 2, STAGE_HEIGHT / 2, wallThickness, STAGE_HEIGHT, {
      isStatic: true,
      render: { fillStyle: '#0e1624' },
    }),
    Bodies.rectangle(STAGE_WIDTH / 2, BOARD_BOTTOM + wallThickness / 2, STAGE_WIDTH, wallThickness, {
      isStatic: true,
      render: { fillStyle: '#0e1624' },
    }),
    // 顶部稍微留空，不放墙
  ];
  Composite.add(world, walls);

  // 发射通道两侧墙体（只到达栅格顶部下方，留出口）
  const CHANNEL_CENTER_Y = (BOARD_BOTTOM + CHANNEL_TOP_STOP) / 2;
  const CHANNEL_HEIGHT = (BOARD_BOTTOM - CHANNEL_TOP_STOP);
  const channelWalls = [
    Bodies.rectangle(CHANNEL_LEFT_X, CHANNEL_CENTER_Y, 10, CHANNEL_HEIGHT, {
      isStatic: true,
      render: { fillStyle: '#152033' },
    }),
    Bodies.rectangle(CHANNEL_RIGHT_X, CHANNEL_CENTER_Y, 10, CHANNEL_HEIGHT, {
      isStatic: true,
      render: { fillStyle: '#152033' },
    }),
  ];
  Composite.add(world, channelWalls);

  // 通道顶部取消挡板，直接连入栅格

  // 栅格的钉子
  const pegs = [];
  const rowSpacing = (GRID_BOTTOM_Y - GRID_TOP_Y) / (NUM_ROWS - 1);
  const colSpacing = GRID_WIDTH / (PEGS_PER_ROW - 1);
  for (let row = 0; row < NUM_ROWS; row++) {
    const y = GRID_TOP_Y + row * rowSpacing;
    for (let col = 0; col < PEGS_PER_ROW; col++) {
      // 锯齿排列（错位），增加随机性
      let x = GRID_LEFT_X + col * colSpacing;
      if (row % 2 === 1) x += colSpacing / 2;
      // 限制在左右边界内
      if (x < GRID_LEFT_X + 20 || x > GRID_RIGHT_X - 20) continue;
      const peg = Bodies.circle(x, y, 8, {
        isStatic: true,
        restitution: 0.95,
        label: 'peg',
        render: {
          fillStyle: '#1f4a7a',
          strokeStyle: '#2a6db2',
          lineWidth: 1,
        },
      });
      pegs.push(peg);
    }
  }
  Composite.add(world, pegs);

  // 出口：12 个槽位 + 传感器
  const exitWidth = (GRID_RIGHT_X - GRID_LEFT_X) / NUM_EXITS;
  const exitTopY = GRID_BOTTOM_Y + 10; // 出口平台顶部
  const exitHeight = 20;
  const exitFloorY = exitTopY + exitHeight / 2;
  const exitSensorY = exitTopY - 24;

  const exitFloors = [];
  const exitSensors = [];
  for (let i = 0; i < NUM_EXITS; i++) {
    const centerX = GRID_LEFT_X + i * exitWidth + exitWidth / 2;
    const floor = Bodies.rectangle(centerX, exitFloorY, exitWidth - 6, exitHeight, {
      isStatic: true,
      label: `exit-floor-${i}`,
      render: { fillStyle: '#1b2a42' },
    });
    exitFloors.push(floor);

    const sensor = Bodies.rectangle(centerX, exitSensorY, exitWidth - 10, 16, {
      isStatic: true,
      isSensor: true,
      label: `exit-sensor-${i}`,
      render: {
        fillStyle: 'rgba(255,255,255,0.04)',
      },
    });
    exitSensors.push(sensor);
  }
  Composite.add(world, exitFloors);
  Composite.add(world, exitSensors);

  // 出口之间的分隔竖墙
  const dividers = [];
  const dividerHeight = 140;
  const dividerTopY = exitTopY - dividerHeight / 2;
  for (let i = 0; i <= NUM_EXITS; i++) {
    const x = GRID_LEFT_X + i * exitWidth;
    const divider = Bodies.rectangle(x, dividerTopY, 6, dividerHeight, {
      isStatic: true,
      render: { fillStyle: '#122038' },
    });
    dividers.push(divider);
  }
  Composite.add(world, dividers);

  // 为防止弹珠穿过出口区域后继续乱跑，增加底部捕捉层
  const catchFloor = Bodies.rectangle(STAGE_WIDTH / 2, BOARD_BOTTOM - 10, STAGE_WIDTH, 20, {
    isStatic: true,
    render: { fillStyle: '#0c111b' },
  });
  Composite.add(world, catchFloor);

  // 轻微的随机性：与 peg 碰撞时给一个很小的切向力
  function addRandomNudgeOnPegCollision(event) {
    for (const pair of event.pairs) {
      const { bodyA, bodyB } = pair;
      const peg = bodyA.label === 'peg' ? bodyA : bodyB.label === 'peg' ? bodyB : null;
      const ball = bodyA.label === 'ball' ? bodyA : bodyB.label === 'ball' ? bodyB : null;
      if (peg && ball) {
        const randomAngle = (Math.random() - 0.5) * Math.PI * 0.2; // +-18°
        const randomMag = 0.001 + Math.random() * 0.0015; // 很小的力
        const force = Vector.mult({ x: Math.cos(randomAngle), y: Math.sin(randomAngle) }, randomMag);
        Body.applyForce(ball, ball.position, force);
      }
    }
  }
  Events.on(engine, 'collisionStart', addRandomNudgeOnPegCollision);

  // 处理通过出口的奖励判定
  function handleExitSensor(event) {
    if (!ballBody) return;
    for (const pair of event.pairs) {
      const { bodyA, bodyB } = pair;
      const sensor = bodyA.isSensor ? bodyA : bodyB.isSensor ? bodyB : null;
      const ball = bodyA.label === 'ball' ? bodyA : bodyB.label === 'ball' ? bodyB : null;
      if (sensor && ball && typeof sensor.label === 'string' && sensor.label.startsWith('exit-sensor-')) {
        const idx = parseInt(sensor.label.replace('exit-sensor-', ''), 10);
        // 标记结果
        const win = idx === targetExitIndex;
        resultLabel.textContent = win ? `命中红色出口！奖励 x${currentPayout}` : '未命中，祝下次好运';
        resultLabel.style.color = win ? '#ff6363' : '#9fb5cc';

        // 一次判定即可
        endRound();
      }
    }
  }
  Events.on(engine, 'collisionStart', handleExitSensor);

  function setExitHighlight(index) {
    for (let i = 0; i < exitFloors.length; i++) {
      exitFloors[i].render.fillStyle = i === index ? '#b71c1c' : '#1b2a42';
    }
  }

  function clearBall() {
    if (ballBody) {
      Composite.remove(world, ballBody);
      ballBody = null;
    }
  }

  function resetUIForNewRound() {
    resultLabel.textContent = '';
    resultLabel.style.color = '';
    payoutLabel.textContent = '-';
    targetExitLabel.textContent = '-';
    setExitHighlight(-1);
  }

  function startRound() {
    clearBall();
    resetUIForNewRound();
    currentPayout = PAYOUT_CHOICES[Math.floor(Math.random() * PAYOUT_CHOICES.length)];
    targetExitIndex = Math.floor(Math.random() * NUM_EXITS);
    payoutLabel.textContent = `x${currentPayout}`;
    targetExitLabel.textContent = String(targetExitIndex + 1);
    setExitHighlight(targetExitIndex);
    roundActive = true;
    fireBtn.disabled = false;
  }

  function endRound() {
    roundActive = false;
    fireBtn.disabled = true;
    // 让球滚动一会儿再清理
    setTimeout(() => {
      clearBall();
    }, 800);
  }

  function fireBall() {
    if (!roundActive) return;
    if (ballBody) clearBall();

    const radius = 10;
    const startY = BOARD_BOTTOM - 40 - radius;
    const startX = CHANNEL_CENTER_X + (Math.random() - 0.5) * 6; // 轻微偏移
    ballBody = Bodies.circle(startX, startY, radius, {
      restitution: 0.45,
      friction: 0.02,
      frictionAir: 0.005,
      density: 0.002,
      label: 'ball',
      render: { fillStyle: '#f0f6ff' },
    });
    Composite.add(world, ballBody);

    // 给予向上的初速度，使其到达栅格顶部附近
    Body.setVelocity(ballBody, { x: 0, y: -26 });

    // 当球接近通道顶端时，给予一个轻微随机的横向扰动，避免直线落下
    const checkInterval = setInterval(() => {
      if (!ballBody) return clearInterval(checkInterval);
      if (ballBody.position.y < CHANNEL_TOP_STOP + 10) {
        const nudgeX = (Math.random() - 0.5) * 2.2;
        Body.setVelocity(ballBody, { x: nudgeX, y: ballBody.velocity.y });
        clearInterval(checkInterval);
      }
    }, 16);

    fireBtn.disabled = true;
  }

  // 绑定事件
  startBtn.addEventListener('click', startRound);
  fireBtn.addEventListener('click', fireBall);

  // 初始状态
  resetUIForNewRound();
})();

