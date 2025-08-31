class PlinkoGame {
    constructor() {
        this.gameBoard = document.getElementById('gameBoard');
        this.gameContainer = document.getElementById('gameContainer');
        this.startBtn = document.getElementById('startBtn');
        this.fireBtn = document.getElementById('fireBtn');
        this.currentOddsElement = document.getElementById('currentOdds');
        this.scoreElement = document.getElementById('score');
        this.chipsElement = document.getElementById('chips');
        this.messageElement = document.getElementById('message');
        this.powerBar = document.getElementById('powerBar');
        
        // 模态框元素
        this.setupScreen = document.getElementById('setupScreen');
        this.betInputContainer = document.getElementById('betInputContainer');
        this.initialChipsInput = document.getElementById('initialChips');
        this.betAmountInput = document.getElementById('betAmount');
        this.currentChipsDisplay = document.getElementById('currentChipsDisplay');
        
        this.score = 0;
        this.chips = 0;
        this.currentBet = 0;
        this.currentOdds = null;
        this.winningChannels = [];
        this.ball = null;
        this.isGameActive = false;
        this.isBallInMotion = false;
        this.isCharging = false;
        this.powerLevel = 0;
        this.maxPower = 100;
        
        // 游戏配置
        this.containerWidth = 620; // 游戏容器内部宽度
        this.containerHeight = 580; // 游戏容器内部高度
        this.pegRows = 5;
        this.pegsPerRow = 10;
        this.exitChannels = 12;
        this.possibleOdds = [2, 4, 6, 8, 10];
        
        // 物理参数
        this.gravity = 0.25;
        this.friction = 0.99;
        this.bounceFactorX = 0.9; // 增加水平弹性
        this.bounceFactorY = 0.85; // 增加垂直弹性
        this.pegRadius = 4;
        this.ballRadius = 18; // 增加到原来的3倍 (6 * 3 = 18)
        
        this.initEventListeners();
        this.createGameElements();
    }

    initEventListeners() {
        // 初始筹码确认
        document.getElementById('confirmChips').addEventListener('click', () => this.confirmInitialChips());
        
        // 下注相关
        document.getElementById('confirmBet').addEventListener('click', () => this.confirmBet());
        document.getElementById('cancelBet').addEventListener('click', () => this.cancelBet());
        
        // 游戏控制
        this.startBtn.addEventListener('click', () => this.startGame());
        
        // 蓄力系统 - 鼠标事件
        this.fireBtn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.startCharging();
        });
        
        this.fireBtn.addEventListener('mouseup', (e) => {
            e.preventDefault();
            this.releaseBall();
        });
        
        this.fireBtn.addEventListener('mouseleave', (e) => {
            if (this.isCharging) {
                this.releaseBall();
            }
        });
        
        // 蓄力系统 - 触摸事件（移动设备支持）
        this.fireBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startCharging();
        });
        
        this.fireBtn.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.releaseBall();
        });
        
        // 键盘事件
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !this.fireBtn.disabled && !this.isCharging) {
                e.preventDefault();
                this.startCharging();
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && this.isCharging) {
                e.preventDefault();
                this.releaseBall();
            }
        });
    }

    confirmInitialChips() {
        const chips = parseInt(this.initialChipsInput.value);
        if (chips >= 100 && chips <= 10000) {
            this.chips = chips;
            this.chipsElement.textContent = this.chips;
            this.setupScreen.style.display = 'none';
            this.messageElement.textContent = '点击"开始游戏"开始新一轮！';
        } else {
            alert('请输入100到10000之间的筹码数量！');
        }
    }

    startGame() {
        if (this.chips < 5) {
            alert('筹码不足！最少需要5个筹码才能开始游戏。');
            return;
        }
        
        // 显示下注输入框
        this.currentChipsDisplay.textContent = this.chips;
        this.betInputContainer.style.display = 'flex';
    }

    confirmBet() {
        const betAmount = parseInt(this.betAmountInput.value);
        if (betAmount >= 5 && betAmount <= 100 && betAmount <= this.chips) {
            this.currentBet = betAmount;
            this.chips -= betAmount;
            this.chipsElement.textContent = this.chips;
            this.betInputContainer.style.display = 'none';
            
            this.isGameActive = true;
            this.startBtn.disabled = true;
            this.fireBtn.disabled = false;
            
            // 生成随机赔率
            this.currentOdds = this.possibleOdds[Math.floor(Math.random() * this.possibleOdds.length)];
            this.currentOddsElement.textContent = `${this.currentOdds}x`;
            
            // 随机选择获奖通道
            this.generateWinningChannels();
            this.updateExitChannels();
            
            this.messageElement.textContent = `已下注 ${betAmount} 筹码！按住击发按钮蓄力！`;
            this.messageElement.className = '';
        } else {
            if (betAmount < 5 || betAmount > 100) {
                alert('下注金额必须在5到100之间！');
            } else if (betAmount > this.chips) {
                alert('筹码不足！');
            }
        }
    }

    cancelBet() {
        this.betInputContainer.style.display = 'none';
    }

    startCharging() {
        if (!this.isGameActive || this.isBallInMotion || this.isCharging) return;
        
        this.isCharging = true;
        this.powerLevel = 0;
        this.messageElement.textContent = '蓄力中... 松开发射！';
        
        // 蓄力动画
        const chargeInterval = setInterval(() => {
            if (!this.isCharging) {
                clearInterval(chargeInterval);
                return;
            }
            
            this.powerLevel = Math.min(this.maxPower, this.powerLevel + 2);
            this.updatePowerBar();
            
            if (this.powerLevel >= this.maxPower) {
                // 自动发射（防止无限蓄力）
                this.releaseBall();
                clearInterval(chargeInterval);
            }
        }, 50);
        
        this.chargingInterval = chargeInterval;
    }

    releaseBall() {
        if (!this.isCharging) return;
        
        this.isCharging = false;
        if (this.chargingInterval) {
            clearInterval(this.chargingInterval);
        }
        
        this.fireBall();
    }

    updatePowerBar() {
        const percentage = (this.powerLevel / this.maxPower) * 100;
        this.powerBar.style.height = `${percentage}%`;
    }

    fireBall() {
        if (this.isBallInMotion) return;
        
        this.isBallInMotion = true;
        this.fireBtn.disabled = true;
        this.messageElement.textContent = '弹珠发射中...';
        this.messageElement.className = '';
        
        // 创建弹珠
        this.createBall();
        
        // 开始物理模拟
        this.startPhysicsSimulation();
        
        // 清空蓄力条
        this.powerLevel = 0;
        this.updatePowerBar();
    }

    createBall() {
        // 移除现有弹珠
        if (this.ball) {
            this.ball.element.remove();
        }

        // 创建新弹珠
        const ballElement = document.createElement('div');
        ballElement.className = 'ball';
        this.gameContainer.appendChild(ballElement);

        // 初始位置：从右侧发射管，通过弯道进入左上角
        const initialPower = Math.max(10, this.powerLevel / 10); // 根据蓄力调整初始速度
        
        this.ball = {
            element: ballElement,
            x: 20, // 从左上角开始
            y: 30,
            vx: 3 + initialPower * 0.2, // 向右的初始速度
            vy: 1, // 轻微向下
            radius: this.ballRadius,
            phase: 'entry' // entry, falling, landed
        };

        this.updateBallPosition();
    }

    updateBallPosition() {
        if (this.ball) {
            this.ball.element.style.left = `${this.ball.x - this.ball.radius}px`;
            this.ball.element.style.top = `${this.ball.y - this.ball.radius}px`;
        }
    }

    generateWinningChannels() {
        // 根据赔率决定获奖通道数量
        let winningCount;
        switch(this.currentOdds) {
            case 2: winningCount = 4; break;
            case 4: winningCount = 3; break;
            case 6: winningCount = 2; break;
            case 8: 
            case 10: winningCount = 1; break;
            default: winningCount = 1;
        }
        
        this.winningChannels = [];
        
        // 随机选择获奖通道
        const availableChannels = Array.from({length: this.exitChannels}, (_, i) => i);
        for (let i = 0; i < winningCount; i++) {
            const randomIndex = Math.floor(Math.random() * availableChannels.length);
            this.winningChannels.push(availableChannels.splice(randomIndex, 1)[0]);
        }
    }

    createGameElements() {
        this.createPegs();
        this.createExitChannels();
    }

    createPegs() {
        // 清除现有的柱子
        const existingPegs = this.gameContainer.querySelectorAll('.peg');
        existingPegs.forEach(peg => peg.remove());

        // 创建5层栅格，每层10个柱子 - 调整间距以适应更大的弹珠
        for (let row = 0; row < this.pegRows; row++) {
            const y = 120 + row * 85; // 从顶部120px开始，每层间隔85px
            const pegsInThisRow = this.pegsPerRow;
            
            for (let col = 0; col < pegsInThisRow; col++) {
                // 交错排列：奇数行偏移半个间距
                const offsetX = (row % 2) * 35;
                const x = 70 + col * 55 + offsetX;
                
                if (x < this.containerWidth - 70) { // 确保柱子在边界内
                    const peg = document.createElement('div');
                    peg.className = 'peg';
                    peg.style.left = `${x}px`;
                    peg.style.top = `${y}px`;
                    this.gameContainer.appendChild(peg);
                }
            }
        }
    }

    createExitChannels() {
        // 清除现有的出口通道
        const existingChannels = this.gameContainer.querySelectorAll('.exit-channel');
        existingChannels.forEach(channel => channel.remove());

        // 创建12个出口通道 - 调整尺寸以适应更大的弹珠
        const channelWidth = 48;
        const totalWidth = channelWidth * this.exitChannels;
        const startX = (this.containerWidth - totalWidth) / 2;

        for (let i = 0; i < this.exitChannels; i++) {
            const channel = document.createElement('div');
            channel.className = 'exit-channel normal';
            channel.style.left = `${startX + i * channelWidth}px`;
            channel.style.width = `${channelWidth}px`;
            channel.textContent = i + 1;
            channel.dataset.channelIndex = i;
            this.gameContainer.appendChild(channel);
        }
    }

    updateExitChannels() {
        const channels = this.gameContainer.querySelectorAll('.exit-channel');
        channels.forEach((channel, index) => {
            const isWinning = this.winningChannels.includes(index);
            channel.className = `exit-channel ${isWinning ? 'winning' : 'normal'}`;
            
            if (isWinning) {
                // 添加赔率显示
                let multiplier = channel.querySelector('.odds-multiplier');
                if (!multiplier) {
                    multiplier = document.createElement('div');
                    multiplier.className = 'odds-multiplier';
                    channel.appendChild(multiplier);
                }
                multiplier.textContent = `${this.currentOdds}x`;
            } else {
                // 移除赔率显示
                const multiplier = channel.querySelector('.odds-multiplier');
                if (multiplier) {
                    multiplier.remove();
                }
            }
        });
    }

    startPhysicsSimulation() {
        const animate = () => {
            if (!this.ball || !this.isBallInMotion) return;

            // 应用重力（仅在下落阶段）
            if (this.ball.phase === 'falling') {
                this.ball.vy += this.gravity;
            }
            
            // 应用摩擦力
            this.ball.vx *= this.friction;
            this.ball.vy *= this.friction;
            
            // 更新位置
            this.ball.x += this.ball.vx;
            this.ball.y += this.ball.vy;
            
            // 阶段检测
            if (this.ball.phase === 'entry' && this.ball.y > 80) {
                this.ball.phase = 'falling';
            }
            
            // 边界碰撞检测（容器内壁）
            this.handleContainerWallCollisions();
            
            // 柱子碰撞检测
            this.handlePegCollisions();
            
            // 检查是否到达底部
            if (this.ball.y > this.containerHeight - 50) {
                this.handleBallLanding();
                return;
            }
            
            // 更新弹珠位置
            this.updateBallPosition();
            
            // 继续动画
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }

    handleContainerWallCollisions() {
        // 左右边界（容器内壁）
        if (this.ball.x - this.ball.radius < 10) {
            this.ball.x = 10 + this.ball.radius;
            this.ball.vx *= -this.bounceFactorX * 1.1; // 增加墙壁反弹弹性
        } else if (this.ball.x + this.ball.radius > this.containerWidth - 10) {
            this.ball.x = this.containerWidth - 10 - this.ball.radius;
            this.ball.vx *= -this.bounceFactorX * 1.1; // 增加墙壁反弹弹性
        }
        
        // 顶部边界
        if (this.ball.y - this.ball.radius < 10) {
            this.ball.y = 10 + this.ball.radius;
            this.ball.vy *= -this.bounceFactorY * 1.1; // 增加墙壁反弹弹性
        }
    }

    handlePegCollisions() {
        const pegs = this.gameContainer.querySelectorAll('.peg');
        
        pegs.forEach(peg => {
            const pegRect = peg.getBoundingClientRect();
            const containerRect = this.gameContainer.getBoundingClientRect();
            
            const pegX = pegRect.left - containerRect.left + this.pegRadius;
            const pegY = pegRect.top - containerRect.top + this.pegRadius;
            
            const dx = this.ball.x - pegX;
            const dy = this.ball.y - pegY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < this.ball.radius + this.pegRadius) {
                // 碰撞发生
                const angle = Math.atan2(dy, dx);
                const targetX = pegX + Math.cos(angle) * (this.ball.radius + this.pegRadius);
                const targetY = pegY + Math.sin(angle) * (this.ball.radius + this.pegRadius);
                
                this.ball.x = targetX;
                this.ball.y = targetY;
                
                // 反弹计算
                const normalX = Math.cos(angle);
                const normalY = Math.sin(angle);
                
                const dotProduct = this.ball.vx * normalX + this.ball.vy * normalY;
                
                this.ball.vx -= 2 * dotProduct * normalX * this.bounceFactorX;
                this.ball.vy -= 2 * dotProduct * normalY * this.bounceFactorY;
                
                // 增加弹性 - 给予额外的反弹速度
                const extraBounce = 1.2;
                this.ball.vx *= extraBounce;
                this.ball.vy *= extraBounce;
                
                // 添加随机性
                this.ball.vx += (Math.random() - 0.5) * 4;
                this.ball.vy += Math.abs(Math.random() - 0.5) * 3;
            }
        });
    }

    handleBallLanding() {
        // 确定弹珠落入哪个通道
        const channelIndex = this.determineLandingChannel();
        const isWinning = this.winningChannels.includes(channelIndex);
        
        if (isWinning) {
            const reward = this.currentBet * this.currentOdds;
            this.chips += reward;
            this.score += reward;
            this.chipsElement.textContent = this.chips;
            this.scoreElement.textContent = this.score;
            this.messageElement.textContent = `🎉 恭喜中奖！获得 ${reward} 筹码！`;
            this.messageElement.className = 'win-message';
        } else {
            this.messageElement.textContent = `很遗憾没有中奖，损失 ${this.currentBet} 筹码。`;
            this.messageElement.className = 'lose-message';
        }
        
        // 重置游戏状态
        setTimeout(() => {
            this.resetForNextRound();
        }, 3000);
    }

    determineLandingChannel() {
        const channelWidth = 48;
        const totalWidth = channelWidth * this.exitChannels;
        const startX = (this.containerWidth - totalWidth) / 2;
        
        // 计算弹珠落入哪个通道
        const relativeX = this.ball.x - startX;
        let channelIndex = Math.floor(relativeX / channelWidth);
        
        // 确保在有效范围内
        channelIndex = Math.max(0, Math.min(this.exitChannels - 1, channelIndex));
        
        return channelIndex;
    }

    resetForNextRound() {
        // 移除弹珠
        if (this.ball) {
            this.ball.element.remove();
            this.ball = null;
        }
        
        this.isBallInMotion = false;
        this.isGameActive = false;
        this.currentBet = 0;
        this.startBtn.disabled = false;
        this.fireBtn.disabled = true;
        
        // 重置出口通道显示
        const channels = this.gameContainer.querySelectorAll('.exit-channel');
        channels.forEach(channel => {
            channel.className = 'exit-channel normal';
            const multiplier = channel.querySelector('.odds-multiplier');
            if (multiplier) {
                multiplier.remove();
            }
        });
        
        this.currentOddsElement.textContent = '-';
        
        if (this.chips >= 5) {
            this.messageElement.textContent = '准备下一轮！点击"开始游戏"继续。';
        } else {
            this.messageElement.textContent = '筹码不足！游戏结束。';
            this.messageElement.className = 'lose-message';
        }
        this.messageElement.className = '';
    }

    resetGame() {
        this.isGameActive = false;
        this.isBallInMotion = false;
        this.isCharging = false;
        this.currentOdds = null;
        this.winningChannels = [];
        this.powerLevel = 0;
        this.currentBet = 0;
        
        if (this.ball) {
            this.ball.element.remove();
            this.ball = null;
        }
        
        this.updatePowerBar();
        this.startBtn.disabled = false;
        this.fireBtn.disabled = true;
        this.currentOddsElement.textContent = '-';
        this.messageElement.textContent = '';
        this.messageElement.className = '';
        
        // 重置出口通道显示
        const channels = this.gameContainer.querySelectorAll('.exit-channel');
        channels.forEach(channel => {
            channel.className = 'exit-channel normal';
            const multiplier = channel.querySelector('.odds-multiplier');
            if (multiplier) {
                multiplier.remove();
            }
        });
    }
}

// 初始化游戏
document.addEventListener('DOMContentLoaded', () => {
    new PlinkoGame();
});