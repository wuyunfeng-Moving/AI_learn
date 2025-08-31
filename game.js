class PlinkoGame {
    constructor() {
        this.gameBoard = document.getElementById('gameBoard');
        this.startBtn = document.getElementById('startBtn');
        this.fireBtn = document.getElementById('fireBtn');
        this.currentOddsElement = document.getElementById('currentOdds');
        this.scoreElement = document.getElementById('score');
        this.messageElement = document.getElementById('message');
        
        this.score = 0;
        this.currentOdds = null;
        this.winningChannels = [];
        this.ball = null;
        this.isGameActive = false;
        this.isBallInMotion = false;
        
        // 游戏配置
        this.boardWidth = 760;
        this.boardHeight = 600;
        this.pegRows = 5;
        this.pegsPerRow = 10;
        this.exitChannels = 12;
        this.possibleOdds = [2, 4, 6, 8, 10];
        
        // 物理参数
        this.gravity = 0.3;
        this.friction = 0.98;
        this.bounceFactorX = 0.7;
        this.bounceFactorY = 0.6;
        this.pegRadius = 4;
        this.ballRadius = 6;
        
        this.initEventListeners();
        this.createGameElements();
    }

    initEventListeners() {
        this.startBtn.addEventListener('click', () => this.startGame());
        this.fireBtn.addEventListener('click', () => this.fireBall());
    }

    startGame() {
        this.isGameActive = true;
        this.startBtn.disabled = true;
        this.fireBtn.disabled = false;
        
        // 生成随机赔率
        this.currentOdds = this.possibleOdds[Math.floor(Math.random() * this.possibleOdds.length)];
        this.currentOddsElement.textContent = `${this.currentOdds}x`;
        
        // 随机选择获奖通道
        this.generateWinningChannels();
        this.updateExitChannels();
        
        this.messageElement.textContent = '准备发射弹珠！';
        this.messageElement.className = '';
    }

    generateWinningChannels() {
        // 根据赔率决定获奖通道数量（赔率越高，获奖通道越少）
        const winningCount = Math.max(1, Math.floor(this.exitChannels / this.currentOdds));
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
        const existingPegs = this.gameBoard.querySelectorAll('.peg');
        existingPegs.forEach(peg => peg.remove());

        // 创建5层栅格，每层10个柱子
        for (let row = 0; row < this.pegRows; row++) {
            const y = 150 + row * 80; // 从顶部150px开始，每层间隔80px
            const pegsInThisRow = this.pegsPerRow;
            
            for (let col = 0; col < pegsInThisRow; col++) {
                // 交错排列：奇数行偏移半个间距
                const offsetX = (row % 2) * 30;
                const x = 80 + col * 60 + offsetX;
                
                if (x < this.boardWidth - 80) { // 确保柱子在边界内
                    const peg = document.createElement('div');
                    peg.className = 'peg';
                    peg.style.left = `${x}px`;
                    peg.style.top = `${y}px`;
                    this.gameBoard.appendChild(peg);
                }
            }
        }
    }

    createExitChannels() {
        // 清除现有的出口通道
        const existingChannels = this.gameBoard.querySelectorAll('.exit-channel');
        existingChannels.forEach(channel => channel.remove());

        // 创建12个出口通道
        const channelWidth = 60;
        const totalWidth = channelWidth * this.exitChannels;
        const startX = (this.boardWidth - totalWidth) / 2;

        for (let i = 0; i < this.exitChannels; i++) {
            const channel = document.createElement('div');
            channel.className = 'exit-channel normal';
            channel.style.left = `${startX + i * channelWidth}px`;
            channel.textContent = i + 1;
            channel.dataset.channelIndex = i;
            this.gameBoard.appendChild(channel);
        }
    }

    updateExitChannels() {
        const channels = this.gameBoard.querySelectorAll('.exit-channel');
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
    }

    createBall() {
        // 移除现有弹珠
        if (this.ball) {
            this.ball.element.remove();
        }

        // 创建新弹珠
        const ballElement = document.createElement('div');
        ballElement.className = 'ball';
        this.gameBoard.appendChild(ballElement);

        // 初始位置：发射管底部中央
        const launchTube = document.getElementById('launchTube');
        const tubeRect = launchTube.getBoundingClientRect();
        const boardRect = this.gameBoard.getBoundingClientRect();
        
        this.ball = {
            element: ballElement,
            x: this.boardWidth / 2,
            y: this.boardHeight - 60, // 发射管顶部
            vx: 0,
            vy: -15, // 向上发射
            radius: this.ballRadius
        };

        this.updateBallPosition();
    }

    updateBallPosition() {
        if (this.ball) {
            this.ball.element.style.left = `${this.ball.x - this.ball.radius}px`;
            this.ball.element.style.top = `${this.ball.y - this.ball.radius}px`;
        }
    }

    startPhysicsSimulation() {
        const animate = () => {
            if (!this.ball || !this.isBallInMotion) return;

            // 应用重力
            this.ball.vy += this.gravity;
            
            // 应用摩擦力
            this.ball.vx *= this.friction;
            this.ball.vy *= this.friction;
            
            // 更新位置
            this.ball.x += this.ball.vx;
            this.ball.y += this.ball.vy;
            
            // 边界碰撞检测
            this.handleWallCollisions();
            
            // 柱子碰撞检测
            this.handlePegCollisions();
            
            // 检查是否到达底部
            if (this.ball.y > this.boardHeight - 50) {
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

    handleWallCollisions() {
        // 左右边界
        if (this.ball.x - this.ball.radius < 0) {
            this.ball.x = this.ball.radius;
            this.ball.vx *= -this.bounceFactorX;
        } else if (this.ball.x + this.ball.radius > this.boardWidth) {
            this.ball.x = this.boardWidth - this.ball.radius;
            this.ball.vx *= -this.bounceFactorX;
        }
        
        // 顶部边界
        if (this.ball.y - this.ball.radius < 0) {
            this.ball.y = this.ball.radius;
            this.ball.vy *= -this.bounceFactorY;
        }
    }

    handlePegCollisions() {
        const pegs = this.gameBoard.querySelectorAll('.peg');
        
        pegs.forEach(peg => {
            const pegRect = peg.getBoundingClientRect();
            const boardRect = this.gameBoard.getBoundingClientRect();
            
            const pegX = pegRect.left - boardRect.left + this.pegRadius;
            const pegY = pegRect.top - boardRect.top + this.pegRadius;
            
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
                
                // 添加随机性
                this.ball.vx += (Math.random() - 0.5) * 2;
                this.ball.vy += (Math.random() - 0.5) * 1;
            }
        });
    }

    handleBallLanding() {
        // 确定弹珠落入哪个通道
        const channelIndex = this.determineLandingChannel();
        const isWinning = this.winningChannels.includes(channelIndex);
        
        if (isWinning) {
            const reward = this.currentOdds * 10; // 基础奖励 * 赔率
            this.score += reward;
            this.scoreElement.textContent = this.score;
            this.messageElement.textContent = `🎉 恭喜！获得 ${reward} 分！`;
            this.messageElement.className = 'win-message';
        } else {
            this.messageElement.textContent = '很遗憾，这次没有中奖。再试一次！';
            this.messageElement.className = 'lose-message';
        }
        
        // 重置游戏状态
        setTimeout(() => {
            this.resetForNextRound();
        }, 2000);
    }

    determineLandingChannel() {
        const channelWidth = 60;
        const totalWidth = channelWidth * this.exitChannels;
        const startX = (this.boardWidth - totalWidth) / 2;
        
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
        this.fireBtn.disabled = false;
        this.messageElement.textContent = '准备下一次发射！';
        this.messageElement.className = '';
    }

    resetGame() {
        this.isGameActive = false;
        this.isBallInMotion = false;
        this.currentOdds = null;
        this.winningChannels = [];
        
        if (this.ball) {
            this.ball.element.remove();
            this.ball = null;
        }
        
        this.startBtn.disabled = false;
        this.fireBtn.disabled = true;
        this.currentOddsElement.textContent = '-';
        this.messageElement.textContent = '';
        this.messageElement.className = '';
        
        // 重置出口通道显示
        const channels = this.gameBoard.querySelectorAll('.exit-channel');
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