// DOM元素
const menuItems = document.querySelectorAll('.menu-item');
const containers = {
    chat: document.querySelector('.chat-container'),
    bazi: document.querySelector('.bazi-container'),
    fortune: document.querySelector('.fortune-container'),
    dream: document.querySelector('.dream-container'),
    divination: document.querySelector('.divination-container')
};
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingOverlay = document.querySelector('.loading-overlay');

// 生成或获取 session_id
let session_id = localStorage.getItem('session_id');
if (!session_id) {
  session_id = Math.random().toString(36).substr(2, 9);
  localStorage.setItem('session_id', session_id);
}

// 切换菜单
menuItems.forEach(item => {
    item.addEventListener('click', () => {
        // 移除所有active类
        menuItems.forEach(i => i.classList.remove('active'));
        // 添加active类到当前项
        item.classList.add('active');
        
        // 隐藏所有容器
        Object.values(containers).forEach(container => container.classList.add('hidden'));
        
        // 显示选中的容器
        const type = item.dataset.type;
        containers[type].classList.remove('hidden');
    });
});

// 发送消息
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // 添加用户消息到聊天界面
    addMessage(message, 'user');
    userInput.value = '';

    // 显示加载动画
    loadingOverlay.classList.remove('hidden');

    try {
        // 发送请求到后端
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message, session_id: session_id })
        });

        const data = await response.json();
        
        // 添加机器人回复到聊天界面
        // 检查返回的数据格式并正确处理
        let botResponse = '';
        if (typeof data === 'object') {
            if (data.output) {
                botResponse = data.output;
            } else if (data.data) {
                botResponse = data.data;
            } else {
                // 如果是其他格式的对象，尝试转换为字符串
                botResponse = JSON.stringify(data, null, 2);
            }
        } else {
            botResponse = data;
        }
        
        addMessage(botResponse, 'bot');
    } catch (error) {
        console.error('Error:', error);
        addMessage('抱歉，发生了一些错误，请稍后再试。', 'bot');
    } finally {
        // 隐藏加载动画
        loadingOverlay.classList.add('hidden');
    }
}

// 添加消息到聊天界面
function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${type}-message`);
    
    // 处理文本中的换行符
    const formattedText = text.replace(/\n/g, '<br>');
    messageDiv.innerHTML = formattedText;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 发送按钮点击事件
sendButton.addEventListener('click', sendMessage);

// 输入框回车事件
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 八字测算表单提交
document.getElementById('baziForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        name: formData.get('name'),
        sex: formData.get('sex'),
        birthdate: formData.get('birthdate'),
        birthtime: formData.get('birthtime')
    };

    loadingOverlay.classList.remove('hidden');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: `请帮我测算八字，姓名：${data.name}，性别：${data.sex === '0' ? '男' : '女'}，出生日期：${data.birthdate}，出生时间：${data.birthtime}`,
                session_id: session_id
            })
        });

        const result = await response.json();
        addMessage(result.output, 'bot');
    } catch (error) {
        console.error('Error:', error);
        addMessage('抱歉，测算过程中发生错误，请稍后再试。', 'bot');
    } finally {
        loadingOverlay.classList.add('hidden');
    }
});

// 解梦表单提交
document.getElementById('dreamForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const dream = formData.get('dream');

    loadingOverlay.classList.remove('hidden');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: `请帮我解梦，我的梦境是：${dream}`,
                session_id: session_id
            })
        });

        const result = await response.json();
        addMessage(result.output, 'bot');
    } catch (error) {
        console.error('Error:', error);
        addMessage('抱歉，解梦过程中发生错误，请稍后再试。', 'bot');
    } finally {
        loadingOverlay.classList.add('hidden');
    }
});

// 摇卦占卜
document.getElementById('startDivination').addEventListener('click', async () => {
    loadingOverlay.classList.remove('hidden');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: '请帮我摇一卦',
                session_id: session_id
            })
        });

        const result = await response.json();
        const divinationResult = document.getElementById('divinationResult');
        divinationResult.classList.remove('hidden');
        divinationResult.innerHTML = `<p>${result.output}</p>`;
    } catch (error) {
        console.error('Error:', error);
        addMessage('抱歉，占卜过程中发生错误，请稍后再试。', 'bot');
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}); 