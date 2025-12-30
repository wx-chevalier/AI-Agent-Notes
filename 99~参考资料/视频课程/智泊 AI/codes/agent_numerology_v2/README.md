# 命理机器人

命理机器人是一个基于 Python 的智能问答系统，结合RAG、AI Agent和现代 Web 前端，能够为用户提供命理相关的自动化问答服务。

## 目录结构

```
命理机器人/
├── local_qdrand/         # 本地知识库及相关数据
├── static/               # 前端静态资源（CSS/JS/图片）
├── templates/            # 前端模板（HTML）
├── mytools.py            # 工具函数
├── logger.py             # 日志配置
├── models.py             # 产生大模型客户端
├── requirements.txt      # Python 依赖包列表
├── server.py             # 主服务端代码
└── README.md             # 本文件，项目说明文档
```

## 快速开始

### 1. 安装依赖

建议使用虚拟环境：

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python server.py
```

默认会在本地 http://127.0.0.1:8000/ 启动 Web 服务。

### 3. 访问前端

浏览器打开 [http://127.0.0.1:8000/index](http://127.0.0.1:8000/index) 即可体验命理机器人。
浏览器打开 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 即可打开API文档，并通过接口新增知识。

## 主要功能

- 命理相关知识问答
- 本地知识库检索和知识增加
- 现代 Web 前端界面
- 可扩展的工具函数（mytools.py）


## 使用以下命令构建和运行Docker镜像：
### 方法一：
使用docker build
docker build -t zhipo-numerology-app .

运行容器
docker run --name zhipo-numerology -p 8000:8000 zhipo-numerology-app

###  方法二： 
或者使用docker-compose（推荐）
docker-compose -p zhipo-numerology up --build
