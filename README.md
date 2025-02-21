# 电商智能助手

这是一个基于LangChain和Deepseek的电商智能助手，能够帮助用户搜索产品、查看产品详情、回答产品相关问题并提供购买建议。

## 功能特点

- 产品搜索
- 产品详情查询
- 智能问答
- 购买建议
- 对话记忆功能

## 安装步骤

1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 复制 `.env.example` 到 `.env` 并填入你的Deepseek API密钥：
   ```bash
   cp .env.example .env
   ```

## 使用方法

运行主程序：
```bash
python ecommerce_agent.py
```

## 示例对话

- 用户：有什么笔记本电脑推荐？
- 助手：让我为您搜索笔记本电脑相关产品...

## 注意事项

- 确保已正确设置Deepseek API密钥
- 当前产品数据库为示例数据，可以根据需要扩展

## 技术栈

- Python 3.8+
- LangChain
- Deepseek AI
- python-dotenv 