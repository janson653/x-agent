from typing import List, Dict
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI  # 使用替代的ChatOpenAI类
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化Deepseek模型
model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_name="deepseek-ai/DeepSeek-V2.5",
    base_url="https://api.siliconflow.cn/v1"
)

# 定义电商相关的工具
class ProductDatabase:
    def __init__(self):
        # 模拟产品数据库
        self.products = {
            "1": {"name": "笔记本电脑", "price": 5999, "stock": 10},
            "2": {"name": "智能手机", "price": 3999, "stock": 20},
            "3": {"name": "无线耳机", "price": 999, "stock": 50}
        }
    
    def search_products(self, query: str) -> List[Dict]:
        """搜索产品"""
        results = []
        for id, product in self.products.items():
            if query.lower() in product["name"].lower():
                results.append({"id": id, **product})
        return results
    
    def get_product_details(self, product_id: str) -> Dict:
        """获取产品详情"""
        if product_id in self.products:
            return {"id": product_id, **self.products[product_id]}
        return None

# 创建工具实例
product_db = ProductDatabase()

tools = [
    Tool(
        name="search_products",
        func=product_db.search_products,
        description="搜索产品。输入搜索关键词，返回匹配的产品列表。"
    ),
    Tool(
        name="get_product_details",
        func=product_db.get_product_details,
        description="获取产品详细信息。输入产品ID，返回产品详情。"
    )
]

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的电商客服助手。你可以帮助用户：
    1. 搜索产品
    2. 查看产品详情
    3. 回答产品相关问题
    4. 提供购买建议
    
    请使用礼貌专业的语气，并确保提供准确的信息。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建agent
agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

# 创建agent执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def main():
    print("欢迎使用电商助手！输入'退出'结束对话。")
    
    while True:
        user_input = input("\n请输入您的问题：")
        if user_input.lower() == "退出":
            break
            
        try:
            response = agent_executor.invoke({"input": user_input})
            print("\n助手：", response["output"])
        except Exception as e:
            print("\n发生错误：", str(e))

if __name__ == "__main__":
    main() 