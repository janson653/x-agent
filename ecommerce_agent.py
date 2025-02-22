from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import json

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
        # 从JSON文件加载更真实的产品数据
        with open('products.json', 'r', encoding='utf-8') as f:
            self.products = json.load(f)
    
    def search_products(self, query: str) -> List[Dict]:
        """使用大模型增强的语义搜索"""
        # 让大模型理解查询意图并提取关键词
        enhanced_query = model.invoke(f"""
        请分析以下用户查询，提取最相关的产品搜索关键词：
        用户查询：{query}
        返回格式：关键词1, 关键词2, 关键词3
        """).content
        
        results = []
        for id, product in self.products.items():
            # 使用大模型增强的匹配
            match_score = model.invoke(f"""
            请评估产品与查询的匹配度（0-10分）：
            产品：{product['name']} {product['description']}
            查询：{enhanced_query}
            返回格式：分数
            """).content
            # 处理分数字符串，提取数字部分
            try:
                score = int(match_score.split("：")[1]) if "：" in match_score else int(match_score)
                if score >= 1:
                    results.append({"id": id, **product})
            except (ValueError, IndexError):
                continue
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
        description="使用语义搜索产品。输入自然语言查询，返回匹配的产品列表。"
    ),
    Tool(
        name="get_product_details",
        func=product_db.get_product_details,
        description="获取产品详细信息。输入产品ID，返回产品详情。"
    )
]

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能电商客服助手。你的能力包括：
    1. 理解用户意图并选择正确的工具
    2. 使用语义搜索匹配产品
    3. 提供详细的产品信息
    4. 给出个性化的购买建议
    
    请严格遵循以下原则：
    - 使用自然语言理解用户意图
    - 必须调用合适的工具来处理用户请求
    - 所有产品推荐必须来自产品数据库
    - 提供专业、友好的服务
    - 对于不支持的功能，给出替代建议
    
    操作流程：
    1. 分析用户意图
    2. 调用合适的工具获取产品信息
    3. 基于工具返回的结果进行回复
    
    重要规则：
    1. 所有产品信息必须来自工具调用结果
    2. 不能凭空生成产品信息
    3. 如果工具返回空结果，请如实告知用户
    4. 推荐产品时必须使用工具获取的完整产品信息
    5. 每次对话必须至少调用一次工具
    6. 工具调用必须使用以下严格格式：
       - search_products(search_term="搜索关键词")
       - get_product_details(product_id=产品ID)
    7. 工具调用时，参数必须用双引号包裹
    8. 工具调用输出必须严格匹配上述格式，不能包含其他内容
    
    工具调用示例：
    1. 当用户询问"推荐一些智能设备"时，输出：search_products(search_term="智能设备")
    2. 当用户询问"1001号产品的详细信息"时，输出：get_product_details(product_id="1001")
    3. 当用户询问"价格在5000元以下的电子产品"时，输出：search_products(search_term="电子产品 价格<5000")
    4. 当用户询问"有什么优惠活动"时，输出：search_products(search_term="优惠")
    5. 当用户询问"帮我推荐一款手机"时，输出：search_products(search_term="手机")
    6. 当用户询问"1002号产品有货吗"时，输出：get_product_details(product_id="1002")
    7. 当用户询问"最贵的电子产品是什么"时，输出：search_products(search_term="电子产品 价格降序")
    """),
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
    verbose=True,
    handle_parsing_errors=True
)

def main():
    print("欢迎使用智能电商助手！输入'退出'结束对话。")
    
    while True:
        user_input = input("\n请输入您的问题：")
        
        if user_input.lower() == '退出':
            print("感谢使用，再见！")
            break
            
        try:
            # 让大模型自动处理意图识别和工具选择
            response = agent_executor.invoke({
                "input": user_input
            })
           
            # 根据output中的工具名判断是否调用了外部工具
            output = response.get("output", "")            
            # 如果调用了get_product_details工具
            if "get_product_details" in output:
                print("\n[系统] 已调用get_product_details工具处理您的请求")
                # 从output中提取产品ID
                try:
                    product_id = output.split('product_id="')[1].split('"')[0]
                    product_info = product_db.get_product_details(product_id)
                    if product_info:
                        # 使用大模型概况产品信息
                        summary_prompt = f"请用简洁的语言概括以下产品信息，突出主要特点和优势：\n{product_info}"
                        summary_response = model.invoke(summary_prompt)
                        print(f"\n产品概况：\n{summary_response.content}")
                        print(f"\n完整产品详情：\n{product_info}")

                    else:
                        print("\n[系统] 未找到该产品信息，请检查产品ID是否正确")
                except (IndexError, ValueError) as e:
                    print("\n[系统] 无法提取有效的产品ID，请检查输入格式或重新输入")
            
            # 如果调用了search_products工具
            elif "search_products" in output:
                print("\n[系统] 已调用search_products工具处理您的请求")
                # 从output中提取搜索关键词
                try:
                    search_term = output.split('search_term="')[1].split('"')[0]
                    if search_term:  # 确保搜索关键词不为空
                        search_results = product_db.search_products(search_term)
                        if search_results:
                            # 使用大模型概况搜索结果
                            summary_prompt = f"请用简洁的语言概括以下搜索结果，突出主要产品特点和优势：\n{search_results}"
                            summary_response = model.invoke(summary_prompt)
                            print(f"\n搜索结果概况：\n{summary_response.content}")
                            print(f"\n完整搜索结果：\n{search_results}")
                        else:
                            print("\n[系统] 未找到相关产品，请尝试其他关键词")
                    else:
                        print("\n[系统] 未获取到有效的搜索关键词，请重新输入")
                except IndexError:
                    print("\n[系统] 无法提取搜索关键词，请检查输入格式")
            # 如果没有调用工具，直接输出助手回复
            else:
                print("\n助手：", response["output"])

        except Exception as e:
            print("\n发生错误：", str(e))

if __name__ == "__main__":
    main()
