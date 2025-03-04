from typing import List, Dict, Optional, Any, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import json
import logging
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class AgentConfig:
    model_name: str = "deepseek-ai/DeepSeek-V2.5"
    base_url: str = "https://api.siliconflow.cn/v1"
    max_search_results: int = 10
    min_match_score: int = 5

class EcommerceAgent:
    """智能电商客服助手，提供产品搜索、推荐和详细信息查询功能"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """初始化电商助手
        
        Args:
            config (Optional[AgentConfig]): 配置参数，如果未提供则使用默认值
        """
        self._load_environment()
        self.config = config or AgentConfig()
        self.model = self._initialize_model()
        self.product_db = ProductDatabase()
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        self.memory = self._create_memory()
        self.agent_executor = self._create_agent_executor()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """配置并返回日志记录器"""
        logger = logging.getLogger("EcommerceAgent")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_environment(self) -> None:
        """加载环境变量"""
        try:
            load_dotenv()
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("DEEPSEEK_API_KEY 未在环境变量中设置")
        except Exception as e:
            self.logger.error(f"加载环境变量失败: {str(e)}")
            raise

    def _initialize_model(self) -> ChatOpenAI:
        """初始化语言模型"""
        try:
            return ChatOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model_name=self.config.model_name,
                base_url=self.config.base_url
            )
        except Exception as e:
            self.logger.error(f"初始化语言模型失败: {str(e)}")
            raise

    def _create_tools(self) -> List[Tool]:
        """创建工具列表"""
        return [
            Tool(
                name="search_products",
                func=self._safe_search_products,
                description="使用语义搜索产品。输入自然语言查询，返回匹配的产品列表。"
            ),
            Tool(
                name="get_product_details",
                func=self._safe_get_product_details,
                description="获取产品详细信息。输入产品ID，返回产品详情。"
            )
        ]

    def _safe_search_products(self, query: str) -> List[Dict]:
        """安全的搜索产品方法，包含错误处理"""
        try:
            return self.product_db.search_products(query)
        except Exception as e:
            self.logger.error(f"搜索产品失败: {str(e)}")
            return []

    def _safe_get_product_details(self, product_id: str) -> Optional[Dict]:
        """安全获取产品详情方法，包含错误处理"""
        try:
            return self.product_db.get_product_details(product_id)
        except Exception as e:
            self.logger.error(f"获取产品详情失败: {str(e)}")
            return None


    def _create_prompt(self):
        return ChatPromptTemplate.from_messages([
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

    def _create_memory(self):
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def _create_agent_executor(self):
        agent = create_openai_functions_agent(
            llm=self.model,
            prompt=self.prompt,
            tools=self.tools
        )
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def _handle_product_details(self, output: str) -> None:
        """处理产品详情请求"""
        self.logger.info("已调用get_product_details工具处理您的请求")
        try:
            product_id = output.split('product_id="')[1].split('"')[0]
            product_info = self._safe_get_product_details(product_id)
            if product_info:
                summary_prompt = f"请用简洁的语言概括以下产品信息，突出主要特点和优势：\n{product_info}"
                summary_response = self.model.invoke(summary_prompt)
                self.logger.info(f"产品概况：\n{summary_response.content}")
                self.logger.info(f"完整产品详情：\n{product_info}")
            else:
                self.logger.warning("未找到该产品信息，请检查产品ID是否正确")
        except (IndexError, ValueError) as e:
            self.logger.error(f"无法提取有效的产品ID: {str(e)}")

    def _handle_search_products(self, output: str) -> None:
        """处理产品搜索请求"""
        self.logger.info("已调用search_products工具处理您的请求")
        try:
            search_term = output.split('search_term="')[1].split('"')[0]
            if search_term:
                search_results = self._safe_search_products(search_term)
                if search_results:
                    summary_prompt = f"请用简洁的语言概括以下搜索结果，突出主要产品特点和优势：\n{search_results}"
                    summary_response = self.model.invoke(summary_prompt)
                    self.logger.info(f"搜索结果概况：\n{summary_response.content}")
                    self.logger.info(f"完整搜索结果：\n{search_results}")
                else:
                    self.logger.warning("未找到相关产品，请尝试其他关键词")
            else:
                self.logger.warning("未获取到有效的搜索关键词，请重新输入")
        except IndexError as e:
            self.logger.error(f"无法提取搜索关键词: {str(e)}")

    def run(self):
        print("欢迎使用智能电商助手！输入'退出'结束对话。")
        
        while True:
            user_input = input("\n请输入您的问题：")
            
            if user_input.lower() == '退出':
                print("感谢使用，再见！")
                break
                
            try:
                response = self.agent_executor.invoke({"input": user_input})
                output = response.get("output", "")
                
                if "get_product_details" in output:
                    self._handle_product_details(output)
                elif "search_products" in output:
                    self._handle_search_products(output)
                else:
                    print("\n助手：", response["output"])

            except Exception as e:
                print("\n发生错误：", str(e))

class ProductDatabase:
    def __init__(self):
        with open('products.json', 'r', encoding='utf-8') as f:
            self.products = json.load(f)
        self.model = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model_name="deepseek-ai/DeepSeek-V2.5",
            base_url="https://api.siliconflow.cn/v1"
        )
    
    def search_products(self, query: str) -> List[Dict]:
        enhanced_query = self.model.invoke(f"""
        请分析以下用户查询，提取最相关的产品搜索关键词：
        用户查询：{query}
        返回格式：关键词1, 关键词2, 关键词3
        """).content
        
        results = []
        for id, product in self.products.items():
            match_score = self.model.invoke(f"""
            请评估产品与查询的匹配度（0-10分）：
            产品：{product['name']} {product['description']}
            查询：{enhanced_query}
            返回格式：分数
            """).content
            try:
                score = int(match_score.split("：")[1]) if "：" in match_score else int(match_score)
                if score >= 1:
                    results.append({"id": id, **product})
            except (ValueError, IndexError):
                continue
        return results
    
    def get_product_details(self, product_id: str) -> Dict:
        if product_id in self.products:
            return {"id": product_id, **self.products[product_id]}
        return None

if __name__ == "__main__":
    agent = EcommerceAgent()
    agent.run()



