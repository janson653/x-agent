import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import logging
import os

# Assuming the classes AgentConfig and ProductDatabase are defined somewhere
# For the sake of testing, we'll mock them
class AgentConfig:
    model_name = "test_model"
    base_url = "http://test.url"

class ProductDatabase:
    def search_products(self, query: str) -> List[Dict]:
        return [{"id": "1", "name": "Product 1"}]

    def get_product_details(self, product_id: str) -> Optional[Dict]:
        return {"id": product_id, "name": f"Product {product_id}"}

class TestEcommerceAgent(unittest.TestCase):

    @patch('os.getenv')
    @patch('dotenv.load_dotenv')
    def setUp(self, mock_load_dotenv, mock_getenv):
        mock_getenv.return_value = "test_api_key"
        self.agent = EcommerceAgent(config=AgentConfig())
    
    @patch('logging.Logger')
    def test_setup_logger(self, mock_logger):
        logger = self.agent._setup_logger()
        self.assertIsInstance(logger, logging.Logger)
    
    @patch('os.getenv')
    def test_load_environment_success(self, mock_getenv):
        mock_getenv.return_value = "test_api_key"
        self.agent._load_environment()
        mock_getenv.assert_called_with("DEEPSEEK_API_KEY")
    
    @patch('os.getenv')
    def test_load_environment_failure(self, mock_getenv):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            self.agent._load_environment()
    
    @patch('langchain_community.chat_models.ChatOpenAI')
    def test_initialize_model(self, mock_chat_openai):
        model = self.agent._initialize_model()
        mock_chat_openai.assert_called_with(
            api_key="test_api_key",
            model_name="test_model",
            base_url="http://test.url"
        )
    
    def test_create_tools(self):
        tools = self.agent._create_tools()
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0].name, "search_products")
        self.assertEqual(tools[1].name, "get_product_details")
    
    @patch('EcommerceAgent._safe_search_products')
    def test_safe_search_products(self, mock_safe_search_products):
        mock_safe_search_products.return_value = [{"id": "1", "name": "Product 1"}]
        result = self.agent._safe_search_products("test query")
        self.assertEqual(result, [{"id": "1", "name": "Product 1"}])
    
    @patch('EcommerceAgent._safe_get_product_details')
    def test_safe_get_product_details(self, mock_safe_get_product_details):
        mock_safe_get_product_details.return_value = {"id": "1", "name": "Product 1"}
        result = self.agent._safe_get_product_details("1")
        self.assertEqual(result, {"id": "1", "name": "Product 1"})
    
    def test_create_prompt(self):
        prompt = self.agent._create_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)
    
    def test_create_memory(self):
        memory = self.agent._create_memory()
        self.assertIsInstance(memory, ConversationBufferMemory)
    
    @patch('langchain.agents.create_openai_functions_agent')
    @patch('langchain.agents.AgentExecutor')
    def test_create_agent_executor(self, mock_agent_executor, mock_create_agent):
        agent_executor = self.agent._create_agent_executor()
        self.assertIsInstance(agent_executor, AgentExecutor)
    
    @patch('EcommerceAgent._safe_get_product_details')
    def test_handle_product_details(self, mock_safe_get_product_details):
        mock_safe_get_product_details.return_value = {"id": "1", "name": "Product 1"}
        self.agent._handle_product_details('get_product_details(product_id="1")')
        self.assertTrue(mock_safe_get_product_details.called)
    
    @patch('EcommerceAgent._safe_search_products')
    def test_handle_search_products(self, mock_safe_search_products):
        mock_safe_search_products.return_value = [{"id": "1", "name": "Product 1"}]
        self.agent._handle_search_products('search_products(search_term="test")')
        self.assertTrue(mock_safe_search_products.called)
    
    @patch('builtins.input', return_value="退出")
    def test_run_exit(self, mock_input):
        self.agent.run()
        mock_input.assert_called_once()

if __name__ == '__main__':
    unittest.main()


