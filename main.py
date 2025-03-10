import asyncio

from app.agent.manus import Manus
from app.config import config
from app.logger import logger


async def main():
    # Get the default LLM configuration from global settings
    default_llm = config.global_.default_llm
    
    # Initialize agent with configured LLM
    agent = Manus(llm_config_name=default_llm)
    
    while True:
        try:
            prompt = input("Enter your prompt (or 'exit' to quit): ")
            if prompt.lower() == "exit":
                logger.info("Goodbye!")
                break
            logger.warning("Processing your request...")
            await agent.run(prompt)
        except KeyboardInterrupt:
            logger.warning("Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
