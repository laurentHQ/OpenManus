from typing import Dict, List, Literal, Optional, Union
import time
import asyncio
import tiktoken
import random
from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
    InternalServerError
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    RetryError
)

from app.config import LLMSettings, config
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import Message

# Define custom exceptions
class OverloadedError(Exception):
    """Raised when the LLM service is overloaded"""
    pass

class RetryableError(Exception):
    """Base class for errors that should be retried"""
    pass

class TokenBucket:
    """Token bucket for rate limiting."""
    def __init__(self, tokens_per_minute: int = 40000):
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.last_updated = time.time()
        self.tokens_per_minute = tokens_per_minute
        self.backoff_factor = 1.0  # Dynamic backoff factor

    async def consume(self, tokens: int) -> bool:
        """Try to consume tokens from the bucket."""
        now = time.time()
        # Refill tokens based on time passed
        time_passed = now - self.last_updated
        self.tokens = min(
            self.capacity,
            self.tokens + (time_passed * (self.tokens_per_minute / 60.0))
        )
        self.last_updated = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def wait_for_tokens(self, tokens: int):
        """Wait until enough tokens are available."""
        attempts = 0
        max_attempts = 5
        while not await self.consume(tokens):
            attempts += 1
            if attempts > max_attempts:
                raise OverloadedError("Token bucket exhausted after maximum attempts")
            # Exponential backoff with jitter
            wait_time = min(60, (2 ** attempts) * self.backoff_factor + random.uniform(0, 1))
            logger.warning(f"Token bucket depleted. Waiting {wait_time:.2f}s before retry (attempt {attempts}/{max_attempts})")
            await asyncio.sleep(wait_time)

    def increase_backoff(self):
        """Increase backoff factor when encountering errors"""
        self.backoff_factor = min(10.0, self.backoff_factor * 1.5)

    def reset_backoff(self):
        """Reset backoff factor after successful requests"""
        self.backoff_factor = 1.0

def handle_api_error(error: OpenAIError) -> Exception:
    """Convert API errors to appropriate exception types"""
    if isinstance(error, InternalServerError):
        if "overloaded" in str(error).lower():
            return OverloadedError("Service is overloaded")
    return error

@retry(
    wait=wait_random_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((OverloadedError, RetryableError))
)
async def retry_with_backoff(func, *args, **kwargs):
    """Generic retry wrapper with backoff"""
    try:
        result = await func(*args, **kwargs)
        # Reset backoff on success
        if hasattr(args[0], 'token_bucket'):
            args[0].token_bucket.reset_backoff()
        return result
    except OpenAIError as e:
        # Increase backoff on failure
        if hasattr(args[0], 'token_bucket'):
            args[0].token_bucket.increase_backoff()
        raise handle_api_error(e)
    except Exception as e:
        logger.error(f"Unexpected error in retry_with_backoff: {e}")
        raise

class LLM:
    _instances: Dict[str, "LLM"] = {}
    _token_buckets: Dict[str, TokenBucket] = {}  # One bucket per config
    _encoding = None  # tiktoken encoding

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.base_url = llm_config.base_url
            self.config_name = config_name
            self.client = AsyncOpenAI(
                api_key=llm_config.api_key, base_url=llm_config.base_url
            )
            
            # Initialize token bucket for this config if not already initialized
            if config_name not in self._token_buckets:
                self.__class__._token_buckets[config_name] = TokenBucket(
                    tokens_per_minute=llm_config.tokens_per_minute
                )
            
            # Initialize tiktoken encoding if not already initialized
            if self._encoding is None:
                try:
                    self.__class__._encoding = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    # Fallback to cl100k_base for Claude models
                    self.__class__._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def token_bucket(self) -> TokenBucket:
        """Get the token bucket for this config."""
        return self._token_buckets[self.config_name]

    def count_tokens(self, messages: List[dict]) -> int:
        """Count the number of tokens in the messages."""
        num_tokens = 0
        for message in messages:
            # Count message role
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if key == "tool_calls":
                    for tool_call in value:
                        num_tokens += len(self._encoding.encode(str(tool_call)))
                else:
                    num_tokens += len(self._encoding.encode(str(value)))
        return num_tokens

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((OverloadedError, RetryableError, RateLimitError))
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to the LLM and get the response.
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            return await retry_with_backoff(self._ask_internal, messages, system_msgs, stream, temperature)
        except RetryError as e:
            logger.error(f"All retry attempts failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    async def _ask_internal(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Internal implementation of ask method"""
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                formatted_messages = system_msgs + self.format_messages(messages)
            else:
                formatted_messages = self.format_messages(messages)

            # Log request details
            logger.info("Request details:")
            logger.info(f"Model: {self.model}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Temperature: {temperature or self.temperature}")
            logger.info(f"Stream: {stream}")
            
            logger.info("Formatted messages:")
            for idx, msg in enumerate(formatted_messages):
                logger.info(f"Message {idx + 1}:")
                logger.info(f"  Role: {msg.get('role')}")
                if 'content' in msg:
                    logger.info(f"  Content: {msg.get('content')[:200]}..." if len(msg.get('content', '')) > 200 else f"  Content: {msg.get('content')}")
                if 'tool_calls' in msg:
                    logger.info(f"  Tool calls: {msg.get('tool_calls')}")

            # Count tokens and wait if necessary
            num_tokens = self.count_tokens(formatted_messages)
            logger.info(f"Token count: {num_tokens}")
            await self.token_bucket.wait_for_tokens(num_tokens)

            if not stream:
                # Non-streaming request
                logger.info("Making non-streaming request")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=False,
                )
                if not response.choices or not response.choices[0].message.content:
                    logger.error(f"Invalid response received: {response}")
                    raise ValueError("Empty or invalid response from LLM")
                
                # Log response
                response_content = response.choices[0].message.content
                logger.info("Response received:")
                logger.info(f"  Content: {response_content[:200]}..." if len(response_content) > 200 else f"  Content: {response_content}")
                return response_content

            # Streaming request
            logger.info("Making streaming request")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                stream=True,
            )

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                logger.error("Empty response from streaming LLM")
                raise ValueError("Empty response from streaming LLM")

            # Log final streamed response
            logger.info("Streamed response completed:")
            logger.info(f"  Content: {full_response[:200]}..." if len(full_response) > 200 else f"  Content: {full_response}")
            return full_response

        except OpenAIError as e:
            logger.error(f"OpenAI API error in ask: {str(e)}")
            logger.error(f"Request details: model={self.model}, base_url={self.base_url}")
            raise handle_api_error(e)
        except Exception as e:
            logger.error(f"Error in _ask_internal: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((OverloadedError, RetryableError, RateLimitError))
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Ask LLM using functions/tools and return the response.
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            return await retry_with_backoff(
                self._ask_tool_internal,
                messages,
                system_msgs,
                timeout,
                tools,
                tool_choice,
                temperature,
                **kwargs
            )
        except RetryError as e:
            logger.error(f"All retry attempts failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

    async def _ask_tool_internal(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Internal implementation of ask_tool method"""
        try:
            # Validate tool_choice
            if tool_choice not in ["none", "auto", "required"]:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                formatted_messages = system_msgs + self.format_messages(messages)
            else:
                formatted_messages = self.format_messages(messages)

            # Log formatted messages for debugging
            logger.info("Request details:")
            logger.info(f"Model: {self.model}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Temperature: {temperature or self.temperature}")
            logger.info(f"Tool choice: {tool_choice}")
            
            logger.info("Formatted messages:")
            for idx, msg in enumerate(formatted_messages):
                logger.info(f"Message {idx + 1}:")
                logger.info(f"  Role: {msg.get('role')}")
                if 'content' in msg:
                    logger.info(f"  Content: {msg.get('content')[:200]}..." if len(msg.get('content', '')) > 200 else f"  Content: {msg.get('content')}")
                if 'tool_calls' in msg:
                    logger.info(f"  Tool calls: {msg.get('tool_calls')}")

            if tools:
                logger.info("Tools configuration:")
                for idx, tool in enumerate(tools):
                    logger.info(f"Tool {idx + 1}:")
                    logger.info(f"  Type: {tool.get('type')}")
                    if 'function' in tool:
                        logger.info(f"  Name: {tool['function'].get('name')}")
                        logger.info(f"  Description: {tool['function'].get('description')}")

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Count tokens and wait if necessary
            num_tokens = self.count_tokens(formatted_messages)
            if tools:
                # Add estimated tokens for tools
                tool_tokens = self.count_tokens([{"content": str(tools)}])
                num_tokens += tool_tokens
                logger.info(f"Token counts - Messages: {num_tokens - tool_tokens}, Tools: {tool_tokens}, Total: {num_tokens}")
            else:
                logger.info(f"Token count: {num_tokens}")

            await self.token_bucket.wait_for_tokens(num_tokens)

            # Set up the completion request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                logger.error(f"Invalid response received: {response}")
                raise ValueError("Invalid or empty response from LLM")

            # Log response summary
            response_msg = response.choices[0].message
            logger.info("Response received:")
            logger.info(f"  Role: {response_msg.role}")
            if hasattr(response_msg, 'content') and response_msg.content:
                logger.info(f"  Content: {response_msg.content[:200]}..." if len(response_msg.content) > 200 else f"  Content: {response_msg.content}")
            if hasattr(response_msg, 'tool_calls') and response_msg.tool_calls:
                logger.info(f"  Tool calls: {response_msg.tool_calls}")

            return response.choices[0].message

        except OpenAIError as e:
            logger.error(f"OpenAI API error in ask_tool: {str(e)}")
            logger.error(f"Request details: model={self.model}, base_url={self.base_url}")
            raise handle_api_error(e)
        except Exception as e:
            logger.error(f"Error in _ask_tool_internal: {e}")
            raise
