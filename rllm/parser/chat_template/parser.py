from .utils import PARSER_TEST_MESSAGES


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, disable_thinking=False) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            # Add Falcon detection
            if "falcon" in model_name:
                print(f"Using FalconChatTemplateParser for {tokenizer.name_or_path}")
                return FalconChatTemplateParser(tokenizer)
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                print(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                print(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer)
        print(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser


class FalconChatTemplateParser(ChatTemplateParser):
    """Parser for Falcon's previous chat template (ChatML-style with tools)."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token or ""
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        self.generation_prompt = "<|im_start|>assistant"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, tools=None, **kwargs):
        result = ""

        # Add BOS token
        result += self.bos_token

        # Handle tools
        if tools:
            result += self.format_tools_section(messages, tools)
            # Skip system message if already included in tools section
            remaining_messages = messages[1:] if messages and messages[0]["role"] == "system" else messages
        else:
            # No tools: process system message normally
            if messages and messages[0]["role"] == "system":
                result += self.parse_system(messages[0])
                remaining_messages = messages[1:]
            else:
                remaining_messages = messages

        # Parse remaining messages
        for message in remaining_messages:
            if message["role"] == "system":
                continue  # Already handled
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt

        return result

    def format_tools_section(self, messages, tools):
        """Format tools section according to previous Falcon template."""
        import json

        result = self.system_token

        # Include system message content if present
        if messages and messages[0]["role"] == "system":
            result += messages[0]["content"] + "\n\n"

        # Add tools instructions
        result += "You are a function calling AI model. You are provided with function signature within <tools> </tools> XML tags. "
        result += "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n"
        result += "<tools>\n"

        for tool in tools:
            result += "[" + json.dumps(tool) + "]"

        result += "\n</tools>\n"
        result += "For each function call, return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n"
        result += "<tool_call>\n"
        result += "{'arguments': <args-dict>, 'name': <function-name>}\n"
        result += "</tool_call>\n"

        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        # Tool responses as user messages in previous template
        return self.user_token + message["content"] + self.eot_token


class FalconChatTemplateParserV1(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token or ""
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"
        self.system_token = "<|system|>\n"
        self.user_token = "<|user|>\n"
        self.assistant_token = "<|assistant|>\n"
        self.generation_prompt = self.assistant_token

        # Tool call tokens
        self.tool_call_start = "<tool_call>\n"
        self.tool_call_end = "\n</tool_call>\n"
        self.tool_response_start = "<tool_response>\n"
        self.tool_response_end = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, tools=None, **kwargs):
        result = ""

        # Handle system message
        if messages and messages[0]["role"] == "system":
            result += self.parse_system(messages[0])
            remaining_messages = messages[1:]
        else:
            # Default system prompt if none provided
            result += self.system_token + "You are Falcon, created by Technology Innovation Institute (TII). You are a helpful assistant.\n"
            remaining_messages = messages

        # Add tools section if provided
        if tools:
            result += self.format_tools(tools)

        # Parse remaining messages
        for idx, message in enumerate(remaining_messages):
            is_last = idx == len(remaining_messages) - 1

            if message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, tools=tools, is_last=is_last)
            elif message["role"] == "tool":
                result += self.parse_tool(message, remaining_messages, idx)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt

        return result

    def format_tools(self, tools):
        """Format tools section according to Falcon template."""
        import json

        result = "# Tools\n"
        result += "You may call one or more functions to assist with the user query.\n"
        result += "You are provided with function signatures within <tools></tools> XML tags.\n"
        result += "<tools>\n"

        for tool in tools:
            result += json.dumps(tool) + "\n"

        result += "</tools>\n"
        result += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        result += "<tool_call>\n"
        result += '{"name": <function-name>, "arguments": <args-json-object>}\n'
        result += "</tool_call>\n"

        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + "\n"

    def parse_user(self, message):
        return self.user_token + message["content"] + "\n"

    def parse_assistant(self, message, tools=None, is_last=False):
        import json

        result = self.assistant_token

        # Add content if present
        if message.get("content"):
            result += message["content"] + "\n"

        # Add tool calls if present
        if tools and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                # Handle both formats: direct tool_call or tool_call.function
                if hasattr(tool_call, "function"):
                    tool_call = tool_call.function
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    tool_call = tool_call["function"]

                result += self.tool_call_start
                result += '{"name": "' + tool_call["name"] + '", "arguments":'

                # Handle arguments as string or dict
                if isinstance(tool_call["arguments"], str):
                    result += tool_call["arguments"]
                else:
                    result += json.dumps(tool_call["arguments"])

                result += "}\n"
                result += self.tool_call_end

        # Add EOS token if not last message
        if not is_last:
            result += self.eos_token + "\n"
        else:
            result += self.eos_token

        return result

    def parse_tool(self, message, remaining_messages, idx):
        """Parse tool response message."""
        result = ""

        # Check if this is the first tool message in a sequence
        if idx == 0 or remaining_messages[idx - 1]["role"] != "tool":
            result += self.user_token

        result += "\n" + self.tool_response_start + message["content"] + self.tool_response_end

        # Check if this is the last tool message in a sequence
        if idx == len(remaining_messages) - 1 or remaining_messages[idx + 1]["role"] != "tool":
            result += "\n"

        return result


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.eos_token + self.assistant_token + "<think>\n"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"]

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eos_token


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\\n\\n</think>\\n\\n"
        self.generation_prompt = self.assistant_token

        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"

        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        result = self.assistant_token + message["content"] + self.eot_token
        return result

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # took tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token
