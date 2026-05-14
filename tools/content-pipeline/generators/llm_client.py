#!/usr/bin/env python3
"""
Unified LLM Client for Content Generation

Supports multiple providers:
- Anthropic (Claude)
- OpenAI (GPT-4, o1)
- DeepSeek
- Any OpenAI-compatible API
"""

import os
import json
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class Message:
    """Message structure for chat completions."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7


class LLMClient:
    """Unified client for multiple LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.config.provider == "anthropic":
            return self._init_anthropic()
        elif self.config.provider == "openai":
            return self._init_openai()
        elif self.config.provider == "deepseek":
            return self._init_deepseek()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.config.api_key)
            if self.config.base_url:
                client.base_url = self.config.base_url
            return client
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _init_deepseek(self):
        """Initialize DeepSeek client (OpenAI-compatible)."""
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url="https://api.deepseek.com"
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def generate(
        self,
        messages: List[Message],
        stream: bool = False
    ) -> str:
        """Generate completion from messages."""
        if self.config.provider == "anthropic":
            return self._generate_anthropic(messages, stream)
        elif self.config.provider in ["openai", "deepseek"]:
            return self._generate_openai_compatible(messages, stream)

    def _generate_anthropic(
        self,
        messages: List[Message],
        stream: bool
    ) -> str:
        """Generate using Anthropic API."""
        # Separate system message
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": chat_messages,
        }

        if system_message:
            kwargs["system"] = system_message

        if stream:
            return self._stream_anthropic(kwargs)
        else:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text

    def _stream_anthropic(self, kwargs) -> str:
        """Stream Anthropic response."""
        full_response = ""
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text
        print()  # Newline after stream
        return full_response

    def _generate_openai_compatible(
        self,
        messages: List[Message],
        stream: bool
    ) -> str:
        """Generate using OpenAI-compatible API."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        if stream:
            return self._stream_openai(formatted_messages)
        else:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=formatted_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content

    def _stream_openai(self, messages) -> str:
        """Stream OpenAI-compatible response."""
        full_response = ""
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_response += text
        print()  # Newline after stream
        return full_response


class ContentGenerator:
    """High-level interface for content generation."""

    def __init__(self, client: LLMClient):
        self.client = client

    def generate_blog_post(
        self,
        paper_data: Dict,
        tone: str = "accessible",
        target_length: int = 1500
    ) -> str:
        """Generate blog post from paper data."""
        system_prompt = self._get_blog_system_prompt(tone, target_length)
        user_prompt = self._format_paper_for_blog(paper_data)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]

        return self.client.generate(messages)

    def generate_twitter_thread(
        self,
        paper_data: Dict,
        max_tweets: int = 10
    ) -> List[str]:
        """Generate Twitter thread from paper data."""
        system_prompt = self._get_thread_system_prompt(max_tweets)
        user_prompt = self._format_paper_for_thread(paper_data)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]

        response = self.client.generate(messages)
        return self._parse_thread(response)

    def generate_summary(
        self,
        paper_data: Dict,
        max_length: int = 300
    ) -> str:
        """Generate concise summary."""
        system_prompt = f"""You are a research summarizer. Create clear, concise summaries of academic papers.

Maximum length: {max_length} words.
Focus on: Problem, Method, Results, Impact."""

        user_prompt = f"""Summarize this paper:

Title: {paper_data['title']}
Authors: {', '.join(paper_data['authors'][:5])}
Abstract: {paper_data['summary']}

Create a {max_length}-word summary covering: problem addressed, methodology, key results, and why it matters."""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]

        return self.client.generate(messages)

    def _get_blog_system_prompt(self, tone: str, target_length: int) -> str:
        """Get system prompt for blog post generation."""
        tone_descriptions = {
            "technical": "formal, detailed, assumes expert knowledge",
            "accessible": "clear, friendly, explains concepts simply",
            "enthusiastic": "excited, engaging, story-driven",
            "professional": "business-focused, practical applications"
        }

        return f"""You are a technical writer specializing in AI and machine learning.

Writing style: {tone_descriptions.get(tone, tone)}
Target length: ~{target_length} words
Format: Blog post with:
- Compelling title
- Hook/introduction
- Clear explanation of problem
- Technical details (appropriate to tone)
- Key insights
- Practical implications
- Conclusion

Use markdown formatting. Include code examples if relevant. Cite the paper."""

    def _format_paper_for_blog(self, paper_data: Dict) -> str:
        """Format paper data for blog post prompt."""
        authors_str = ", ".join(paper_data['authors'][:3])
        if len(paper_data['authors']) > 3:
            authors_str += " et al."

        return f"""Write a blog post about this paper:

**Title**: {paper_data['title']}
**Authors**: {authors_str}
**Published**: {paper_data['published']}
**arXiv**: {paper_data['arxiv_id']}

**Abstract**:
{paper_data['summary']}

**Your task**:
1. Create an engaging blog post explaining this research
2. Make it understandable to your target audience
3. Highlight why this matters
4. Include practical implications if applicable
5. Add a call-to-action at the end

Begin writing:"""

    def _get_thread_system_prompt(self, max_tweets: int) -> str:
        """Get system prompt for Twitter thread."""
        return f"""You are a social media expert creating engaging Twitter threads about AI research.

Guidelines:
- Maximum {max_tweets} tweets
- Each tweet max 280 characters
- First tweet: Hook that grabs attention
- Middle tweets: Explain key points
- Last tweet: Takeaway + CTA
- Use simple language
- Add emoji strategically (but not excessively)
- Include relevant hashtags

Format: Number each tweet (1/, 2/, etc.)"""

    def _format_paper_for_thread(self, paper_data: Dict) -> str:
        """Format paper data for thread prompt."""
        return f"""Create a Twitter thread about this paper:

Title: {paper_data['title']}
Authors: {', '.join(paper_data['authors'][:3])}
Published: {paper_data['published']}

Summary: {paper_data['summary'][:500]}...

Make it engaging and accessible!"""

    def _parse_thread(self, response: str) -> List[str]:
        """Parse thread response into individual tweets."""
        tweets = []
        lines = response.strip().split('\n')

        current_tweet = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with tweet number
            if line.startswith(tuple(f"{i}/" for i in range(1, 51))):
                if current_tweet:
                    tweets.append(current_tweet.strip())
                # Remove the number prefix
                current_tweet = line.split('/', 1)[1].strip()
            else:
                current_tweet += " " + line

        # Add last tweet
        if current_tweet:
            tweets.append(current_tweet.strip())

        return tweets


# Convenience functions for quick setup

def get_client_from_env(
    provider: str = "anthropic",
    model: Optional[str] = None
) -> LLMClient:
    """Create client from environment variables."""
    # Load API key from environment
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    api_key = os.getenv(key_map.get(provider, f"{provider.upper()}_API_KEY"))
    if not api_key:
        raise ValueError(f"API key not found for {provider}. Set {key_map.get(provider)} in environment.")

    # Default models
    if model is None:
        model_defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "deepseek": "deepseek-chat",
        }
        model = model_defaults.get(provider, "default")

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key
    )

    return LLMClient(config)


if __name__ == "__main__":
    # Example usage
    print("LLM Client Library")
    print("==================")
    print("\nSupported providers: anthropic, openai, deepseek")
    print("\nExample usage:")
    print("""
from llm_client import get_client_from_env, ContentGenerator

# Initialize
client = get_client_from_env("anthropic", "claude-sonnet-4-20250514")
generator = ContentGenerator(client)

# Load paper data
paper = {...}  # From papers_database.json

# Generate content
blog = generator.generate_blog_post(paper, tone="accessible")
thread = generator.generate_twitter_thread(paper)
summary = generator.generate_summary(paper)
""")
