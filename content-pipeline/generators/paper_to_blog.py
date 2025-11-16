#!/usr/bin/env python3
"""
Paper to Blog Post Generator

Converts research papers from your collection into blog posts.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client_from_env, ContentGenerator


def load_papers_database(db_path: str = "../../scripts/papers_database.json") -> dict:
    """Load the papers database."""
    full_path = Path(__file__).parent / db_path
    with open(full_path, 'r') as f:
        return json.load(f)


def find_paper(arxiv_id: str, database: dict) -> dict:
    """Find a paper by arXiv ID in the database."""
    for category, papers in database.items():
        for paper in papers:
            if paper['arxiv_id'] == arxiv_id:
                return paper
    raise ValueError(f"Paper {arxiv_id} not found in database")


def save_blog_post(content: str, paper: dict, output_dir: str = "../outputs/blogs"):
    """Save blog post to file."""
    # Create output directory
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    arxiv_id_clean = paper['arxiv_id'].replace('/', '-').replace('.', '-')
    filename = f"{date_str}_{arxiv_id_clean}_blog.md"

    # Add metadata header
    full_content = f"""---
title: Blog Post - {paper['title']}
arxiv_id: {paper['arxiv_id']}
authors: {', '.join(paper['authors'][:3])}
generated: {datetime.now().isoformat()}
---

{content}

---

**Source Paper**: [{paper['arxiv_id']}]({paper['url']})
**Published**: {paper['published']}
"""

    # Save file
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(full_content)

    return filepath


def generate_blog_post(
    arxiv_id: str,
    provider: str = "anthropic",
    model: str = None,
    tone: str = "accessible",
    target_length: int = 1500,
    stream: bool = False
):
    """Generate a blog post from a paper."""
    print(f"📚 Loading paper database...")
    database = load_papers_database()

    print(f"🔍 Finding paper {arxiv_id}...")
    paper = find_paper(arxiv_id, database)
    print(f"✓ Found: {paper['title'][:60]}...")

    print(f"\n🤖 Initializing {provider} ({model or 'default model'})...")
    client = get_client_from_env(provider, model)
    generator = ContentGenerator(client)

    print(f"\n✍️  Generating blog post (tone: {tone}, length: ~{target_length} words)...")
    if stream:
        print("\n" + "="*60)
        print("STREAMING OUTPUT:")
        print("="*60 + "\n")

    blog_content = generator.generate_blog_post(
        paper,
        tone=tone,
        target_length=target_length
    )

    if not stream:
        print("\n" + "="*60)
        print("GENERATED BLOG POST:")
        print("="*60 + "\n")
        print(blog_content)
        print("\n" + "="*60 + "\n")

    print(f"💾 Saving blog post...")
    filepath = save_blog_post(blog_content, paper)
    print(f"✓ Saved to: {filepath}")

    # Print stats
    word_count = len(blog_content.split())
    print(f"\n📊 Stats:")
    print(f"  - Word count: {word_count}")
    print(f"  - Character count: {len(blog_content)}")
    print(f"  - Target length: {target_length} words")
    print(f"  - Difference: {word_count - target_length:+d} words")

    return blog_content, filepath


def main():
    parser = argparse.ArgumentParser(
        description="Generate blog post from research paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate blog post from CodeRL paper using Claude
  python3 paper_to_blog.py --paper-id 2207.01780 --provider anthropic

  # Use GPT-4 with professional tone
  python3 paper_to_blog.py --paper-id 2203.07814 --provider openai --tone professional

  # Use DeepSeek (cheap!) for technical audience
  python3 paper_to_blog.py --paper-id 2412.20367 --provider deepseek --tone technical

  # Stream output as it's generated
  python3 paper_to_blog.py --paper-id 2207.01780 --stream

Popular paper IDs from our collection:
  - 2207.01780: CodeRL (Salesforce)
  - 2203.07814: AlphaCode (DeepMind)
  - 2412.20367: Survey on RL for Code Generation
  - 2510.08256: Mix-DPO
        """
    )

    parser.add_argument(
        '--paper-id',
        required=True,
        help='arXiv ID of the paper (e.g., 2207.01780)'
    )

    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openai', 'deepseek'],
        default='anthropic',
        help='LLM provider to use (default: anthropic)'
    )

    parser.add_argument(
        '--model',
        help='Specific model to use (optional, uses provider default if not specified)'
    )

    parser.add_argument(
        '--tone',
        choices=['technical', 'accessible', 'enthusiastic', 'professional'],
        default='accessible',
        help='Writing tone (default: accessible)'
    )

    parser.add_argument(
        '--length',
        type=int,
        default=1500,
        help='Target length in words (default: 1500)'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream output as it generates'
    )

    args = parser.parse_args()

    try:
        generate_blog_post(
            arxiv_id=args.paper_id,
            provider=args.provider,
            model=args.model,
            tone=args.tone,
            target_length=args.length,
            stream=args.stream
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
