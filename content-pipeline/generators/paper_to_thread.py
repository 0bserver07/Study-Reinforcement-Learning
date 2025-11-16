#!/usr/bin/env python3
"""
Paper to Twitter Thread Generator

Converts research papers into engaging Twitter threads.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client_from_env, ContentGenerator


def load_papers_database(db_path: str = "../../scripts/papers_database.json") -> dict:
    """Load the papers database."""
    full_path = Path(__file__).parent / db_path
    with open(full_path, 'r') as f:
        return json.load(f)


def find_paper(arxiv_id: str, database: dict) -> dict:
    """Find a paper by arXiv ID."""
    for category, papers in database.items():
        for paper in papers:
            if paper['arxiv_id'] == arxiv_id:
                return paper
    raise ValueError(f"Paper {arxiv_id} not found in database")


def save_thread(tweets: list, paper: dict, output_dir: str = "../outputs/threads"):
    """Save thread to file."""
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    arxiv_id_clean = paper['arxiv_id'].replace('/', '-').replace('.', '-')
    filename = f"{date_str}_{arxiv_id_clean}_thread.txt"

    # Format thread
    content = f"""Twitter Thread - {paper['title']}
Generated: {datetime.now().isoformat()}
Paper: {paper['url']}

{'='*60}

"""
    for i, tweet in enumerate(tweets, 1):
        char_count = len(tweet)
        content += f"Tweet {i}/{len(tweets)} ({char_count} chars):\n{tweet}\n\n"

    content += f"""{'='*60}

Total tweets: {len(tweets)}
Ready to post!

Note: Review and edit before posting. Add images/GIFs for better engagement.
"""

    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def generate_thread(
    arxiv_id: str,
    provider: str = "anthropic",
    model: str = None,
    max_tweets: int = 10
):
    """Generate a Twitter thread from a paper."""
    print(f"📚 Loading paper database...")
    database = load_papers_database()

    print(f"🔍 Finding paper {arxiv_id}...")
    paper = find_paper(arxiv_id, database)
    print(f"✓ Found: {paper['title'][:60]}...")

    print(f"\n🤖 Initializing {provider} ({model or 'default model'})...")
    client = get_client_from_env(provider, model)
    generator = ContentGenerator(client)

    print(f"\n🐦 Generating Twitter thread (max {max_tweets} tweets)...")
    tweets = generator.generate_twitter_thread(paper, max_tweets=max_tweets)

    print(f"\n{'='*60}")
    print(f"GENERATED THREAD ({len(tweets)} tweets):")
    print('='*60 + "\n")

    for i, tweet in enumerate(tweets, 1):
        char_count = len(tweet)
        status = "✓" if char_count <= 280 else f"⚠️  TOO LONG ({char_count - 280} over)"
        print(f"Tweet {i}/{len(tweets)} ({char_count} chars) {status}")
        print(tweet)
        print()

    print('='*60)

    print(f"\n💾 Saving thread...")
    filepath = save_thread(tweets, paper)
    print(f"✓ Saved to: {filepath}")

    # Check for issues
    long_tweets = [i for i, t in enumerate(tweets, 1) if len(t) > 280]
    if long_tweets:
        print(f"\n⚠️  Warning: Tweets {long_tweets} exceed 280 characters. Edit before posting.")
    else:
        print(f"\n✓ All tweets within character limit!")

    print(f"\n📊 Stats:")
    print(f"  - Total tweets: {len(tweets)}")
    print(f"  - Average length: {sum(len(t) for t in tweets) / len(tweets):.0f} chars")
    print(f"  - Shortest: {min(len(t) for t in tweets)} chars")
    print(f"  - Longest: {max(len(t) for t in tweets)} chars")

    return tweets, filepath


def main():
    parser = argparse.ArgumentParser(
        description="Generate Twitter thread from research paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate thread from CodeRL paper
  python3 paper_to_thread.py --paper-id 2207.01780

  # Use GPT-4 for thread generation
  python3 paper_to_thread.py --paper-id 2203.07814 --provider openai

  # Create shorter thread (5 tweets)
  python3 paper_to_thread.py --paper-id 2412.20367 --max-tweets 5

  # Use DeepSeek (very cheap for threads!)
  python3 paper_to_thread.py --paper-id 2207.01780 --provider deepseek

Popular paper IDs:
  - 2207.01780: CodeRL
  - 2203.07814: AlphaCode
  - 2412.20367: RL for Code Survey
  - 2510.08256: Mix-DPO
  - 2511.04286: Efficient RLHF
        """
    )

    parser.add_argument(
        '--paper-id',
        required=True,
        help='arXiv ID of the paper'
    )

    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openai', 'deepseek'],
        default='anthropic',
        help='LLM provider (default: anthropic)'
    )

    parser.add_argument(
        '--model',
        help='Specific model to use'
    )

    parser.add_argument(
        '--max-tweets',
        type=int,
        default=10,
        help='Maximum number of tweets (default: 10)'
    )

    args = parser.parse_args()

    try:
        generate_thread(
            arxiv_id=args.paper_id,
            provider=args.provider,
            model=args.model,
            max_tweets=args.max_tweets
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
