#!/usr/bin/env python3
"""
ArXiv Paper Collector for RL + LLM Research

Automatically fetches and organizes recent papers on:
- Reinforcement Learning for Code Generation
- RLHF and LLM Alignment
- Program Synthesis with RL
- LLM Reasoning and Chain-of-Thought
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import time
from collections import defaultdict
import json
import os


class ArXivCollector:
    """Collector for arXiv papers with rate limiting and filtering."""

    BASE_URL = "http://export.arxiv.org/api/query?"
    RATE_LIMIT_DELAY = 3  # seconds between requests

    SEARCH_QUERIES = {
        "rlhf_code": '(cat:cs.LG OR cat:cs.AI OR cat:cs.SE) AND (all:"reinforcement learning" AND (all:"code generation" OR all:"program synthesis" OR all:"RLHF"))',
        "llm_alignment": '(cat:cs.LG OR cat:cs.AI OR cat:cs.CL) AND (all:"RLHF" OR all:"reinforcement learning from human feedback" OR all:"preference optimization" OR all:"DPO")',
        "code_synthesis": '(cat:cs.SE OR cat:cs.PL OR cat:cs.AI) AND (all:"program synthesis" AND all:"reinforcement learning")',
        "llm_reasoning": '(cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND (all:"chain-of-thought" OR all:"reasoning" AND all:"reinforcement learning" AND all:"language model")',
        "alphaco_related": '(cat:cs.AI OR cat:cs.SE) AND (all:"AlphaCode" OR all:"CodeRL" OR all:"competitive programming" AND all:"reinforcement learning")',
    }

    def __init__(self, start_year=2022, max_results=100):
        self.start_year = start_year
        self.max_results = max_results
        self.papers = defaultdict(list)

    def search_arxiv(self, query, category_name):
        """Search arXiv with the given query."""
        print(f"\n🔍 Searching: {category_name}")
        print(f"Query: {query[:100]}...")

        params = {
            'search_query': query,
            'start': 0,
            'max_results': self.max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        url = self.BASE_URL + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')
            return data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def parse_response(self, xml_data):
        """Parse arXiv API XML response."""
        papers = []
        root = ET.fromstring(xml_data)

        # Namespace for Atom feed
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}

        for entry in root.findall('atom:entry', ns):
            # Extract paper details
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text[:10]  # YYYY-MM-DD

            # Get arXiv ID
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]

            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)

            # Get categories
            categories = []
            for category in entry.findall('atom:category', ns):
                categories.append(category.get('term'))

            # Filter by year
            year = int(published[:4])
            if year >= self.start_year:
                papers.append({
                    'title': title,
                    'authors': authors,
                    'summary': summary,
                    'arxiv_id': arxiv_id,
                    'published': published,
                    'categories': categories,
                    'url': f"https://arxiv.org/abs/{arxiv_id}",
                    'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                })

        return papers

    def collect_all(self):
        """Collect papers for all search queries."""
        for category, query in self.SEARCH_QUERIES.items():
            print(f"\n{'='*60}")
            xml_data = self.search_arxiv(query, category)

            if xml_data:
                papers = self.parse_response(xml_data)
                self.papers[category].extend(papers)
                print(f"✓ Found {len(papers)} papers (>={self.start_year})")

            # Rate limiting
            time.sleep(self.RATE_LIMIT_DELAY)

        return self.papers

    def remove_duplicates(self):
        """Remove duplicate papers across categories."""
        seen_ids = set()
        for category in self.papers:
            unique_papers = []
            for paper in self.papers[category]:
                if paper['arxiv_id'] not in seen_ids:
                    seen_ids.add(paper['arxiv_id'])
                    unique_papers.append(paper)
            self.papers[category] = unique_papers

    def format_markdown(self, paper):
        """Format a single paper as markdown."""
        authors_str = ", ".join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += " et al."

        return f"""### {paper['title']}
**Authors**: {authors_str}
**Published**: {paper['published']}
**arXiv**: [{paper['arxiv_id']}]({paper['url']})
**PDF**: [Download]({paper['pdf_url']})

**Abstract**: {paper['summary'][:300]}...

---
"""

    def save_to_markdown(self, output_dir="../Modern-RL-Research"):
        """Save collected papers to markdown files."""
        category_mapping = {
            'rlhf_code': 'LLM-Code-Generation',
            'llm_alignment': 'RLHF-and-Alignment',
            'code_synthesis': 'LLM-RL-Program-Synthesis',
            'llm_reasoning': 'LLM-Code-Generation',
            'alphaco_related': 'LLM-RL-Program-Synthesis'
        }

        organized_papers = defaultdict(list)
        for category, papers in self.papers.items():
            target_dir = category_mapping.get(category, 'LLM-Code-Generation')
            organized_papers[target_dir].extend(papers)

        # Save to files
        for dir_name, papers in organized_papers.items():
            output_file = os.path.join(output_dir, dir_name, "PAPERS.md")

            # Sort by date (newest first)
            papers.sort(key=lambda x: x['published'], reverse=True)

            with open(output_file, 'w') as f:
                f.write(f"# Recent Papers - {dir_name}\n\n")
                f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n")
                f.write(f"Total Papers: {len(papers)}\n\n")
                f.write("---\n\n")

                # Group by year
                by_year = defaultdict(list)
                for paper in papers:
                    year = paper['published'][:4]
                    by_year[year].append(paper)

                for year in sorted(by_year.keys(), reverse=True):
                    f.write(f"## {year}\n\n")
                    for paper in by_year[year]:
                        f.write(self.format_markdown(paper))

            print(f"✓ Saved {len(papers)} papers to {output_file}")

    def save_to_json(self, output_file="papers_database.json"):
        """Save all papers to JSON for programmatic access."""
        with open(output_file, 'w') as f:
            json.dump(dict(self.papers), f, indent=2)
        print(f"\n✓ Saved database to {output_file}")

    def print_summary(self):
        """Print summary statistics."""
        print(f"\n{'='*60}")
        print("📊 COLLECTION SUMMARY")
        print(f"{'='*60}")

        total = sum(len(papers) for papers in self.papers.values())
        print(f"\nTotal papers collected: {total}")

        for category, papers in self.papers.items():
            print(f"  • {category}: {len(papers)} papers")

        # Year distribution
        year_counts = defaultdict(int)
        for papers in self.papers.values():
            for paper in papers:
                year = paper['published'][:4]
                year_counts[year] += 1

        print(f"\nPapers by year:")
        for year in sorted(year_counts.keys(), reverse=True):
            print(f"  • {year}: {year_counts[year]} papers")


def main():
    """Main execution function."""
    print("="*60)
    print("📚 ArXiv Paper Collector for RL + LLM Research")
    print("="*60)

    # Initialize collector
    collector = ArXivCollector(start_year=2022, max_results=150)

    # Collect papers
    print("\n🚀 Starting collection...")
    collector.collect_all()

    # Remove duplicates
    print("\n🔄 Removing duplicates...")
    collector.remove_duplicates()

    # Print summary
    collector.print_summary()

    # Save results
    print("\n💾 Saving results...")

    # Get the script's directory and compute the correct output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "Modern-RL-Research")
    output_dir = os.path.abspath(output_dir)

    collector.save_to_markdown(output_dir=output_dir)

    json_output = os.path.join(script_dir, "papers_database.json")
    collector.save_to_json(output_file=json_output)

    print("\n✅ Collection complete!")
    print(f"\nCheck the following directories for papers:")
    print(f"  • {output_dir}/LLM-Code-Generation/PAPERS.md")
    print(f"  • {output_dir}/LLM-RL-Program-Synthesis/PAPERS.md")
    print(f"  • {output_dir}/RLHF-and-Alignment/PAPERS.md")


if __name__ == "__main__":
    main()
