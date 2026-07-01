# arxiv-collector

`arxiv_paper_collector.py` queries the arXiv API for papers on RL + LLM topics, deduplicates results, and writes output files.

## Usage

```bash
python3 tools/arxiv-collector/arxiv_paper_collector.py
```

Run from the repo root, or adjust paths accordingly.

## What it does

1. Searches arXiv using queries defined in the `SEARCH_QUERIES` dictionary (topics include RL for code generation, RLHF and alignment, program synthesis with RL, and LLM reasoning).
2. Filters results to papers from 2022 onward (configurable via `start_year`).
3. Deduplicates across queries by arXiv ID.
4. Writes one `PAPERS.md` per topic into `reference/papers/<topic>/`. These files are generated output, not hand-curated.
5. Writes a `papers_database.json` in the collector directory with all results for programmatic access.

`papers_database.json` is large and fully regenerable; it is gitignored.

## Customization

Edit `SEARCH_QUERIES` in the script to add or change search terms. Adjust `start_year` and `max_results` to control the result window and volume.

## Rate limiting

The script waits 3 seconds between arXiv API requests to stay within their rate limits.

## Requirements

Python 3.6+. No external dependencies: uses only `urllib`, `xml.etree.ElementTree`, and `json` from the standard library.

## Future enhancements

- [ ] Citation counts via Semantic Scholar API
- [ ] Automatic GitHub repo detection from paper metadata
- [ ] PDF download and text extraction
- [ ] Cross-version deduplication (arXiv v1, v2, etc.)
- [ ] Email notifications for new papers
- [ ] Integration with reference managers (Zotero, etc.)
