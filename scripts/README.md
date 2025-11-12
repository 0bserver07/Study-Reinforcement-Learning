# Research Collection Scripts

Automated tools for collecting and organizing RL + LLM research papers.

## Scripts

### `arxiv_paper_collector.py`

Automatically fetches recent papers from arXiv on:
- Reinforcement Learning for Code Generation
- RLHF and LLM Alignment
- Program Synthesis with RL
- LLM Reasoning with RL
- AlphaCode and related work

**Usage:**
```bash
cd scripts
python3 arxiv_paper_collector.py
```

**What it does:**
1. Searches arXiv with targeted queries
2. Filters papers from 2022-2025
3. Removes duplicates
4. Organizes by topic
5. Generates markdown files with paper details
6. Saves JSON database for programmatic access

**Output:**
- `../Modern-RL-Research/*/PAPERS.md` - Organized paper lists
- `papers_database.json` - Complete database

**Customization:**
Edit the `SEARCH_QUERIES` dictionary in the script to add/modify search terms.

**Rate Limiting:**
The script respects arXiv's rate limits (3 seconds between requests).

## Requirements

No external dependencies! Uses only Python standard library:
- `urllib` - For API requests
- `xml.etree.ElementTree` - For XML parsing
- `json` - For data export

Python 3.6+ required.

## Tips

**Updating the collection:**
```bash
# Run monthly to keep papers up-to-date
python3 arxiv_paper_collector.py
```

**Filtering results:**
Adjust `start_year` and `max_results` parameters in the script.

**Custom searches:**
Add your own queries to `SEARCH_QUERIES` dictionary.

**JSON output:**
Use `papers_database.json` for custom analysis:
```python
import json
with open('papers_database.json') as f:
    papers = json.load(f)
```

## Future Enhancements

Potential additions:
- [ ] Paper citation counts via Semantic Scholar API
- [ ] Automatic GitHub repo detection
- [ ] PDF download and text extraction
- [ ] Duplicate detection across versions (v1, v2, etc.)
- [ ] Email notifications for new papers
- [ ] Integration with paper management tools (Zotero, etc.)

---

*Part of the Study-Reinforcement-Learning repository*
