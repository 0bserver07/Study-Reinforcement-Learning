# LLM Content Pipeline

Automated content generation pipeline using reasoning models (Kimi K2, Claude, GPT-4, DeepSeek R1, etc.) to write about RL and LLM research.

## 🎯 What This Does

Transform your research collection into:
- Blog posts
- Twitter/X threads
- LinkedIn articles
- Paper summaries
- Tutorial content
- Newsletter sections

## 📂 Directory Structure

```
content-pipeline/
├── config/                    # API keys and settings
├── templates/                 # Content templates
├── generators/                # LLM integration scripts
├── outputs/                   # Generated content
└── workflows/                 # End-to-end pipelines
```

## 🚀 Quick Start

### 1. Set Up API Keys

Create `config/api_keys.env`:
```bash
# Add your API keys (never commit this file!)
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
# Add others as needed
```

### 2. Choose a Content Type

```bash
# Generate a blog post from a paper
python3 generators/paper_to_blog.py --paper-id 2207.01780 --model claude

# Create a Twitter thread
python3 generators/paper_to_thread.py --paper-id 2203.07814 --model gpt4

# Weekly research summary
python3 workflows/weekly_digest.py
```

### 3. Review and Edit

Generated content appears in `outputs/` - always review before publishing!

## 🤖 Supported Models

### Reasoning Models (Best for Complex Content)
- **Claude Sonnet 4+** - Excellent for technical writing, explanations
- **OpenAI o1/o3** - Strong reasoning, good for analysis
- **DeepSeek R1** - Open, competitive performance
- **Kimi K2** - Chinese model with strong reasoning
- **Qwen 3** - Multilingual support

### Fast Models (For Drafts/Social Media)
- **Claude Haiku** - Quick, cost-effective
- **GPT-4o-mini** - Fast, cheap
- **Gemini Flash** - Good for quick tasks

## 📝 Content Templates

### Blog Post Template
- Introduction with hook
- Problem statement
- Technical explanation
- Key insights
- Practical implications
- Conclusion and next steps

### Twitter Thread Template
- 1: Hook tweet (attention grabber)
- 2-3: Problem/context
- 4-6: Main points
- 7-8: Key takeaways
- 9: Call to action

### Paper Summary Template
- TL;DR (2-3 sentences)
- Problem addressed
- Methodology
- Results
- Why it matters
- Limitations

## 🛠️ Tools Included

### Core Generators
- `paper_to_blog.py` - Convert papers to blog posts
- `paper_to_thread.py` - Create Twitter threads
- `paper_summary.py` - Generate summaries
- `weekly_digest.py` - Compile weekly updates

### Utilities
- `compare_models.py` - Test different models
- `optimize_prompt.py` - Improve prompts
- `cost_calculator.py` - Estimate API costs

## 💰 Cost Management

Approximate costs per content piece:

| Model | Blog Post | Thread | Summary |
|-------|-----------|--------|---------|
| Claude Sonnet | $0.15 | $0.03 | $0.02 |
| GPT-4 | $0.20 | $0.04 | $0.03 |
| DeepSeek | $0.01 | $0.002 | $0.001 |
| Claude Haiku | $0.02 | $0.005 | $0.003 |

**Budget-friendly approach**: Use reasoning models for final draft, fast models for initial drafts.

## 📊 Workflow Examples

### Weekly Research Post
1. Script fetches new papers from `papers_database.json`
2. Model generates summaries
3. Creates combined post highlighting top 3-5 papers
4. Outputs markdown ready for Medium/Dev.to

### Deep Dive Article
1. Select important paper
2. Model reads paper + context
3. Generates detailed explanation
4. Adds code examples if applicable
5. Creates graphics prompts
6. Outputs full article

### Social Media Campaign
1. Choose research topic
2. Generate thread explaining concept
3. Create LinkedIn version
4. Draft accompanying images
5. Schedule posts

## 🎨 Customization

### Tone Settings
```python
TONES = {
    "technical": "Formal, academic, detailed",
    "accessible": "Clear, friendly, minimal jargon",
    "enthusiastic": "Excited, engaging, story-driven",
    "professional": "Business-focused, practical",
}
```

### Content Styles
- **Tutorial**: Step-by-step, beginner-friendly
- **Analysis**: Deep dive, expert-level
- **News**: Timely, concise, newsworthy
- **Opinion**: Personal take, thought-provoking

## 🔄 Automation

### Daily
- Scan for new papers
- Generate quick summaries

### Weekly
- Compile digest of top papers
- Generate thread of key insights

### Monthly
- Create comprehensive review
- Analyze trends

## 📱 Publishing Integration

### Supported Platforms
- **Medium**: API integration
- **Dev.to**: API integration
- **Twitter/X**: API via tweepy
- **LinkedIn**: API integration
- **Ghost**: API integration
- **Hugo/Jekyll**: Direct markdown export

## 🧪 Testing Your Pipeline

```bash
# Test with a single paper
python3 generators/paper_to_blog.py \
  --paper-id 2207.01780 \
  --model claude \
  --tone accessible \
  --test-mode

# Compare different models
python3 tools/compare_models.py \
  --paper-id 2203.07814 \
  --models claude,gpt4,deepseek

# Estimate costs
python3 tools/cost_calculator.py \
  --content-type blog \
  --model claude \
  --papers-per-week 3
```

## 📚 Best Practices

### Content Quality
1. **Always review** - LLMs make mistakes
2. **Add your voice** - Edit for personality
3. **Verify facts** - Check paper claims
4. **Add examples** - Include code/demos
5. **Cite sources** - Link to papers

### API Usage
1. **Cache results** - Don't regenerate unnecessarily
2. **Use cheaper models** for drafts
3. **Batch requests** when possible
4. **Monitor costs** regularly
5. **Rate limit** to avoid bans

### Publishing
1. **Schedule posts** for consistency
2. **Cross-post** to multiple platforms
3. **Track engagement** - see what works
4. **Iterate prompts** based on feedback
5. **Build series** around themes

## 🎯 Content Ideas

### Regular Series
- **Paper Monday**: Weekly paper deep dive
- **Trend Thursday**: What's hot in RL+LLM
- **Tutorial Tuesday**: How-to guides
- **Friday Finds**: Cool discoveries

### One-Off Content
- "Top 10 RLHF Papers of 2024"
- "AlphaCode Explained Simply"
- "DPO vs PPO: A Developer's Guide"
- "Building Your First Code LLM"

### Engagement Drivers
- Controversial takes (backed by research)
- Prediction threads
- Paper battle threads (Model A vs B)
- "Explain like I'm 5" series

## 🔮 Advanced Features

### Multi-Modal
- Generate image prompts for DALL-E/Midjourney
- Create diagrams with code
- Video script generation

### Interactive
- Generate quiz questions
- Create discussion prompts
- Build interactive demos

### Analytics
- Track which topics perform best
- A/B test different tones
- Optimize posting times

## 🤝 Contributing

Ideas for improvements:
- [ ] Video script generator
- [ ] Podcast outline creator
- [ ] Presentation slide generator
- [ ] Newsletter template
- [ ] SEO optimizer

---

## 📖 Example Usage

```bash
# Full workflow: Paper to published post
./workflows/paper_to_publication.sh \
  --paper 2207.01780 \
  --platform medium \
  --review manual

# Weekly digest
./workflows/weekly_digest.sh \
  --date 2025-01-06 \
  --output newsletter

# Social campaign
./workflows/launch_campaign.sh \
  --topic "AlphaCode" \
  --platforms twitter,linkedin \
  --schedule spread
```

## 🆘 Troubleshooting

**High costs?** Use cheaper models or reduce token limits
**Quality issues?** Add examples to prompts, try reasoning models
**Rate limits?** Implement exponential backoff
**Boring content?** Adjust tone, add personality in edit

---

*Ready to start creating content? Check the templates directory!*
