"""Literature builder for ML conference papers (ICLR, NeurIPS, ICML, ...).

Pipeline: fetch papercopilot JSON -> ingest into SQLite -> keyword filter ->
Haiku score -> on-demand deep digest -> markdown / mkdocs export.
"""

__version__ = "0.1.0"
