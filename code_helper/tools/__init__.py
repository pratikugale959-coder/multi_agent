"""Custom tool packages for the Multi-Agent Code Helper.

Each module holds the tools for one worker:

* ``search_tools``       -> ``rag_search``                (Code Searcher)
* ``explain_tools``      -> ``read_project_file``        (Code Explainer)
* ``architecture_tools`` -> ``list_project_files``, ``inspect_imports``
* ``onboarding_tools``   -> ``get_project_overview``     (Onboarding Guide)
"""

from code_helper.tools.architecture_tools import inspect_imports, list_project_files
from code_helper.tools.explain_tools import read_project_file
from code_helper.tools.onboarding_tools import get_project_overview
from code_helper.tools.search_tools import rag_search

__all__ = [
    "rag_search",
    "read_project_file",
    "list_project_files",
    "inspect_imports",
    "get_project_overview",
]
