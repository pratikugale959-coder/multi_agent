"""ADK module entry point used by ``adk web`` and ``adk run``.

``code_helper/__init__.py`` exposes ``root_agent`` for package imports, but ADK
2.12's web loader only recognises a directory containing ``agent.py`` or
``root_agent.yaml``. Re-exporting the coordinator here keeps both entry points
working without duplicating agent construction.
"""

from code_helper.agents.coordinator import root_agent

__all__ = ["root_agent"]
