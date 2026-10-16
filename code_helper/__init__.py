"""Multi-Agent Code Helper.

Google ADK discovers ``root_agent`` from this package. Importing it here as well
makes ``from code_helper import root_agent`` work for tests and scripts.
"""

from code_helper.agent import root_agent

__all__ = ["root_agent"]
