from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result  = search_tool.invoke('IPL News')

print(result)


# --------------------------------------------------------------

from langchain_community.tools import ShellTool


shell_tool = ShellTool()

result = shell_tool.invoke('whoami')
print(result)

# result = shell_tool.invoke('dir')
# print(result)


# --------------------------------------------------------------
