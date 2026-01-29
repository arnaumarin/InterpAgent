import os
from typing import Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from .utils import fig_to_compressed_base64

class Py_tool():
    def __init__(self, python_repl):
        self.python_repl = python_repl
    
    def create(self):
        python_repl = self.python_repl
        def wrap(python_repl):
            @tool(response_format="content_and_artifact")
            def python_repl_tool(query: str) -> Tuple[str, List[str]]:
                """A Python shell. Use this to execute python commands. Input should be a valid python command. 
                If you want to see the output of a value, you should print it out with `print(...)`. """
                encoded_imgs = []  # List to store file paths of generated plots
                result_parts = []  # List to store different parts of the output
                
                try:
                    output = python_repl.run(query)

                    if output and output.strip():
                        result_parts.append(output.strip())
                    
                    figures = [plt.figure(i) for i in plt.get_fignums()]
                    if figures:
                        for fig in figures:
                            encoded_imgs.append(fig_to_compressed_base64(fig))                        
                        result_parts.append(f"Generated {len(encoded_imgs)} plot(s).")
                    
                    if not result_parts:  # If no output and no figures
                        result_parts.append("Executed code successfully with no output.")

                except Exception as e:
                    result_parts.append(f"Error executing code: {e}")
                
                # Join all parts of the result with newlines
                result_summary = "\n".join(result_parts)
                
                # Return both the summary and plot paths (if any)
                return result_summary, encoded_imgs
            return python_repl_tool
        return wrap(python_repl)