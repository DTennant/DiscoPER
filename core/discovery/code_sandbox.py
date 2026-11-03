# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Safe code execution sandbox for exploration agent."""

from typing import Any, Optional
import io
import sys
import traceback
import signal
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
from functools import partial


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def _execute_in_process(code: str, data_dict: dict, result_queue: multiprocessing.Queue):
    """Execute code in a separate process (for true isolation)."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    # Reconstruct DataFrame from dict
    df = pd.DataFrame(data_dict)
    
    # Create restricted globals
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "isinstance": isinstance,
            "type": type,
            "any": any,
            "all": all,
            "map": map,
            "filter": filter,
            "reversed": reversed,
            "slice": slice,
            "format": format,
            "repr": repr,
            "hash": hash,
            "id": id,
            "open": None,  # Explicitly disabled
            "exec": None,  # Explicitly disabled
            "eval": None,  # Explicitly disabled
            "__import__": None,  # Explicitly disabled
        },
        "np": np,
        "pd": pd,
        "stats": stats,
        "df": df,
    }
    
    # Try to import optional packages
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        restricted_globals["plt"] = plt
    except ImportError:
        pass
    
    try:
        from sklearn import preprocessing, cluster, decomposition
        restricted_globals["preprocessing"] = preprocessing
        restricted_globals["cluster"] = cluster
        restricted_globals["decomposition"] = decomposition
    except ImportError:
        pass
    
    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, restricted_globals)
        
        result_queue.put({
            "success": True,
            "output": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": None
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "output": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        })


class CodeSandbox:
    """
    Safe execution environment for exploration code.
    
    Features:
    - Restricted builtins (no file I/O, no imports, no exec/eval)
    - Timeout protection
    - Output capture
    - Process isolation (optional)
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_length: int = 50000,
        use_process_isolation: bool = False
    ):
        """
        Initialize sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of captured output
            use_process_isolation: Whether to use multiprocessing for isolation
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.use_process_isolation = use_process_isolation
    
    def execute(self, code: str, data) -> dict[str, Any]:
        """
        Execute code safely with the given data.
        
        Args:
            code: Python code to execute
            data: pandas DataFrame to make available as 'df'
            
        Returns:
            Dict with success, output, error fields
        """
        # Validate code first (basic security check)
        validation = self._validate_code(code)
        if not validation["valid"]:
            return {
                "success": False,
                "output": "",
                "error": f"Code validation failed: {validation['reason']}"
            }
        
        if self.use_process_isolation:
            return self._execute_with_process_isolation(code, data)
        else:
            return self._execute_in_thread(code, data)
    
    def _validate_code(self, code: str) -> dict[str, Any]:
        """Basic security validation of code."""
        # Check for dangerous patterns
        dangerous_patterns = [
            "import os",
            "import sys",
            "import subprocess",
            "import shutil",
            "__import__",
            "exec(",
            "eval(",
            "open(",
            "file(",
            "input(",
            "raw_input(",
            "compile(",
            "globals(",
            "locals(",
            "vars(",
            "dir(",
            "getattr(",
            "setattr(",
            "delattr(",
            "hasattr(",
            "__dict__",
            "__class__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "os.system",
            "os.popen",
            "subprocess.run",
            "subprocess.Popen",
            "requests.",
            "urllib.",
            "socket.",
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                # Some patterns are OK in certain contexts
                if pattern == "import os" and "environ" not in code_lower:
                    continue
                return {"valid": False, "reason": f"Dangerous pattern detected: {pattern}"}
        
        return {"valid": True, "reason": None}
    
    def _execute_in_thread(self, code: str, data) -> dict[str, Any]:
        """Execute code in the current process with timeout."""
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Create restricted globals
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "isinstance": isinstance,
                "type": type,
                "any": any,
                "all": all,
                "map": map,
                "filter": filter,
                "reversed": reversed,
            },
            "np": np,
            "pd": pd,
            "stats": stats,
            "df": data.copy() if hasattr(data, 'copy') else data,
        }
        
        # Try to add optional packages
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            restricted_globals["plt"] = plt
        except ImportError:
            pass
        
        try:
            from sklearn import preprocessing, cluster, decomposition, linear_model
            restricted_globals["preprocessing"] = preprocessing
            restricted_globals["cluster"] = cluster
            restricted_globals["decomposition"] = decomposition
            restricted_globals["linear_model"] = linear_model
        except ImportError:
            pass
        
        # Set up timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {self.timeout}s")
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Set timeout (Unix only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            except (ValueError, OSError):
                pass  # Windows or threading context
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)
            
            # Cancel timeout
            try:
                signal.alarm(0)
            except (ValueError, OSError):
                pass
            
            output = stdout_capture.getvalue()
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
            
            return {
                "success": True,
                "output": output if output else "(no output)",
                "error": None
            }
            
        except TimeoutError as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
        finally:
            # Always cancel the alarm
            try:
                signal.alarm(0)
            except (ValueError, OSError):
                pass
    
    def _execute_with_process_isolation(self, code: str, data) -> dict[str, Any]:
        """Execute code in a separate process for true isolation."""
        # Convert DataFrame to dict for passing between processes
        data_dict = data.to_dict() if hasattr(data, 'to_dict') else {}
        
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(code, data_dict, result_queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            return {
                "success": False,
                "output": "",
                "error": f"Code execution timed out after {self.timeout}s"
            }
        
        try:
            result = result_queue.get_nowait()
            return result
        except:
            return {
                "success": False,
                "output": "",
                "error": "Failed to get result from subprocess"
            }
