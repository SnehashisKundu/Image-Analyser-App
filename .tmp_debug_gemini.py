import sys, json, traceback
from pathlib import Path
import importlib.util

print('sys.executable:', sys.executable)
print('sys.version:', sys.version)

# Resolve the project module file explicitly to avoid issues with spaces
# in folder names and to make the debug loader more robust for static analysis.
gemini_path = Path(r'D:/Plant Tribe/Image Analysis/app/trained model/gemini.py')
if not gemini_path.exists():
    print(f'gemini.py not found at expected path: {gemini_path}')
else:
    try:
        spec = importlib.util.spec_from_file_location('gemini', str(gemini_path))
        gemini = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            raise ImportError('Could not get loader for gemini spec')
        loader.exec_module(gemini)

        # Call the debug helper if present
        if hasattr(gemini, 'debug_gemini'):
            dbg = gemini.debug_gemini()
            print('DEBUG_GEMINI RESULT:')
            try:
                print(json.dumps(dbg, default=str, indent=2))
            except Exception:
                print(repr(dbg))
        else:
            print('gemini.py loaded but has no debug_gemini() function')
    except Exception:
        traceback.print_exc()
        print('Failed to load or run gemini.debug_gemini()')
