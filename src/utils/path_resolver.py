import os

def resolve_path(path, project_root):
    if os.path.exists(path):
        return path
    candidate = os.path.join(project_root, path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"Could not find '{path}'.\n  Tried: {os.path.abspath(path)}\n  Tried: {candidate}\n"
        f"  cwd: {os.getcwd()}"
    )