#Paths
# Folders
# common helper functions

from pathlib import Path

def get_project_paths():
    project_dir = Path(__file__).resolve().parent.parent

    paths = {
        "project": project_dir,
        "images": project_dir / "images",
        "outputs": project_dir / "outputs",
        "models": project_dir / "models"
    }

    paths["outputs"].mkdir(exist_ok=True)

    return paths