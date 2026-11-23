"""Execute documentation examples and validate local Markdown image links."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (ROOT / "README.md", ROOT / "docs" / "cookbook.md")


def python_blocks(path):
    """Return the Python fenced code blocks in *path*."""
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)


def check_examples(path):
    """Execute a document's examples in order in a clean namespace."""
    namespace = {"__name__": "__documentation__"}
    for number, block in enumerate(python_blocks(path), start=1):
        exec(compile(block, f"{path.name} block {number}", "exec"), namespace)
    plt.close("all")


def check_images(path):
    """Require every relative Markdown image to resolve from its document."""
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", path.read_text())
    missing = [
        link for link in links
        if "://" not in link and not (path.parent / link).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing images in {path}: {missing}")


def main():
    for document in DOCUMENTS:
        check_examples(document)
        check_images(document)
        print(f"Checked {document.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
