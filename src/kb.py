import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"

@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    text: str


def load_sections(directory=KB_DIR):
    sections = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        pieces = re.split(r"(?m)^## +", text)
        for piece in pieces[1:]:
            heading, _, body = piece.partition("\n")
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            sections.append(Section(f"{path.stem}#{slug}", heading, body.strip()))
    return sections
