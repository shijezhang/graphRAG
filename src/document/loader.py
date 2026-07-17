from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")


class TextLoader:
    def load(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
        return [Document(content=text, metadata={"source": str(path)})]


class MarkdownLoader:
    def load(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
        return [Document(content=text, metadata={"source": str(path), "format": "markdown"})]


class PDFLoader:
    def load(self, path: Path) -> list[Document]:
        import fitz  # pymupdf

        docs = []
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():
                    docs.append(
                        Document(
                            content=text,
                            metadata={"source": str(path), "page": page_num},
                        )
                    )
        return docs


_LOADERS = {
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".pdf": PDFLoader,
}


def load_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    if path.is_file():
        return _load_single(path)
    if path.is_dir():
        docs = []
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() in _LOADERS:
                docs.extend(_load_single(file))
        return docs
    raise FileNotFoundError(f"Path not found: {path}")


def _load_single(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    loader_cls = _LOADERS.get(suffix)
    if not loader_cls:
        raise ValueError(f"Unsupported file format: {suffix}")
    return loader_cls().load(path)
