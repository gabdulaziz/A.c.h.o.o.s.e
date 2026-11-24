"""Memory tests. They need numpy (and the document packages for some checks):
    .venv/bin/python -m unittest -v
Tests that need a missing package are skipped, not failed."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path

from achoose.memory.extract import chunk_text, extract_text

HAS = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "docx", "pptx", "pypdf", "reportlab")}


class FakeEmbedder:
    """Deterministic 'fingerprints' from word hashes; no Ollama needed."""

    def __init__(self, name: str = "fake", dims: int = 64) -> None:
        self.name = name
        self.dims = dims
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
        return v

    def embed_documents(self, texts, title="none"):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_and_html(self):
        (self.dir / "a.md").write_text("# Plan\nBuy a telescope")
        (self.dir / "b.html").write_text("<html><style>x{}</style><p>Hello &amp; welcome</p></html>")
        self.assertIn("telescope", extract_text(self.dir / "a.md"))
        html = extract_text(self.dir / "b.html")
        self.assertIn("Hello & welcome", html)
        self.assertNotIn("x{}", html)

    def test_binary_with_text_name_is_empty(self):
        (self.dir / "x.txt").write_bytes(b"\x00\x01\x02binary")
        self.assertEqual(extract_text(self.dir / "x.txt"), "")

    @unittest.skipUnless(HAS["docx"], "python-docx not installed")
    def test_docx(self):
        import docx

        d = docx.Document()
        d.add_paragraph("Java midterm covers inheritance")
        d.save(self.dir / "m.docx")
        self.assertIn("inheritance", extract_text(self.dir / "m.docx"))

    @unittest.skipUnless(HAS["pptx"], "python-pptx not installed")
    def test_pptx(self):
        from pptx import Presentation

        p = Presentation()
        slide = p.slides.add_slide(p.slide_layouts[1])
        slide.shapes.title.text = "AstroPulse orbit"
        p.save(self.dir / "s.pptx")
        text = extract_text(self.dir / "s.pptx")
        self.assertIn("Slide 1", text)
        self.assertIn("AstroPulse orbit", text)

    @unittest.skipUnless(HAS["pypdf"] and HAS["reportlab"], "pypdf/reportlab not installed")
    def test_pdf(self):
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(self.dir / "p.pdf"))
        c.drawString(72, 720, "Satellite power budget")
        c.save()
        self.assertIn("power budget", extract_text(self.dir / "p.pdf"))

    def test_chunking(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("short"), ["short"])
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
        pieces = chunk_text(text, size=800, overlap=120)
        self.assertGreater(len(pieces), 5)
        self.assertTrue(all(len(p) <= 800 for p in pieces))
        self.assertIn("Paragraph 19", pieces[-1])


@unittest.skipUnless(HAS["numpy"], "numpy not installed")
class MemoryTests(unittest.TestCase):
    def setUp(self):
        from achoose.memory.indexer import Memory
        from achoose.memory.store import MemoryStore

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = root / "Documents"
        (self.docs / "school").mkdir(parents=True)
        (self.docs / "node_modules").mkdir()
        (self.docs / "school" / "java.md").write_text("The Java midterm covers inheritance and interfaces.")
        (self.docs / "space.txt").write_text("AstroPulse satellite power budget is 40 watts.")
        (self.docs / "photo.png").write_bytes(b"\x89PNG")
        (self.docs / "node_modules" / "junk.js").write_text("inheritance junk")
        self.embedder = FakeEmbedder()
        self.db = root / "memory.db"
        self.make = lambda emb=None: Memory(MemoryStore(self.db), emb or self.embedder, [self.docs])
        self.memory = self.make()

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_search_and_skip_rules(self):
        report = self.memory.update()
        self.assertEqual((report.added, report.failed), (2, []))  # png and node_modules skipped
        hits = self.memory.search("what does the java midterm cover")
        self.assertTrue(hits[0].path.endswith("java.md"))
        hits = self.memory.search("satellite power")
        self.assertTrue(hits[0].path.endswith("space.txt"))

    def test_program_installs_and_skip_patterns_are_ignored(self):
        from achoose.memory.indexer import Memory
        from achoose.memory.store import MemoryStore

        jdk = self.docs / "OpenJDK25U-jdk"
        (jdk / "bin").mkdir(parents=True)
        (jdk / "lib").mkdir()
        (jdk / "README.txt").write_text("java inheritance readme")
        old = self.docs / "old-stuff"
        old.mkdir()
        (old / "note.txt").write_text("old note")
        found = Memory(MemoryStore(self.db), self.embedder, [self.docs], skip=["old-*"]).find_files()
        self.assertEqual(sorted(Path(p).name for p in found), ["java.md", "space.txt"])

    def test_only_changes_are_reread(self):
        self.memory.update()
        calls = self.embedder.calls
        report = self.memory.update()
        self.assertEqual((report.added, report.updated, report.unchanged), (0, 0, 2))
        self.assertEqual(self.embedder.calls, calls)

        f = self.docs / "space.txt"
        f.write_text("AstroPulse battery holds 90 watt hours, much more now.")
        os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))
        (self.docs / "school" / "java.md").unlink()
        report = self.memory.update()
        self.assertEqual((report.updated, report.removed), (1, 1))
        self.assertTrue(self.memory.search("battery")[0].path.endswith("space.txt"))
        self.assertEqual(self.memory.store.counts()[0], 1)

    def test_new_fingerprint_model_rebuilds(self):
        self.memory.update()
        other = self.make(FakeEmbedder(name="other", dims=32))
        report = other.update()
        self.assertEqual(report.added, 2)
        self.assertTrue(other.search("java midterm")[0].path.endswith("java.md"))

    def test_search_tool_output(self):
        from achoose.tools.memory_tool import memory_tools

        tool = memory_tools(self.memory)[0]
        self.assertIn("/index", tool.run(query="java"))  # nothing indexed yet
        self.memory.update()
        out = tool.run(query="java midterm")
        self.assertIn("java.md", out)
        self.assertIn("inheritance", out)


if __name__ == "__main__":
    unittest.main()
