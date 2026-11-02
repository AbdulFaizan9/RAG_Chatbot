import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from aiprojects import p60_rag_chatbot as p60


class P60(unittest.TestCase):
    def bot(self):
        b = p60.RAGChatbot()
        b.add_document("py.txt", "Python was created by Guido van Rossum. It was first released in 1991.")
        b.add_document("np.txt", "NumPy provides fast arrays for scientific computing.")
        return b

    def test_grounded_answers_and_abstention(self):
        b = self.bot()
        a = b.ask("Who created Python?")
        self.assertTrue(a.grounded)
        self.assertIn("Guido", a.text)
        self.assertEqual(a.citations[0], "py.txt#0")
        self.assertIn("arrays", b.ask("What does NumPy provide?").text)
        na = b.ask("What is the capital of Mars?")
        self.assertFalse(na.grounded)
        self.assertEqual(na.text, p60.RAGChatbot.NO_ANSWER)
        self.assertFalse(b.ask("  ").grounded)

    def test_generator_hook_and_persistence(self):
        b = p60.RAGChatbot(generator=lambda prompt: "LLM:" + str("Context" in prompt))
        b.add_document("d.txt", "Rust is a systems language focused on memory safety.")
        self.assertEqual(b.ask("What is Rust?").text, "LLM:True")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "idx.json")
            b.index.save(path)
            loaded = p60.BM25Index.load(path)
            self.assertEqual(len(loaded.chunks), len(b.index.chunks))
            self.assertTrue(loaded.search("rust memory"))

    def test_chunking_many_chunks(self):
        b = p60.RAGChatbot(chunk_size=10, overlap=2)
        self.assertGreater(b.add_document("x", " ".join(f"w{i}" for i in range(50))), 3)


if __name__ == "__main__":
    unittest.main()
