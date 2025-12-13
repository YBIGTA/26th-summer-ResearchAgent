import os
import tempfile
import unittest

from .iterative_sft import build_argparser, run_iterative_sft


class EngineIntegrationTest(unittest.TestCase):
    def test_single_fen_jsonl_generation(self):
        engine_path = os.environ.get("STOCKFISH_PATH")
        if not engine_path:
            self.skipTest("STOCKFISH_PATH not set; skipping engine integration test")

        with tempfile.TemporaryDirectory() as tmpdir:
            parser = build_argparser()
            args = parser.parse_args(
                [
                    "--engine_path",
                    engine_path,
                    "--fen",
                    "8/8/8/8/8/8/8/8 w - - 0 1",
                    "--iterations",
                    "1",
                    "--num_candidates",
                    "1",
                    "--out_dir",
                    tmpdir,
                    "--mode",
                    "sft_iter",
                ]
            )
            run_iterative_sft(args)
            jsonl_path = os.path.join(tmpdir, "sft_iter_data.jsonl")
            self.assertTrue(os.path.exists(jsonl_path))
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = [line for line in f.readlines() if line.strip()]
            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
