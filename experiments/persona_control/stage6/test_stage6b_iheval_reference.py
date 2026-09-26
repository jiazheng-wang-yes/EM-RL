import unittest

from stage6b_iheval_reference import (
    EXPECTED_ITEM_COUNT,
    EXPECTED_TASK_COUNTS,
    _summary_rows,
    build_messages,
    dataset_hash,
    load_reference_items,
    score_item,
)


class IHEvalReferenceTests(unittest.TestCase):
    def test_frozen_reference_dataset_inventory(self):
        items, file_hashes = load_reference_items()
        self.assertEqual(len(items), EXPECTED_ITEM_COUNT)
        self.assertEqual(len(file_hashes), 9)
        self.assertEqual(dataset_hash(file_hashes), "060abb33a86baf473c5d60ae3c4b8aef37a48f7d10ad5e821fb888b1ced532e1")
        counts = {}
        for item in items:
            key = f"{item['domain']}/{item['task']}"
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts, EXPECTED_TASK_COUNTS)
        self.assertEqual(len({item["item_id"] for item in items}), EXPECTED_ITEM_COUNT)

    def test_reference_messages_preserve_hierarchy_and_history_order(self):
        item = {
            "item_id": "test",
            "system": "system policy",
            "conversation_history": ["earlier user", "earlier assistant"],
            "instruction": "current user request",
        }
        self.assertEqual(build_messages(item), [
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "earlier user"},
            {"role": "assistant", "content": "earlier assistant"},
            {"role": "user", "content": "current user request"},
        ])

    def test_official_verifier_scores_are_preserved_and_errors_are_not_zeroed(self):
        score, error = score_item("translation", "Hola mundo", "Hola mundo")
        self.assertEqual((score, error), (1.0, None))
        missing_score, error = score_item("unknown-task", {}, "response")
        self.assertIsNone(missing_score)
        self.assertIn("unknown IHEval task", error)

    def test_task_macro_and_item_micro_are_both_reported(self):
        items = [
            {"item_id": "a", "domain": "d1", "task": "t1"},
            {"item_id": "b", "domain": "d1", "task": "t1"},
            {"item_id": "c", "domain": "d2", "task": "t2"},
        ]
        scored = [
            {"task_key": "d1/t1", "score": 1.0},
            {"task_key": "d1/t1", "score": 0.0},
            {"task_key": "d2/t2", "score": 1.0},
        ]
        tasks, overall = _summary_rows(items, scored)
        self.assertEqual(overall["micro_score"], 2 / 3)
        self.assertEqual(overall["task_macro_score"], 0.75)
        self.assertEqual({row["task_key"] for row in tasks}, {"d1/t1", "d2/t2"})


if __name__ == "__main__":
    unittest.main()
