import unittest

from src.actions import ACTION_TO_ID, ID_TO_ACTION, num_actions


class TestActions(unittest.TestCase):
    def test_bidirectional_mapping(self) -> None:
        self.assertEqual(len(ACTION_TO_ID), num_actions())
        self.assertEqual(len(ID_TO_ACTION), num_actions())
        for name, idx in ACTION_TO_ID.items():
            self.assertEqual(ID_TO_ACTION[idx], name)


if __name__ == "__main__":
    unittest.main()
