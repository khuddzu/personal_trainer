import unittest
from personal_trainer.config_handler import ConfigHandler


class TestConfigHandler(unittest.TestCase):

    def setUp(self):
        self.config_handler = ConfigHandler("../templates/editor.ini")

    def test_convert(self):
        # Check all potential types (boolean, string, list, NoneType, float, int)
        # boolean
        self.assertEqual(self.config_handler._convert("Global", "personal"), True)
        # string
        self.assertEqual(self.config_handler._convert("Global", "functional"), "wb97x")
        # list
        self.assertEqual(
            self.config_handler._convert("Global", "elements"),
            ["H", "C", "N", "O", "S", "F", "Cl"],
        )
        # NoneType
        self.assertEqual(self.config_handler._convert("Global", "constants"), None)
        # float
        self.assertEqual(self.config_handler._convert("Trainer", "lr_factor"), 0.7)
        # int
        self.assertEqual(self.config_handler._convert("Trainer", "batch_size"), 2048)

    def test_get_all_values(self):
        all_values = self.config_handler.load_config()
        self.assertIn("Global", all_values)
        self.assertIn("Trainer", all_values)


if __name__ == "__main__":
    unittest.main()
