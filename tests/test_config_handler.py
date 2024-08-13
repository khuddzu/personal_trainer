import unittest
from personal_trainer.config_handler import ConfigHandler


class TestConfigHandler(unittest.TestCase):

    def setUp(self):
        self.config_handler = ConfigHandler("test_inputs/test_editor.ini")

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

    def test_load_config(self):
        all_values = self.config_handler.load_config()
        self.assertIn("Global", all_values)
        self.assertIn("Trainer", all_values)
    
 
class TestMinimalConfig(unittest.TestCase):

    def setUp(self):
        # Initialize with minimal .ini file
        self.config_handler = ConfigHandler("test_inputs/minimal.ini")

    def test_load_minimal_config(self):
        with self.assertRaises(AssertionError) as context:
            config = self.config_handler.load_config()
            
            # Verify that the error message is about the missing 'constants' or 'elements'
            self.assertIn("Constants pathway not specified in configuration file.", str(context.exception))
        
            # Check if 'Global' section is correctly loaded
            self.assertIn('Global', context)

            # Ensure essential keys in the 'Global' section are handled
            self.assertEqual(context['Global']['personal'], True)

            # Check if missing sections or keys are handled correctly
            self.assertNotIn('Trainer', context)  # Should not exist
            self.assertNotIn('constants', context['Global'])  # Should not exist


if __name__ == "__main__":
    unittest.main()
