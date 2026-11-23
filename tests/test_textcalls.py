import unittest

from achoose.brain.textcalls import extract_text_tool_calls

NAMES = {"get_current_time", "list_folder", "save_note"}


class TextCallTests(unittest.TestCase):
    def test_phi4_style_list(self):
        text = '[{"type":"function","function":{"name":"list_folder","arguments":{"path":"~/Downloads"}}}]'
        calls, rest = extract_text_tool_calls(text, NAMES)
        self.assertEqual([(c.name, c.arguments) for c in calls], [("list_folder", {"path": "~/Downloads"})])
        self.assertEqual(rest, "")

    def test_plain_object_with_parameters(self):
        calls, _ = extract_text_tool_calls('{"name": "get_current_time", "parameters": {}}', NAMES)
        self.assertEqual(calls[0].name, "get_current_time")

    def test_json_inside_text_and_fences(self):
        text = 'Sure!\n```json\n{"name":"save_note","arguments":{"title":"a","text":"b"}}\n```'
        calls, rest = extract_text_tool_calls(text, NAMES)
        self.assertEqual(calls[0].arguments, {"title": "a", "text": "b"})
        self.assertEqual(rest, "Sure!")

    def test_tagged(self):
        text = '<tool_call>{"name":"get_current_time","arguments":{}}</tool_call>'
        calls, _ = extract_text_tool_calls(text, NAMES)
        self.assertEqual(calls[0].name, "get_current_time")

    def test_string_arguments(self):
        text = '{"name":"list_folder","arguments":"{\\"path\\": \\"~\\"}"}'
        calls, _ = extract_text_tool_calls(text, NAMES)
        self.assertEqual(calls[0].arguments, {"path": "~"})

    def test_unknown_tool_is_ignored(self):
        text = '{"name":"delete_everything","arguments":{}}'
        self.assertEqual(extract_text_tool_calls(text, NAMES), ([], text))

    def test_normal_text_and_json_data_untouched(self):
        for text in ["Paris is the capital of France.", 'Here is data: {"a": 1}', ""]:
            self.assertEqual(extract_text_tool_calls(text, NAMES)[0], [])


if __name__ == "__main__":
    unittest.main()
