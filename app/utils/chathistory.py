import json
import os


class ChatHistoryManager:
    FILE = "chat_history.json"

    def __init__(self, max_history=5):
        dir_name = os.path.dirname(self.FILE)
        if dir_name:  # Only make directories if a path is specified
            os.makedirs(dir_name, exist_ok=True)
        self.max_history = max_history
        self.chat_history = self._load()

    def _load(self):
        if os.path.exists(self.FILE):
            with open(self.FILE, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save(self):
        with open(self.FILE, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=2)

    def add_message(self, role, content):
        """Add a message and keep last N entries."""
        self.chat_history.append({"role": role, "content": str(content)})
        self.chat_history = self.chat_history[-self.max_history :]
        self._save()

    def get_history(self):
        return self.chat_history

    def build_context(self):
        """Return conversation string for LLM input."""
        return "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in self.chat_history]
        )

    def rewrite_query(self, user_input: str):
        """Rewrites the user query with context from previous messages."""
        self.add_message("user", user_input)
        context = self.build_context()
        rewritten_query = (
            f"Given the previous discussion:\n{context}\n\nUser Query: {user_input}"
        )
        return rewritten_query
