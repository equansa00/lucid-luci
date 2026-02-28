#!/usr/bin/env python3
"""
LUCI Journal — Simple CLI journaling tool
Usage: python3 journal.py
"""
import json
import os
from datetime import datetime
from pathlib import Path

JOURNAL_FILE = Path.home() / ".luci_journal.json"

def load_journal() -> list:
    if JOURNAL_FILE.exists():
        return json.loads(JOURNAL_FILE.read_text())
    return []

def save_journal(entries: list):
    JOURNAL_FILE.write_text(json.dumps(entries, indent=2))

def add_entry(text: str = ""):
    if not text:
        print("Entry (Ctrl+D when done):")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        text = "\n".join(lines)
    if not text.strip():
        print("Empty entry — not saved.")
        return
    entry = {
        "id": len(load_journal()) + 1,
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%A, %B %d %Y %I:%M %p"),
        "text": text.strip()
    }
    entries = load_journal()
    entries.append(entry)
    save_journal(entries)
    print(f"✅ Entry #{entry['id']} saved — {entry['date']}")

def view_entries(n: int = 10):
    entries = load_journal()
    if not entries:
        print("No journal entries yet.")
        return
    print(f"\n📔 Journal — {len(entries)} entries total\n")
    for e in entries[-n:]:
        print(f"[{e['date']}]")
        print(f"{e['text']}")
        print("─" * 40)

def search_entries(query: str):
    entries = load_journal()
    results = [e for e in entries if query.lower() in e["text"].lower()]
    if not results:
        print(f"No entries matching '{query}'")
        return
    print(f"\n🔍 {len(results)} entries matching '{query}':\n")
    for e in results:
        print(f"[{e['date']}] {e['text'][:100]}...")
        print("─" * 40)

def delete_entry(entry_id: int):
    entries = load_journal()
    entries = [e for e in entries if e.get("id") != entry_id]
    save_journal(entries)
    print(f"✅ Entry #{entry_id} deleted.")

def main():
    import sys
    args = sys.argv[1:]

    if not args:
        # Interactive mode
        while True:
            print("\n📔 LUCI Journal")
            print("  1. Add entry")
            print("  2. View recent entries")
            print("  3. Search entries")
            print("  4. Delete entry")
            print("  5. Exit")
            choice = input("\n❯ ").strip()
            if choice == "1":
                add_entry()
            elif choice == "2":
                view_entries()
            elif choice == "3":
                q = input("Search: ")
                search_entries(q)
            elif choice == "4":
                eid = input("Entry ID to delete: ")
                delete_entry(int(eid))
            elif choice in ("5", "q", "exit"):
                break
    elif args[0] == "add":
        add_entry(" ".join(args[1:]))
    elif args[0] == "view":
        view_entries(int(args[1]) if len(args) > 1 else 10)
    elif args[0] == "search":
        search_entries(" ".join(args[1:]))
    elif args[0] == "delete":
        delete_entry(int(args[1]))
    else:
        print("Usage: journal.py [add|view|search|delete] [args]")

if __name__ == "__main__":
    main()
