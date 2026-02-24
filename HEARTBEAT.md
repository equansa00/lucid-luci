# BEAST Heartbeat Checklist
Checked every 30 minutes. Edit this file to add custom checks.

## Auto Checks (always run)
- System health: disk >= 15% free, RAM >= 10% available, load normal
- Ollama: API responding within 2s
- BEAST service: active, no restart loop, no ERROR in logs
- Tasks: unchecked items in tasks.md
- Repo: clean and in sync with origin
- Telegram: last send within 12h

## Custom Tasks
Add one-off tasks here as: - [ ] your task description
BEAST will report these as pending each heartbeat until you check them off.
