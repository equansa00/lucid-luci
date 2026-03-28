#!/usr/bin/env python3
"""
LUCI Teacher — Daily coding curriculum engine.
Chip's personalized bootcamp, rebuilt for 2026.
"""

import json
import os
from datetime import datetime
from pathlib import Path

WORKSPACE  = Path(os.path.expanduser("~/beast/workspace"))
CURRICULUM = WORKSPACE / "luci_curriculum.json"
SESSIONS   = WORKSPACE / "runs" / "learning_sessions"


def load_curriculum() -> dict:
    return json.loads(CURRICULUM.read_text())


def save_curriculum(data: dict) -> None:
    CURRICULUM.write_text(json.dumps(data, indent=2))


def get_current_lesson(data: dict) -> dict | None:
    phase_num   = data["current_phase"]
    module_num  = data["current_module"]
    lesson_num  = data["current_lesson"]

    for phase in data["phases"]:
        if phase["phase"] != phase_num:
            continue
        for module in phase["modules"]:
            if module["module"] != module_num:
                continue
            lessons = module["lessons"]
            if lesson_num <= len(lessons):
                return {
                    "phase":       phase_num,
                    "phase_title": phase["title"],
                    "module":      module_num,
                    "module_title":module["title"],
                    "lesson":      lesson_num,
                    "lesson_title":lessons[lesson_num - 1],
                    "total_lessons": len(lessons),
                    "phase_goal":  phase["goal"],
                    "module_lessons": lessons,
                }
    return None


def advance_lesson(data: dict) -> dict:
    """Move to next lesson, module, or phase."""
    phase_num  = data["current_phase"]
    module_num = data["current_module"]
    lesson_num = data["current_lesson"]

    # Record completion
    key = f"p{phase_num}_m{module_num}_l{lesson_num}"
    if key not in data["completed_lessons"]:
        data["completed_lessons"].append(key)

    for phase in data["phases"]:
        if phase["phase"] != phase_num:
            continue
        for module in phase["modules"]:
            if module["module"] != module_num:
                continue
            if lesson_num < len(module["lessons"]):
                data["current_lesson"] = lesson_num + 1
                return data
            # End of module — next module
            all_modules = [m["module"] for m in phase["modules"]]
            next_module_idx = all_modules.index(module_num) + 1
            if next_module_idx < len(all_modules):
                data["current_module"] = all_modules[next_module_idx]
                data["current_lesson"] = 1
                return data
        # End of phase — next phase
        all_phases = [p["phase"] for p in data["phases"]]
        next_phase_idx = all_phases.index(phase_num) + 1
        if next_phase_idx < len(all_phases):
            data["current_phase"]  = all_phases[next_phase_idx]
            data["current_module"] = data["phases"][next_phase_idx]["modules"][0]["module"]
            data["current_lesson"] = 1
    return data


def get_progress_summary(data: dict) -> str:
    total_lessons = sum(
        len(m["lessons"])
        for p in data["phases"]
        for m in p["modules"]
    )
    completed = len(data["completed_lessons"])
    pct = round(completed / total_lessons * 100, 1) if total_lessons else 0
    lesson = get_current_lesson(data)
    if not lesson:
        return "Curriculum complete!"
    return (
        f"Phase {lesson['phase']}/10: {lesson['phase_title']}\n"
        f"Module {lesson['module']}: {lesson['module_title']}\n"
        f"Lesson {lesson['lesson']}/{lesson['total_lessons']}: {lesson['lesson_title']}\n"
        f"Overall progress: {completed}/{total_lessons} lessons ({pct}%)"
    )


def build_teaching_prompt(lesson: dict) -> str:
    """Build the system prompt for a teaching session."""
    # Build allowed topics from phase title and module lessons
    phase_keywords = lesson['phase_title'].lower()
    allowed = []
    if 'python' in phase_keywords:
        allowed = ['Python', 'Flask', 'OOP', 'data structures']
    elif 'web' in phase_keywords or 'html' in phase_keywords:
        allowed = ['HTML', 'CSS', 'JavaScript', 'DOM', 'jQuery']
    elif 'typescript' in phase_keywords:
        allowed = ['TypeScript', 'types', 'interfaces', 'JavaScript']
    elif 'database' in phase_keywords or 'sql' in phase_keywords:
        allowed = ['SQL', 'databases', 'queries', 'PostgreSQL', 'SQLAlchemy']
    elif 'api' in phase_keywords or 'backend' in phase_keywords:
        allowed = ['APIs', 'REST', 'Express', 'Node', 'FastAPI', 'HTTP']
    elif 'react' in phase_keywords:
        allowed = ['React', 'components', 'props', 'state', 'hooks', 'JSX']
    elif 'ai' in phase_keywords:
        allowed = ['AI', 'LLMs', 'APIs', 'Python', 'prompts', 'embeddings']
    elif 'devops' in phase_keywords:
        allowed = ['deployment', 'Linux', 'systemd', 'Docker', 'git', 'servers']
    elif 'data structure' in phase_keywords or 'algorithm' in phase_keywords:
        allowed = ['arrays', 'linked lists', 'trees', 'graphs', 'sorting', 'Big-O']
    elif 'foundation' in phase_keywords:
        allowed = ['HTTP', 'DNS', 'TCP', 'browsers', 'servers', 'REST', 'curl']

    allowed_str = ', '.join(allowed) if allowed else 'topics relevant to this phase'

    return f"""You are LUCI acting as Chip's personal coding tutor.

TODAY'S LESSON:
Phase {lesson['phase']}: {lesson['phase_title']}
Goal: {lesson['phase_goal']}

Module {lesson['module']}: {lesson['module_title']}
Lesson {lesson['lesson']}/{lesson['total_lessons']}: {lesson['lesson_title']}

All lessons in this module (for context):
{chr(10).join(f"  {i+1}. {l}" for i, l in enumerate(lesson['module_lessons']))}

STRICT TOPIC GUARDRAILS:
- Only teach concepts relevant to: {allowed_str}
- Stay strictly within Phase {lesson['phase']}: {lesson['phase_title']}
- If the student's repo contains patterns from other phases, acknowledge them briefly but do NOT teach them now
- Never introduce technologies outside this phase's scope
- If asked about out-of-scope topics, say: "We'll cover that in a later phase — for now let's focus on {lesson['lesson_title']}"

YOUR TEACHING APPROACH:
1. Start with a clear explanation of the concept (5-10 min)
2. Use concrete examples from the student's actual repo code
3. Give Chip a hands-on exercise to try
4. Ask a question to check understanding
5. Connect this lesson to what Chip is already building

TEACHING STYLE:
- Speak directly: "Here's what this means..." not "In this lesson we will..."
- Use analogies to things Chip already knows
- Keep it practical — every concept should connect to something real
- Challenge Chip — don't over-explain, let him figure things out
- When he gets something right, move on fast. When he's stuck, come at it from a different angle.

Begin the lesson now. Start with the concept explanation."""


def log_session(lesson: dict, notes: str = "") -> None:
    SESSIONS.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log = {
        "date":         today,
        "phase":        lesson["phase"],
        "module":       lesson["module"],
        "lesson":       lesson["lesson"],
        "lesson_title": lesson["lesson_title"],
        "notes":        notes,
    }
    log_path = SESSIONS / f"{today}_lesson.json"
    log_path.write_text(json.dumps(log, indent=2))


def get_teach_context() -> str:
    """Return current lesson info for injection into LUCI's system prompt."""
    if not CURRICULUM.exists():
        return ""
    try:
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        if not lesson:
            return ""
        return (
            f"\n\nACTIVE LEARNING SESSION:\n"
            f"Phase {lesson['phase']}: {lesson['phase_title']}\n"
            f"Module {lesson['module']}: {lesson['module_title']}\n"
            f"Current lesson: {lesson['lesson_title']}\n"
            f"When Chip asks to learn or study, teach him this lesson "
            f"using real examples from his codebase."
        )
    except Exception:
        return ""


if __name__ == "__main__":
    import sys
    data   = load_curriculum()
    lesson = get_current_lesson(data)
    if not lesson:
        print("Curriculum complete!")
        sys.exit(0)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(get_progress_summary(data))

    elif cmd == "next":
        data = advance_lesson(data)
        save_curriculum(data)
        lesson = get_current_lesson(data)
        print(f"Advanced to: {lesson['lesson_title']}")

    elif cmd == "prompt":
        print(build_teaching_prompt(lesson))

    elif cmd == "reset":
        data["current_phase"]  = 1
        data["current_module"] = 1
        data["current_lesson"] = 1
        data["completed_lessons"] = []
        save_curriculum(data)
        print("Reset to beginning")
