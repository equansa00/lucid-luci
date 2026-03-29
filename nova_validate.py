#!/usr/bin/env python3
"""
Nova Learn — Curriculum Validator
Checks IDs, prerequisites, vignette links, and progress consistency.
"""
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent
errors = []
warnings = []

def err(msg): errors.append(f"❌ {msg}")
def warn(msg): warnings.append(f"⚠️  {msg}")

# Load files
try:
    curriculum = json.load(open(WORKSPACE / "nova_curriculum.json"))
    print("✅ nova_curriculum.json loaded")
except Exception as e:
    print(f"❌ nova_curriculum.json: {e}"); sys.exit(1)

try:
    progress = json.load(open(WORKSPACE / "nova_progress.json"))
    print("✅ nova_progress.json loaded")
except Exception as e:
    print(f"❌ nova_progress.json: {e}"); sys.exit(1)

try:
    vignettes = json.load(open(WORKSPACE / "nova_vignettes.json"))
    print("✅ nova_vignettes.json loaded")
except Exception as e:
    print(f"❌ nova_vignettes.json: {e}"); sys.exit(1)

# Build maps
all_lesson_ids = set()
all_lessons = {}
all_module_ids = set()
phase_ids = set()
phase_to_modules = {}
module_to_phase = {}
lesson_to_module = {}

for phase in curriculum["phases"]:
    pid = phase["id"]
    phase_ids.add(pid)
    phase_to_modules[pid] = set()
    for module in phase["modules"]:
        mid = module["id"]
        all_module_ids.add(mid)
        phase_to_modules[pid].add(mid)
        module_to_phase[mid] = pid
        for lesson in module["lessons"]:
            lid = lesson["id"]
            if lid in all_lesson_ids:
                err(f"Duplicate lesson ID: {lid}")
            all_lesson_ids.add(lid)
            all_lessons[lid] = lesson
            lesson_to_module[lid] = mid

print(f"✅ {len(phase_ids)} phases, {len(all_module_ids)} modules, {len(all_lesson_ids)} lessons")

# Build vignette maps
seen_cat_ids = set()
all_vignette_category_ids = set()
for cat in vignettes["categories"]:
    cid = cat["id"]
    if cid in seen_cat_ids:
        err(f"Duplicate vignette category ID: {cid}")
    seen_cat_ids.add(cid)
    all_vignette_category_ids.add(cid)

all_vignette_ids = set()
for cat in vignettes["categories"]:
    if not cat.get("vignettes"):
        warn(f"Vignette category {cat['id']} has no vignettes")
    for vig in cat["vignettes"]:
        vid = vig["id"]
        if vid in all_vignette_ids:
            err(f"Duplicate vignette ID: {vid}")
        all_vignette_ids.add(vid)

print(f"✅ {len(all_vignette_category_ids)} vignette categories, {len(all_vignette_ids)} vignettes")

# Check prerequisites
for lid, lesson in all_lessons.items():
    for prereq in lesson.get("prerequisites", []):
        if prereq not in all_lesson_ids:
            err(f"Lesson {lid} has unknown prerequisite: {prereq}")

# Check linked_vignette_categories
missing_cats = set()
for lid, lesson in all_lessons.items():
    for cat_id in lesson.get("linked_vignette_categories", []):
        if cat_id not in all_vignette_category_ids:
            missing_cats.add(cat_id)
if missing_cats:
    warn(f"{len(missing_cats)} vignette categories referenced in curriculum but not in vignette bank:")
    for c in sorted(missing_cats):
        warn(f"  missing category: {c}")

# Check vignette category lesson_links
for cat in vignettes["categories"]:
    for ll in cat.get("lesson_links", []):
        if ll not in all_lesson_ids:
            err(f"Vignette category {cat['id']} links to unknown lesson: {ll}")

# Check correct_index
for cat in vignettes["categories"]:
    for vig in cat["vignettes"]:
        n = len(vig.get("options", []))
        ci = vig.get("correct_index", -1)
        if ci < 0 or ci >= n:
            err(f"Vignette {vig['id']} invalid correct_index {ci} (options: {n})")

# Progress consistency
cl = progress.get("current_lesson")
cm = progress.get("current_module")
cp = progress.get("current_phase")

if cl and cl not in all_lesson_ids:
    err(f"progress.current_lesson '{cl}' not in curriculum")
if cm and cm not in all_module_ids:
    err(f"progress.current_module '{cm}' not in curriculum")
if cp and cp not in phase_ids:
    err(f"progress.current_phase '{cp}' not in curriculum")
if cl and cm and cl in lesson_to_module and lesson_to_module[cl] != cm:
    err(f"current_lesson '{cl}' does not belong to current_module '{cm}'")
if cm and cp and cm in module_to_phase and module_to_phase[cm] != cp:
    err(f"current_module '{cm}' does not belong to current_phase '{cp}'")

rn = progress.get("recommended_next")
if rn and rn not in all_lesson_ids:
    err(f"progress.recommended_next '{rn}' not in curriculum")

for lid in progress.get("started_lessons", []):
    if lid not in all_lesson_ids:
        err(f"started_lessons contains unknown: {lid}")
for lid in progress.get("completed_lessons", []):
    if lid not in all_lesson_ids:
        err(f"completed_lessons contains unknown: {lid}")
for lid in progress.get("lesson_scores", {}).keys():
    if lid not in all_lesson_ids:
        err(f"lesson_scores contains unknown: {lid}")
for lid in progress.get("lesson_status", {}).keys():
    if lid not in all_lesson_ids:
        err(f"lesson_status contains unknown: {lid}")
for vid in progress.get("vignette_scores", {}).keys():
    if vid not in all_vignette_ids:
        err(f"vignette_scores contains unknown: {vid}")

required_patterns = [
    "premature_intervention", "assessment_skipping", "ethics_law_confusion",
    "overthinking", "distractor_selection", "first_step_vs_best_plan",
    "safety_threshold_error", "overreacts_to_risk", "skips_structured_assessment", "empathy_trap"
]
for key in required_patterns:
    if key not in progress.get("miss_patterns", {}):
        warn(f"miss_patterns missing key: {key}")

# Summary
print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors: print(f"  {e}")
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings: print(f"  {w}")
if not errors and not warnings:
    print("✅ All checks passed — Nova Learn foundation is valid")
elif not errors:
    print(f"✅ No errors — {len(warnings)} warnings")
else:
    print(f"❌ {len(errors)} errors, {len(warnings)} warnings")
