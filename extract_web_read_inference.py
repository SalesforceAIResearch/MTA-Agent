#!/usr/bin/env python3
"""
Extract web_read entries from trajectory JSON files (inference/mm-verl format).
Finds steps with gpt_action containing "web_read" (or "content_extraction"/"url_reader"),
gets url from action_parameters and question from trajectory, and saves:
  - url||question -> observation_summary
  - url||original -> observation
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_question_from_trajectory(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get question dict (text, image, image_url) from trajectory data.
    For web_read we need question text; image_url is optional.
    """
    question = data.get("question")
    if question and isinstance(question, dict) and question.get("text"):
        return question
    if question and isinstance(question, dict) and question.get("image_url"):
        return question
    traj = data.get("trajectory") or data.get("trajectory_data")
    if isinstance(traj, dict):
        question = traj.get("question")
        if question and isinstance(question, dict) and (question.get("text") or question.get("image_url")):
            return question
    steps = data.get("steps") or []
    if steps and isinstance(steps[0], dict) and steps[0].get("question"):
        q = steps[0].get("question")
        if isinstance(q, dict) and (q.get("text") or q.get("image_url")):
            return q
    return None


def extract_web_read_from_trajectory_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a single trajectory JSON and extract web_read step entries.
    Returns list of dicts: {url, question_text, observation, observation_summary}.
    """
    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading {file_path}: {e}")
        return results

    question = get_question_from_trajectory(data)
    question_text = (question.get("text") or "").strip() if question else ""

    steps = data.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        gpt_action = step.get("gpt_action") or step.get("action")
        if not isinstance(gpt_action, dict):
            continue
        action_type = (gpt_action.get("action_type") or "").strip().lower()
        if "web_read" not in action_type and "content_extraction" not in action_type and "url_reader" not in action_type:
            continue

        params = gpt_action.get("action_parameters") or {}
        if isinstance(params, dict):
            raw_url = params.get("url") or params.get("page_url") or params.get("page_urls") or ""
            if isinstance(raw_url, list):
                url = (raw_url[0] or "").strip() if raw_url else ""
            else:
                url = (raw_url or "").strip()
        else:
            url = ""
        if not url:
            continue

        observation = step.get("observation")
        observation_summary = step.get("observation_summary")
        if observation is None and observation_summary is None:
            continue

        obs_str = observation if isinstance(observation, str) else (json.dumps(observation) if observation is not None else "")
        summary_str = observation_summary if isinstance(observation_summary, str) else (json.dumps(observation_summary) if observation_summary is not None else "")

        results.append({
            "url": url,
            "question_text": question_text,
            "observation": obs_str,
            "observation_summary": summary_str,
        })
    return results


def read_directories_from_files(file_paths: List[str]) -> List[str]:
    """
    Read base directory paths from text files (one path per line).
    Each line is a base path; we will find its subfolders and check subfolder/trajectories.
    """
    dirs = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    dirs.append(line.rstrip("/"))
        except OSError as e:
            print(f"Warning: could not read {path}: {e}")
    return dirs


def find_trajectory_json_files(directory: str) -> List[str]:
    """
    For the given base path, list all subfolders, then for each subfolder
    check subfolder/trajectories and collect *.json there.
    """
    found = []
    base = Path(directory)
    if base.name == "trajectories" and base.parent.exists():
        base = base.parent
    if not base.exists():
        return found
    for subdir in base.iterdir():
        if subdir.is_dir():
            traj_dir = subdir / "trajectories"
            if traj_dir.is_dir():
                for p in traj_dir.glob("*.json"):
                    found.append(str(p))
    return sorted(set(found))


def main():
    parser = argparse.ArgumentParser(
        description="Extract web_read entries from trajectory JSON files (inference format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_web_read_inference.py -d /path/to/mm-verl/examples/train/mm_deep_research_rl
  python extract_web_read_inference.py -f subfolders_abs.txt -o web_read_inference_extracted.json -v
        """,
    )
    parser.add_argument(
        "-d", "--directories",
        nargs="+",
        default=[],
        help="Directories to search for trajectory JSONs (each can be a .../trajectories dir)",
    )
    parser.add_argument(
        "-f", "--from-file",
        nargs="+",
        dest="from_files",
        metavar="FILE",
        help="Text files listing one base path per line; for each, search base/*/trajectories for JSONs",
    )
    parser.add_argument(
        "-o", "--output",
        default="web_read_inference_extracted.json",
        help="Output JSON file (default: web_read_inference_extracted.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose per-file output",
    )
    args = parser.parse_args()

    directories = list(args.directories)
    if args.from_files:
        directories.extend(read_directories_from_files(args.from_files))
    if not directories:
        directories = ["."]

    all_files = []
    for d in directories:
        if not os.path.exists(d):
            print(f"Warning: directory '{d}' does not exist, skipping.")
            continue
        files = find_trajectory_json_files(d)
        all_files.extend(files)
        if args.verbose and files:
            print(f"Found {len(files)} trajectory file(s) in {d}")

    all_files = sorted(set(all_files))
    if not all_files:
        print(f"No trajectory JSON files found under the {len(directories)} directory/ies specified")
        return

    total_files = len(all_files)
    print(f"Processing {total_files} trajectory file(s)...")

    by_url_question: Dict[str, str] = {}   # url||question -> observation_summary
    by_url_original: Dict[str, str] = {}   # url||original -> observation
    total_steps = 0

    for i, path in enumerate(all_files, 1):
        print(f"  [{i}/{total_files}] {os.path.basename(path)}", flush=True)
        entries = extract_web_read_from_trajectory_file(path)
        for e in entries:
            total_steps += 1
            url = e["url"]
            question_text = (e["question_text"] or "").strip().lower()
            key_question = f"{url}||{question_text}" if question_text else url
            key_original = f"{url}||original"
            if e["observation_summary"] and key_question not in by_url_question:
                by_url_question[key_question] = e["observation_summary"]
            if e["observation"] and key_original not in by_url_original:
                by_url_original[key_original] = e["observation"]
        if args.verbose and entries:
            print(f"  {path}: {len(entries)} web_read step(s)")

    out = {
        "by_url_question": by_url_question,
        "by_url_original": by_url_original,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Total web_read steps processed: {total_steps}")
    print(f"Unique url||question (observation_summary): {len(by_url_question)}")
    print(f"Unique url||original (observation): {len(by_url_original)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
