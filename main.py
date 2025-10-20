import argparse
import json
import os
import logging
from pathlib import Path
from datetime import datetime

from agents.langchain_agent import LlmAgent
from tools.playwright_tools import PlaywrightController

# Configure logging (same format as Playwright tools)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_testcases(dir_path: str):
    """Load all .txt test case files from a directory."""
    tests = []
    for p in sorted(Path(dir_path).glob("*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().strip()
        tests.append({"name": p.stem, "content": content})
    return tests


def main():
    parser = argparse.ArgumentParser(description="Run LLM-powered Playwright test automation.")
    parser.add_argument("--testcases_dir", default="testcases", help="Directory containing .txt test files")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--storage_state", default=None, help="Path to Playwright storage state JSON file")
    parser.add_argument("--model_name", default=None, help="Optional local LLM model name (for Ollama or custom)")
    parser.add_argument("--report_file", default="run_report.json", help="Where to store JSON test results")
    args = parser.parse_args()

    logging.info("=== Starting LLM + Playwright automation run ===")
    logging.info(f"Test cases directory: {args.testcases_dir}")
    logging.info(f"Headless mode: {args.headless}")
    logging.info(f"Storage state: {args.storage_state}")
    logging.info(f"Using model: {args.model_name or 'default (OpenAI fallback)'}")

    # Start Playwright browser once and reuse
    pw = PlaywrightController(headless=args.headless)
    llm_agent = LlmAgent(model_name=args.model_name)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        pw.launch()

        # Optionally restore storage state
        if args.storage_state and os.path.exists(args.storage_state):
            pw.new_context(storage_state=args.storage_state)
        else:
            pw.new_context()

        tests = load_testcases(args.testcases_dir)
        if not tests:
            logging.warning(f"No test cases found in directory: {args.testcases_dir}")
            return

        for test in tests:
            logging.info(f"=== Running test: {test['name']} ===")
            try:
                result = llm_agent.execute_test(pw, test["content"])
                result["test_name"] = test["name"]
                all_results.append(result)
                logging.info(f"Test '{test['name']}' completed with status: {result.get('status')}")
            except Exception as e:
                logging.exception(f"Error running test '{test['name']}': {e}")
                all_results.append({
                    "test_name": test["name"],
                    "status": "failed",
                    "error": str(e)
                })

        # Save final state (e.g., logged-in cookies)
        try:
            pw.save_storage_state(f"storage_state_{timestamp}.json")
        except Exception as e:
            logging.warning(f"Could not save storage state: {e}")

    finally:
        pw.close()

        # Write results to report file
        try:
            report_path = Path(args.report_file)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            logging.info(f"Test results saved to {report_path.resolve()}")
        except Exception as e:
            logging.warning(f"Failed to write test report: {e}")

    logging.info("=== Test run complete ===")


if __name__ == "__main__":
    main()