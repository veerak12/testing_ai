"""
High-level LangChain agent that plans actions (via an LLM)
and executes them safely through a PlaywrightController.
Supports Groq (for GPT-OSS / Llama models), LangChain (OpenAI),
and local Ollama models. Loads configuration from .env.
"""

import json
import re
import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv  # âœ… Added dotenv support

# Load environment variables from .env (once)
load_dotenv()

# --- Optional Ollama (local) ---
# try:
#     from ollama import Ollama  # type: ignore
# except Exception:
#     Ollama = None
Ollama = None  # Disabled for now

# --- Optional LangChain Fallback ---
# try:
#     try:
#         from langchain.llms import OpenAI  # type: ignore
#     except Exception:
#         from langchain import OpenAI  # type: ignore
# except Exception:
#     OpenAI = None
OpenAI = None  # Disabled for now

# --- GROQ Client Setup ---
try:
    from groq import Groq
except ImportError:
    Groq = None


class LlmAgent:
    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: Optional name of the LLM model.
                        e.g. "gpt-oss-20b", "llama-3.3-70b-versatile", "deepseek-coder"
        """
        self.model_name = model_name or os.getenv("GROQ_MODEL", "gpt-oss-20b")
        self.backend = None
        self.client = None

        # --- Prefer GROQ if available ---
        if Groq is not None and os.getenv("GROQ_API_KEY"):
            try:
                self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                self.backend = "groq"
                logging.info(f"[LLM] Using Groq backend with model: {self.model_name}")
            except Exception as e:
                logging.warning(f"[LLM] Groq init failed: {e}")

        # --- Ollama local model (commented for now) ---
        # elif Ollama is not None and model_name:
        #     self.llm = Ollama(model=model_name)
        #     self.backend = "ollama"
        #     logging.info(f"[LLM] Using Ollama model: {model_name}")

        # --- OpenAI fallback via LangChain (commented for now) ---
        # elif OpenAI is not None:
        #     self.llm = OpenAI(temperature=0)
        #     self.backend = "openai"
        #     logging.info("[LLM] Using OpenAI fallback")

        if not self.backend:
            raise RuntimeError("No valid LLM backend found (Groq/Ollama/OpenAI).")

    # ------------------------- LLM Call Wrapper -------------------------
    def _call_llm(self, prompt: str) -> str:
        """Unified call wrapper for Groq / Ollama / OpenAI."""
        try:
            if self.backend == "groq":
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a QA automation assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model_name,
                )
                return chat_completion.choices[0].message.content

            elif self.backend == "ollama":
                return self.llm(prompt)

            elif self.backend == "openai":
                res = self.llm(prompt)
                if isinstance(res, str):
                    return res
                if hasattr(res, "generations"):
                    gens = res.generations
                    if gens and gens[0] and hasattr(gens[0][0], "text"):
                        return gens[0][0].text
                return str(res)

        except Exception as e:
            logging.error(f"[LLM] call failed: {e}")
            return f"__LLM_CALL_ERROR__ {e}"

        return "__NO_VALID_LLM_BACKEND__"

    # ------------------------- JSON Parsing -------------------------
    def _extract_json(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract and parse JSON array from the model output."""
        match = re.search(r"(\[[\s\S]*\])", text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                try:
                    fixed = match.group(1).replace("'", '"')
                    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
                    return json.loads(fixed)
                except Exception:
                    return None
        return None

    # ------------------------- Planning -------------------------
    def _plan_actions(self, page_summary: List[Dict[str, Any]], test_step: str) -> List[Dict[str, Any]]:
        """Converts natural-language test step into structured actions."""
        system = (
     "You are a QA automation assistant.\n"
    "Given a web page element summary and a test instruction, output a JSON array of actions.\n"
    "Each action MUST be an object with keys: 'action', 'selector', and optionally 'value' or 'expected'.\n"
    "Valid actions: open, fill, click, assert, wait.\n"
    "For navigation, use: {\"action\": \"open\", \"selector\": \"<url>\"}.\n"
    "Selectors should prefer data-testid, id, name, aria-label, or visible text.\n"
    "Before performing any main actions, always check for and close visible cookie or privacy banners "
    "(for example buttons with text like 'Accept all', 'Allow', 'OK', 'Continue', 'Consent' or 'Got it'). "
    "Add a 'click' action for them at the start of the sequence if found.\n"
    "If elements may take time to load, add a short 'wait' action before interacting.\n"
    "Example output:\n"
    "[\n"
    "  {\"action\": \"open\", \"selector\": \"https://example.com\"},\n"
    "  {\"action\": \"click\", \"selector\": \"button:has-text('Accept all')\"},\n"
    "  {\"action\": \"fill\", \"selector\": \"#email\", \"value\": \"user@test.com\"},\n"
    "  {\"action\": \"click\", \"selector\": \"button:has-text('Login')\"},\n"
    "  {\"action\": \"assert\", \"selector\": \"#welcome-message\", \"expected\": \"Welcome\"}\n"
    "]\n"
    "Output ONLY valid JSON â€” no explanations outside the array."
)

        prompt = (
            f"{system}\n\nPage Summary:\n{json.dumps(page_summary, ensure_ascii=False, indent=2)}\n\n"
            f"Instruction:\n{test_step}\n\nRespond with JSON only."
        )

        raw = self._call_llm(prompt)
        parsed = self._extract_json(raw)
        if parsed:
            return parsed
        return [{"action": "explain", "message": "Failed to parse model output", "raw": raw}]

    # ------------------------- Execution -------------------------
    def execute_test(self, page_controller: Any, test_step: str) -> Dict[str, Any]:
        """Plan and execute actions for a given test step."""
        result: Dict[str, Any] = {
            "instruction": test_step,
            "actions": [],
            "status": "ok",
            "errors": [],
        }

        # --- Step 1: Describe the page for LLM context ---
        try:
            page_summary = (
                page_controller.describe_page()
                if hasattr(page_controller, "describe_page")
                else []
            )
        except Exception as e:
            page_summary = []
            result["errors"].append({"message": f"describe_page() failed: {e}"})

        # --- Step 2: Ask LLM to plan structured actions ---
        actions = self._plan_actions(page_summary, test_step)
        result["actions"] = actions

        # --- Step 3: Normalize the LLM output ---
        def normalize_action(act: dict) -> dict:
            """Normalize LLM output so both structured and shorthand JSON are supported."""
            if "action" not in act:
                for key in ["click", "fill", "assert", "wait", "open", "explain"]:
                    if key in act:
                        new_act = {"action": key}
                        if isinstance(act[key], str):
                            new_act["selector"] = act[key]
                        elif isinstance(act[key], dict):
                            new_act.update(act[key])
                        return new_act
            return act

        # --- Step 4: Execute actions sequentially ---
        for idx, raw_act in enumerate(actions):
            act = normalize_action(raw_act)
            a = act.get("action")
            sel = act.get("selector")

            try:
                if a == "open":
                    # âœ… NEW: Handle navigation
                    page_controller.open_page(sel)

                elif a == "fill":
                    page_controller.fill(sel, act.get("value", ""))

                elif a == "click":
                    page_controller.click(sel)

                elif a == "assert":
                    page_controller.assert_text(sel, act.get("expected", ""))

                elif a == "wait":
                    page_controller.wait_for(sel)

                elif a == "explain":
                    # Explanation-only step, no browser action
                    result["errors"].append({
                        "index": idx,
                        "message": act.get("message", "explain step")
                    })

                else:
                    result["errors"].append({
                        "index": idx,
                        "message": f"Unknown action '{a}'"
                    })

            except Exception as e:
                result["status"] = "failed"
                result["errors"].append({
                    "index": idx,
                    "error": str(e),
                    "action": act,
                })

        # --- Step 5: Final status ---
        if result["errors"]:
            result["status"] = "failed"

        return result