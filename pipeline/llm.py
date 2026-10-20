"""
pipeline/llm.py -- the one place that talks to the language model.

    from pipeline.llm import make_llm
    llm = make_llm()                       # service + model + key from settings.py (.env)
    obj = llm.complete_json(messages, validate=check)

complete_json() asks for ONE JSON object and returns it (or what `validate`
returns for it). It deals with what goes wrong with a language model:

  * the JSON comes wrapped in ```json fences, with a sentence before it, or
    after a reasoning model's <think> ... </think> notes  -> cleaned off
  * the reply is not JSON, or `validate` rejects it (raises ValueError)
        -> the model is told why and asked again, up to max_attempts times
  * the service is busy (429 / 5xx)  -> wait, then send again
  * no connection or a time-out  -> two more tries, then LLMFatalError
  * the key, model id or account is wrong (401 / 402 / 403 / 404), or the
    daily limit is used up  -> LLMFatalError at once: asking again cannot
    help, and every later question would fail the same way

Any service that speaks the OpenAI chat API works (OpenRouter, Groq, a local
Ollama ...): only LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env change.
The `openai` package is imported only inside make_llm(), so the tests --
which hand LLMClient a stand-in -- need neither it nor a network.
"""
import json
import re
import time

MAX_WAIT = 60.0           # s: never wait longer than this for a busy service
MAX_BUSY_RETRIES = 5      # waits for a busy service before giving up on one request
MAX_NETWORK_RETRIES = 2   # no connection / time-out: try twice more, then stop the run
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
FATAL_STATUS = {
    401: "the API key was rejected",
    402: "the account has no credit for this model",
    403: "this key may not use this model",
    404: "the model id or the base URL is wrong",
}
DAILY_LIMIT_RE = re.compile(r"per[- ]day|daily|quota", re.IGNORECASE)

# A reasoning model may put its notes before the answer. An unclosed <think>
# (the reply was cut off while still thinking) leaves nothing usable.
THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """No usable answer for THIS request. Other questions may still work."""


class LLMFatalError(LLMError):
    """Every request will fail until something is fixed (settings, credit,
    daily limit). The run should stop."""


def parse_json_reply(text):
    """The JSON object inside a model's reply. Raises ValueError (with a
    reason the model can act on) when there is none."""
    if not text or not text.strip():
        raise ValueError("the reply was empty")
    cleaned = THINK_RE.sub("", text).strip()
    if not cleaned:
        raise ValueError("the reply held only reasoning notes, no answer")
    candidates = [m.group(1) for m in FENCE_RE.finditer(cleaned)] + [cleaned]
    for candidate in candidates:
        candidate = candidate.strip()
        start, end = candidate.find("{"), candidate.rfind("}")
        for piece in (candidate, candidate[start:end + 1] if 0 <= start < end else None):
            if not piece:
                continue
            try:
                obj = json.loads(piece)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("the reply is not a JSON object: " + cleaned[:120].replace("\n", " "))


def _retry_after(exc):
    """Seconds the service asked us to wait (Retry-After header), if it said."""
    try:
        return float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _is_network_error(exc):
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


class LLMClient:
    """`chat` is anything with .completions.create(**kwargs) that returns an
    object shaped like the OpenAI SDK's reply -- the real client's .chat, or a
    stand-in in the tests."""

    def __init__(self, chat, model, *, max_attempts=3, json_mode=True,
                 sleep=time.sleep, log=print):
        self.chat, self.model = chat, model
        self.max_attempts = max_attempts
        self.json_mode = json_mode        # ask the service for JSON-only output (turned off if unsupported)
        self.sleep, self.log = sleep, log
        self.requests = 0                 # requests sent so far: what a run cost

    def complete_json(self, messages, *, temperature=0.7, validate=None):
        """Send `messages` (a list of {"role", "content"}) and return the JSON
        object of the reply -- or validate(obj), when validate is given. A
        reply that is not JSON, or that validate rejects with ValueError, is
        answered with the reason and the model tries again. Raises LLMError
        when no attempt is usable. `messages` itself is not changed."""
        messages = list(messages)
        problem = None
        for attempt in range(1, self.max_attempts + 1):
            text = self._send(messages, temperature)
            try:
                obj = parse_json_reply(text)
                return validate(obj) if validate else obj
            except ValueError as exc:
                problem = str(exc)
                self.log(f"  [llm] reply {attempt}/{self.max_attempts} not accepted: {problem}")
                messages += [
                    {"role": "assistant", "content": text or ""},
                    {"role": "user", "content": f"Your reply was not accepted: {problem}\n"
                                                "Reply again with only the corrected JSON object."},
                ]
        raise LLMError(f"{self.model} gave no usable reply in {self.max_attempts} attempt(s). "
                       f"Last problem: {problem}")

    def _send(self, messages, temperature):
        """One request; waits out a busy service. Returns the reply text."""
        busy = 0
        while True:
            kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            self.requests += 1
            try:
                reply = self.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - sorted by status below
                status = getattr(exc, "status_code", None)
                message = str(exc)
                if status == 400 and self.json_mode and re.search(r"response_format|json", message, re.I):
                    self.log("  [llm] this model has no JSON mode; asking for JSON in the prompt only")
                    self.json_mode = False
                    continue
                if status in FATAL_STATUS:
                    raise LLMFatalError(
                        f"{FATAL_STATUS[status]} ({status}). Check LLM_BASE_URL, LLM_MODEL and "
                        f"LLM_API_KEY in .env (model now: {self.model}). Service said: {message}") from exc
                if status == 429 and DAILY_LIMIT_RE.search(message):
                    raise LLMFatalError(f"the daily limit for {self.model} is used up. Try again "
                                        f"tomorrow, or set another LLM_MODEL in .env. ({message})") from exc
                network = status is None and _is_network_error(exc)
                if status not in RETRY_STATUS and not network:
                    raise LLMError(f"request failed ({status or type(exc).__name__}): {message}") from exc
                busy += 1
                if network and busy > MAX_NETWORK_RETRIES:
                    # every later question would wait just as long, so stop here
                    raise LLMFatalError(
                        f"cannot reach {self.model}'s service ({message}). Check the internet "
                        f"connection and LLM_BASE_URL; a slow free model may need a larger "
                        f"LLM_TIMEOUT in .env.") from exc
                wait = _retry_after(exc) or min(MAX_WAIT, 2.0 ** busy)
                if wait > MAX_WAIT:
                    raise LLMFatalError(f"the service asks to wait {wait:.0f}s (limit used up?). Try "
                                        f"again later, or set another LLM_MODEL in .env. ({message})") from exc
                if busy > MAX_BUSY_RETRIES:
                    raise LLMError(f"the service was still busy after {MAX_BUSY_RETRIES} waits: {message}") from exc
                self.log(f"  [llm] {status or type(exc).__name__}: waiting {wait:.0f}s, then sending again")
                self.sleep(wait)
                continue
            choices = getattr(reply, "choices", None) or []
            return choices[0].message.content if choices else None


def make_llm(base_url=None, model=None, api_key=None, timeout=None):
    """The real client, set up from settings.py (.env) unless told otherwise."""
    from settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT
    base_url = base_url or LLM_BASE_URL
    model = model or LLM_MODEL
    api_key = api_key or LLM_API_KEY
    if not api_key:
        raise SystemExit(
            f"No API key for the language model ({base_url}).\n"
            f"Put OPENROUTER_API_KEY=... (or LLM_API_KEY=...) in .env -- see .env.example --\n"
            f"or run without the model:  python -m pipeline.graph <topic> --no-llm")
    from openai import OpenAI
    headers = None
    if "openrouter.ai" in base_url:      # OpenRouter asks free-tier apps to name themselves
        headers = {"HTTP-Referer": "https://github.com/vihantandon/Question_Paper_Generator",
                   "X-Title": "Question Paper Generator"}
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout or LLM_TIMEOUT,
                    max_retries=0,       # LLMClient does the waiting, and says so
                    default_headers=headers)
    return LLMClient(client.chat, model)
