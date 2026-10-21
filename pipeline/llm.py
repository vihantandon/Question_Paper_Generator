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
        -> the model is told why and asked again, up to max_attempts times.
           Only the LAST rejected reply goes back with the reason, and only
           when it is short: a checker's long report sent back twice made the
           third request too large for Groq's free per-minute limit (413).
  * the reply was cut off (the model used up its output tokens -- a
    reasoning model can spend them all thinking), or the service could not
    make valid JSON of it (Groq's json_validate_failed)  -> asked again, and
    the reason is shown. json_validate_failed used to be taken for "this
    model has no JSON mode", which turned JSON mode off and left gpt-oss
    giving empty replies.
  * reasoning_effort ("low" by default, LLM_REASONING_EFFORT in .env) is sent
    to models that think before they answer (gpt-oss, Gemini): fewer tokens
    spent thinking, so the free daily token budget lasts longer. A service
    or model that does not know the setting -> sent again without it.
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
ECHO_LIMIT = 2000         # characters: a longer rejected reply is not sent back with the reason
# the service could not make the reply into the JSON it was asked for (Groq: json_validate_failed)
JSON_FAILED_RE = re.compile(r"json_validate_failed|failed to generate json", re.IGNORECASE)
# ... versus: this model does not do JSON mode at all
NO_JSON_MODE_RE = re.compile(r"(response_format|json[ _]?(mode|object)).{0,120}(not supported|unsupported|"
                             r"does not support|not available)|(not supported|unsupported|does not support)"
                             r".{0,120}(response_format|json)", re.IGNORECASE | re.DOTALL)
NO_REASONING_RE = re.compile(r"reasoning", re.IGNORECASE)

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


class BadReply(ValueError):
    """The service answered, but with nothing usable (cut off, empty, or no
    valid JSON). Counts as one rejected attempt."""


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

    def __init__(self, chat, model, *, max_attempts=3, json_mode=True, reasoning_effort=None,
                 sleep=time.sleep, log=print):
        self.chat, self.model = chat, model
        self.max_attempts = max_attempts
        self.json_mode = json_mode        # ask the service for JSON-only output (turned off if unsupported)
        self.reasoning_effort = reasoning_effort or None   # "low" | "medium" | "high"; dropped if unknown
        self.sleep, self.log = sleep, log
        self.requests = 0                 # requests sent so far: what a run cost
        self.tokens = 0                   # tokens used (prompt + reply), as the service counts them

    def complete_json(self, messages, *, temperature=0.7, validate=None):
        """Send `messages` (a list of {"role", "content"}) and return the JSON
        object of the reply -- or validate(obj), when validate is given. A
        reply that is not JSON, or that validate rejects with ValueError, is
        answered with the reason and the model tries again. Raises LLMError
        when no attempt is usable. `messages` itself is not changed."""
        original = list(messages)
        messages = original
        problem = None
        for attempt in range(1, self.max_attempts + 1):
            text = None
            try:
                text = self._send(messages, temperature)
                obj = parse_json_reply(text)
                return validate(obj) if validate else obj
            except ValueError as exc:
                problem = str(exc)
                self.log(f"  [llm] reply {attempt}/{self.max_attempts} not accepted: {problem}")
                again = "Reply again with only the corrected JSON object."
                if text and len(text) > ECHO_LIMIT:
                    again = "Reply again with the whole JSON object, corrected."
                    echo = []                     # too long to send back: the reason is enough
                else:
                    echo = [{"role": "assistant", "content": text}] if text else []
                messages = original + echo + [
                    {"role": "user", "content": f"Your reply was not accepted: {problem}\n{again}"}]
        raise LLMError(f"{self.model} gave no usable reply in {self.max_attempts} attempt(s). "
                       f"Last problem: {problem}")

    def _send(self, messages, temperature):
        """One request; waits out a busy service. Returns the reply text."""
        busy = 0
        while True:
            kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.reasoning_effort:       # extra_body: works with every version of the openai package
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
            self.requests += 1
            try:
                reply = self.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - sorted by status below
                status = getattr(exc, "status_code", None)
                message = str(exc)
                if status == 400 and JSON_FAILED_RE.search(message):
                    raise BadReply("the service could not make valid JSON of the reply (it was probably "
                                   "cut off or not a single JSON object)") from exc
                if status == 400 and self.json_mode and NO_JSON_MODE_RE.search(message):
                    self.log("  [llm] this model has no JSON mode; asking for JSON in the prompt only")
                    self.json_mode = False
                    continue
                if status == 400 and self.reasoning_effort and NO_REASONING_RE.search(message):
                    self.log(f"  [llm] {self.model} does not take reasoning_effort; sending without it")
                    self.reasoning_effort = None
                    continue
                if status == 413:
                    raise LLMError(
                        f"the request is too large for {self.model}'s limit (413). A model with a larger "
                        f"limit (LLM_CHECK_MODEL / LLM_MODEL in .env), or fewer sources, would fit. "
                        f"Service said: {message}") from exc
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
            usage = getattr(reply, "usage", None)
            self.tokens += getattr(usage, "total_tokens", 0) or 0
            choices = getattr(reply, "choices", None) or []
            if not choices:
                return None
            text = choices[0].message.content
            if getattr(choices[0], "finish_reason", None) == "length":
                raise BadReply("the reply was cut off: the model used up its output tokens"
                               + ("" if (text or "").strip() else " while still thinking (set "
                                  "LLM_REASONING_EFFORT=low in .env, or use another model)")
                               + ". Keep the JSON short: no explanations outside it")
            return text


def make_llm(base_url=None, model=None, api_key=None, timeout=None, reasoning_effort=None):
    """The real client, set up from settings.py (.env) unless told otherwise."""
    from settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_REASONING_EFFORT, LLM_TIMEOUT
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
    return LLMClient(client.chat, model, reasoning_effort=reasoning_effort or LLM_REASONING_EFFORT)


def make_checker():
    """The model that checks and classifies, when .env names one
    (LLM_CHECK_MODEL, and -- for another service, with its own daily budget --
    LLM_CHECK_BASE_URL and LLM_CHECK_API_KEY). None: the drafting model does it."""
    from settings import LLM_CHECK_API_KEY, LLM_CHECK_BASE_URL, LLM_CHECK_MODEL
    if not (LLM_CHECK_MODEL or LLM_CHECK_BASE_URL):
        return None
    if LLM_CHECK_BASE_URL and not (LLM_CHECK_MODEL and LLM_CHECK_API_KEY):
        raise SystemExit("LLM_CHECK_BASE_URL is set in .env: set LLM_CHECK_MODEL (a model of that "
                         "service) and LLM_CHECK_API_KEY (its key) too")
    return make_llm(base_url=LLM_CHECK_BASE_URL or None, model=LLM_CHECK_MODEL or None,
                    api_key=LLM_CHECK_API_KEY or None)


def usage_line(*clients):
    """"14 requests, 52,310 tokens to openai/gpt-oss-20b" for each client used."""
    seen, parts = set(), []
    for c in clients:
        if c is not None and id(c) not in seen:
            seen.add(id(c))
            parts.append(f"{c.requests} request(s), {c.tokens:,} tokens to {c.model}")
    return "; ".join(parts)
