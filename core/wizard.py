"""Goal-driven auto-wizard: search rule configurations for a persona goal and
score each candidate empirically on the LIVE model.

A goal ("this model is named Ametista, not Qwen") expands into candidate rule
stacks drawn from two strategies with an inherent trade-off (measured live on
Qwen3.5-27B):

  SAFE   — preserve_lm_head=True, wide upper band. The output vocabulary is
           untouched (the old name stays fully usable, "Spell Qwen" works),
           but an unknown name may snap to its closest known word
           ("Lunaria" → "Luna").
  EXACT  — head transformed, late surface band, first-piece-anchored
           replacement + rank-2 companion zaps for the source's extra pieces.
           Spells arbitrary names exactly ("I am Lunaria"), but positions
           READING the old word can degenerate — the vocabulary battery
           catches it and scores it down.

Each candidate is scored with three probe batteries (identity: does the new
name appear; vocabulary: is the old word still usable; control: is the model
still coherent) plus a degeneration detector. The winner's rules are left
active for inspection/export; every candidate's scorecard is reported.
"""

import json
import re
import threading

from core import editing
from core.tokens import token_candidates

IDENTITY_PROMPTS = (
    "Who are you? Answer in one sentence.",
    "What is your name?",
    "Introduce yourself briefly.",
    "Tell me about yourself in one sentence.",
)
ADDRESS_PROMPTS = (
    "Greet me.",
    "Say hello to me and offer your help.",
    "Ask me how my day was.",
    "What do you call the person you are talking to?",
)
CONTROL_PROMPTS = (
    ("What is 7 times 8?", ("56",)),
    ("What is the capital of France?", ("Paris",)),
)
# neutral openers for the style battery: the boosted/suppressed words should
# shift here without a prompt that itself invites them
STYLE_PROMPTS = (
    "Write the opening line of a story.",
    "Describe a room in two sentences.",
    "Continue: The evening settled over the city and",
    "Tell me about your day.",
)
GEN_TOKENS = 45
STYLE_GEN_TOKENS = 60


def degenerate(text):
    """Loop detector: long reply dominated by few distinct words."""
    words = text.split()
    if len(words) < 10:
        return False
    return len(set(w.casefold() for w in words)) / len(words) < 0.4


def resolve_word(tokenizer, word):
    """``(ids, display)`` for a word — single token preferred, else the
    leading-space split (composite)."""
    singles, splits = token_candidates(tokenizer, word)
    pick = next((c for c in singles if c["str"].startswith(" ")), None) or (
        singles[0] if singles else None)
    if pick:
        return [pick["id"]], pick["str"]
    sp = next((s for s in splits if s["str"].startswith(" ")), None) or (
        splits[0] if splits else None)
    if sp:
        return sp["ids"], sp["str"]
    raise ValueError(f"no token for {word!r}")


def band(n_layers, lo_frac, hi_frac):
    lo = max(0, int(n_layers * lo_frac))
    hi = min(n_layers - 2, int(n_layers * hi_frac))  # last layer excluded
    return list(range(lo, hi + 1))


def style_candidates(n_layers, rebase_supported=True):
    """Candidate (boost, suppress) scale strengths for a style goal. Scale
    rules act on CONCEPT layers (mid-depth), like the nikusui preset's ×1.1
    boosts and ×0 zaps — never the surface band, which would just spam the raw
    word."""
    layers = list(range(n_layers)) if not rebase_supported else band(n_layers, 0.33, 0.62)
    return [
        {"name": "gentle", "layers": layers, "boost": 1.1, "suppress": 0.3},
        {"name": "medium", "layers": layers, "boost": 1.2, "suppress": 0.0},
        {"name": "strong", "layers": layers, "boost": 1.35, "suppress": 0.0},
    ]


def candidates_for(n_layers, rebase_supported=True):
    """The candidate grid.

    Read-projection architectures (Qwen/Llama/Mistral style): layer bands ×
    head-protection, encoding the live-measured behavior on Qwen3.5-27B.

    Write-norm architectures (Gemma style): the read projection is unavailable,
    edits go through the GLOBAL projection (W_U abliteration) where layer
    ranges and head-protection do not exist — the search space is the
    projection's own knobs (factor, keep, anchor decay, companions). This grid
    is a starting spread, not yet live-tuned; the probes decide empirically.
    """
    if not rebase_supported:
        every = list(range(n_layers))  # layers only mark the rule active: the
        # projection itself is global
        return [
            {"name": "abl-1.0", "preserve": False, "layers": every,
             "factor": 1.0, "keep": 0.0, "decay": 0.5, "companions": False},
            {"name": "abl-1.0-keep30", "preserve": False, "layers": every,
             "factor": 1.0, "keep": 0.3, "decay": 0.5, "companions": False},
            {"name": "abl-0.8-keep15", "preserve": False, "layers": every,
             "factor": 0.8, "keep": 0.15, "decay": 0.5, "companions": False},
            {"name": "abl-1.2-comp", "preserve": False, "layers": every,
             "factor": 1.2, "keep": 0.0, "decay": 0.35, "companions": True},
            {"name": "abl-1.5-comp", "preserve": False, "layers": every,
             "factor": 1.5, "keep": 0.0, "decay": 0.5, "companions": True},
        ]
    late = band(n_layers, 0.78, 0.98)
    upper = band(n_layers, 0.50, 0.98)
    return [
        {"name": "safe-wide-1.5", "preserve": True, "layers": upper,
         "factor": 1.5, "decay": 0.7, "companions": False},
        {"name": "safe-wide-2.2", "preserve": True, "layers": upper,
         "factor": 2.2, "decay": 0.35, "companions": False},
        {"name": "exact-late-1.0", "preserve": False, "layers": late,
         "factor": 1.0, "decay": 0.7, "companions": True},
        {"name": "exact-late-1.2", "preserve": False, "layers": late,
         "factor": 1.2, "decay": 0.7, "companions": True},
        {"name": "exact-late-1.5", "preserve": False, "layers": late,
         "factor": 1.5, "decay": 0.45, "companions": True},
    ]


def total_score(identity, vocab, control, bad):
    """Identity GATES the score: a candidate that renames nothing scores ~0 no
    matter how harmless it is (measured: without the gate, a no-op config wins).
    Vocabulary and coherence then discount the achieved rename."""
    return identity * (4.0 + 2.0 * vocab + control) - 1.5 * bad


PARSE_PROMPT = """You configure a model-editing tool that edits token \
directions inside a language model. It can ONLY: (a) rename the model's \
identity or how it addresses the user, and (b) shift its vocabulary/tone by \
boosting or suppressing individual words. It CANNOT add knowledge, backstory, \
memories, or complex multi-word speech mannerisms.

The model being edited is: {model_id}

Read the user's description of the persona they want and extract goals as JSON \
with exactly this shape:
{{"goals": [
  {{"type": "rename", "sources": ["<word(s) the model says today to replace — \
for an identity rename, the CURRENT model name(s)>"], "target": "<new word>", \
"battery": "identity"}},
  {{"type": "style", "boost": ["<single words to make more frequent>"], \
"suppress": ["<single words to make rarer>"]}}
], "notes": "<requests the tool cannot fulfil, one short sentence — or \"\">"}}

Emit a "rename" goal for each identity/address change (battery "identity" when \
the model should call ITSELF the target, "address" when it should call the USER \
the target; sources at most 3 short words, never a full model-id path). Emit AT \
MOST ONE "style" goal, collecting tone words: e.g. a gentle melancholic persona \
boosts ["gentle","softly","quiet"] and suppresses ["assist","certainly"]. Use \
single, common words. Omit goal types that do not apply. Output ONLY the JSON.

User description:
{description}"""


def _lenient_json(cleaned, start):
    """Parse from ``start``, repairing a reply truncated at max_tokens: walk
    the brackets, then retry from the last completed value outward, appending
    the closers still open there. Returns the first prefix that parses."""
    stack = []
    in_str = esc = False
    candidates = []
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("]" if c == "[" else "}")
        elif c in "]}":
            if stack:
                stack.pop()
            candidates.append((i, "".join(reversed(stack))))
            if not stack:
                break
    for pos, closers in reversed(candidates):
        try:
            return json.loads(cleaned[start:pos + 1] + closers)
        except json.JSONDecodeError:
            continue
    raise ValueError("unparseable JSON in the reply")


def extract_goals_json(text):
    """The goals object from a model reply: tolerates code fences, prose around
    the JSON, and truncation. Raises ValueError when nothing usable is found."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    data = _lenient_json(cleaned, start)

    def clean_list(items, cap):
        out = [str(s).strip() for s in (items or []) if str(s).strip()]
        return list(dict.fromkeys(out))[:cap]

    goals = []
    for g in data.get("goals", []):
        gtype = g.get("type")
        if gtype == "style":
            boost = clean_list(g.get("boost"), 8)
            suppress = clean_list(g.get("suppress"), 8)
            if boost or suppress:
                goals.append({"type": "style", "boost": boost, "suppress": suppress})
            continue
        # default to rename (covers a missing/odd type with sources+target)
        sources = clean_list(g.get("sources"), 4)
        target = str(g.get("target", "")).strip()
        if not sources or not target:
            continue
        battery = g.get("battery")
        goals.append({
            "type": "rename",
            "sources": sources,
            "target": target,
            "battery": battery if battery in ("identity", "address") else "identity",
        })
    if not goals:
        raise ValueError("no usable goal in the reply")
    return {"goals": goals, "notes": str(data.get("notes", "") or "")}


class WizardRun:
    def __init__(self, manager, lens_manager, interventions, thinking=False):
        self.manager = manager
        self.lens_manager = lens_manager
        self.interventions = interventions
        self.stop_event = threading.Event()
        self.state = None
        # probe under the conditions the model is actually used: thinking-tuned
        # models degrade with reasoning forced off, so match deployment. Thinking
        # replies are longer — the token budget grows to fit the <think> block.
        self.thinking = bool(thinking)

    # --- model helpers ------------------------------------------------------

    def generate(self, prompt, max_tokens=GEN_TOKENS, ablated=True,
                 repetition_penalty=1.0):
        done = {}

        def emit(frame):
            if frame.get("type") == "done":
                done.update(frame)
            elif frame.get("type") == "error":
                done["error"] = frame.get("message")

        # probes run WORST-CASE for sampling: greedy, no repetition penalty — a
        # candidate that only holds up behind a penalty must be seen looping
        # here, not after export. Thinking follows the wizard's setting (parsing
        # always stays non-thinking, ablated=False).
        think = self.thinking and ablated
        budget = max_tokens + (512 if think else 0)  # room for the <think> block
        self.manager.generate(
            [{"role": "user", "content": prompt}],
            {"temperature": 0, "max_tokens": budget,
             "repetition_penalty": repetition_penalty, "min_p": 0.0, "thinking": think},
            threading.Event(), emit, lens=None,
            ablator=self.interventions if ablated and self.interventions.active else None,
        )
        if done.get("error"):
            raise RuntimeError(done["error"])
        text = (done.get("text") or "").strip()
        # score the ANSWER, not the reasoning: drop everything up to </think>
        if think and "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        return text

    def parse_description(self, description):
        """Free-text persona description → structured goals, extracted by the
        LOADED model itself (no external service). Runs with the current rules
        DETACHED so an already-edited identity cannot distort the parsing."""
        model_id = (self.manager.meta or {}).get("model_id", "unknown")
        prompt = PARSE_PROMPT.format(model_id=model_id, description=description.strip())
        # mild repetition penalty: greedy list-generation is where models loop
        text = self.generate(prompt, max_tokens=400, ablated=False,
                             repetition_penalty=1.05)
        return extract_goals_json(text)

    # --- rule stack for one candidate --------------------------------------

    def apply_candidate(self, cand, goal_tokens):
        """Adds the candidate's rules; returns their ids for removal."""
        added = []
        for src_ids, _src_str in goal_tokens["sources"]:
            rules = self.interventions.add(
                self.lens_manager, self.manager.jl,
                token_ids=src_ids, mode="replace", factor=cand["factor"],
                keep=cand.get("keep", 0.0),
                replacement_ids=goal_tokens["target"][0],
                anchor_decay=cand["decay"], layers=cand["layers"],
            )
            added.append(rules[-1]["id"])
            if cand["companions"] and len(src_ids) > 1:
                # rank-2 companion: fully spend the extra source pieces so the
                # replacement does not re-fire on their residue (measured:
                # without this, "What is your name?" loops)
                for piece in src_ids[1:]:
                    rules = self.interventions.add(
                        self.lens_manager, self.manager.jl,
                        token_ids=[piece], mode="scale", factor=0.0,
                        layers=cand["layers"],
                    )
                    added.append(rules[-1]["id"])
        self.interventions.set_preserve_lm_head(cand["preserve"])
        return added

    def remove_rules(self, ids):
        for rid in ids:
            self.interventions.remove(rid)

    def apply_style(self, cand, style_tokens):
        """Scale rules for a style goal: boost words ×boost, suppress words
        ×suppress, on the candidate's concept layers."""
        added = []
        for ids, _s in style_tokens["boost"]:
            rules = self.interventions.add(
                self.lens_manager, self.manager.jl, token_ids=ids,
                mode="scale", factor=cand["boost"], layers=cand["layers"])
            added.append(rules[-1]["id"])
        for ids, _s in style_tokens["suppress"]:
            rules = self.interventions.add(
                self.lens_manager, self.manager.jl, token_ids=ids,
                mode="scale", factor=cand["suppress"], layers=cand["layers"])
            added.append(rules[-1]["id"])
        self.interventions.set_preserve_lm_head(False)
        return added

    # --- scoring ------------------------------------------------------------

    def score_candidate(self, cand, goal, goal_tokens, state):
        target_word = goal["target"].strip().casefold()
        prompts = ADDRESS_PROMPTS if goal.get("battery") == "address" else IDENTITY_PROMPTS

        samples = []
        identity = 0.0
        bad = 0
        for prompt in prompts:
            if self.stop_event.is_set():
                raise InterruptedError
            text = self.generate(prompt)
            samples.append({"prompt": prompt, "text": text[:220]})
            low = text.casefold()
            if degenerate(text):
                bad += 1
            elif target_word and target_word in low:
                identity += 1.0
            elif len(target_word) >= 5 and target_word[:4] in low:
                identity += 0.4  # partial adoption ("Luna" for "Lunaria")
        identity /= len(prompts)

        vocab = 0.0
        vocab_n = 0
        for _ids, src_str in goal_tokens["sources"]:
            word = src_str.strip()
            for prompt, expect_word in (
                (f"What is {word}? Answer in one short sentence.", True),
                (f"Spell the word {word} letter by letter.", False),
            ):
                if self.stop_event.is_set():
                    raise InterruptedError
                text = self.generate(prompt)
                vocab_n += 1
                if degenerate(text):
                    bad += 1
                    continue
                if expect_word and word.casefold() not in text.casefold():
                    vocab += 0.5  # coherent but dodges the word
                else:
                    vocab += 1.0
        vocab = vocab / vocab_n if vocab_n else 1.0

        control = 0.0
        for prompt, expects in CONTROL_PROMPTS:
            if self.stop_event.is_set():
                raise InterruptedError
            text = self.generate(prompt)
            if degenerate(text):
                bad += 1
            elif any(e.casefold() in text.casefold() for e in expects):
                control += 1.0
        control /= len(CONTROL_PROMPTS)

        score = total_score(identity, vocab, control, bad)
        return {
            "candidate": cand["name"],
            "preserve_lm_head": cand["preserve"],
            "factor": cand["factor"],
            "keep": cand.get("keep", 0.0),
            "decay": cand["decay"],
            "layers": [cand["layers"][0], cand["layers"][-1]],
            "identity": round(identity, 2),
            "vocabulary": round(vocab, 2),
            "control": round(control, 2),
            "degenerate_replies": bad,
            "score": round(score, 2),
            "samples": samples[:2],
        }

    def _word_rate(self, words):
        """Fraction of the style prompts whose reply contains any of ``words``
        (case-insensitive substring). Also flags degeneration."""
        if not words:
            return 0.0, 0
        hit = bad = 0
        for prompt in STYLE_PROMPTS:
            if self.stop_event.is_set():
                raise InterruptedError
            text = self.generate(prompt, max_tokens=STYLE_GEN_TOKENS)
            if degenerate(text):
                bad += 1
                continue
            low = text.casefold()
            if any(w in low for w in words):
                hit += 1
        return hit / len(STYLE_PROMPTS), bad

    def score_style(self, cand, goal, style_tokens, baseline, state):
        """A style candidate is good when it lifts the boosted words above
        baseline and pushes the suppressed words below it, without breaking
        coherence."""
        boost_words = [s.strip().casefold() for _i, s in style_tokens["boost"]]
        supp_words = [s.strip().casefold() for _i, s in style_tokens["suppress"]]
        boost_rate, b1 = self._word_rate(boost_words)
        supp_rate, b2 = self._word_rate(supp_words)

        control, bad = 0.0, b1 + b2
        samples = []
        for prompt, expects in CONTROL_PROMPTS:
            if self.stop_event.is_set():
                raise InterruptedError
            text = self.generate(prompt)
            samples.append({"prompt": prompt, "text": text[:180]})
            if degenerate(text):
                bad += 1
            elif any(e.casefold() in text.casefold() for e in expects):
                control += 1.0
        control /= len(CONTROL_PROMPTS)

        boost_gain = boost_rate - baseline["boost"]      # want > 0
        supp_drop = baseline["suppress"] - supp_rate     # want > 0
        # coherence gates: a stylistic shift that breaks the model is worthless
        score = control * (2.0 * boost_gain + 2.0 * supp_drop + 1.0) - 1.5 * bad
        return {
            "candidate": cand["name"],
            "preserve_lm_head": False,
            "factor": cand["boost"],
            "keep": 0.0,
            "decay": cand["suppress"],  # reuse the column for the suppress factor
            "layers": [cand["layers"][0], cand["layers"][-1]],
            "identity": round(boost_gain, 2),   # shown as "boost gain"
            "vocabulary": round(supp_drop, 2),  # shown as "suppress drop"
            "control": round(control, 2),
            "degenerate_replies": bad,
            "score": round(score, 2),
            "samples": samples[:2],
        }

    # --- spell-chaser -------------------------------------------------------

    def chase(self, cand, goal_tokens, state, max_chains=2):
        """Head-transformed candidates redirect the FIRST piece of the name and
        the model then free-completes it ("Amaze" for "Ametista"). The chaser
        finds the diverging piece in the actual reply and chains it to the
        remaining target pieces, iterating until spelled or out of budget.
        Returns ``(rule_ids, chains)`` with the rules left ACTIVE."""
        tokenizer = self.manager.tokenizer
        target_ids, _ = goal_tokens["target"]
        target_word = None  # the target as it should appear in text
        added = self.apply_candidate(cand, goal_tokens)
        chains = []
        for _ in range(max_chains):
            if self.stop_event.is_set():
                raise InterruptedError
            text = self.generate(IDENTITY_PROMPTS[0])
            if degenerate(text):
                break
            target_word = tokenizer.decode(target_ids).strip()
            if target_word.casefold() in text.casefold():
                break  # fully spelled
            first = tokenizer.decode([target_ids[0]]).strip()
            attempt = next(
                (w.strip("*,.!?;:()\"'") for w in text.split()
                 if w.strip("*,.!?;:()\"'").startswith(first)), None)
            if not attempt:
                break
            got = tokenizer.encode(" " + attempt, add_special_tokens=False)
            k = 0
            while k < len(got) and k < len(target_ids) and got[k] == target_ids[k]:
                k += 1
            if k == 0 or k >= len(target_ids) or k >= len(got):
                break
            wrong, remaining = got[k], target_ids[k:]
            if any(c["from_id"] == wrong for c in chains):
                break  # already chained this piece — no progress
            state.update(step=f"spell-chaser: «{tokenizer.decode([wrong])}» → "
                              f"«{tokenizer.decode(remaining)}»")
            rules = self.interventions.add(
                self.lens_manager, self.manager.jl,
                token_ids=[wrong], mode="replace", factor=cand["factor"],
                replacement_ids=remaining, anchor_decay=0.4,
                layers=cand["layers"],
            )
            added.append(rules[-1]["id"])
            chains.append({
                "from_id": wrong, "from": tokenizer.decode([wrong]),
                "to": tokenizer.decode(remaining),
            })
        return added, chains

    def _run_style_goal(self, goal, tokenizer, n_layers, rebase_supported):
        """Search boost/suppress scale strengths against a measured baseline."""
        state = self.state

        def resolve_list(words):
            out, skip = [], []
            for w in (words or [])[:8]:
                try:
                    out.append(resolve_word(tokenizer, w))
                except ValueError:
                    skip.append(w)
            return out, skip

        boost, skip_b = resolve_list(goal.get("boost"))
        suppress, skip_s = resolve_list(goal.get("suppress"))
        if not boost and not suppress:
            raise ValueError("style goal has no resolvable boost/suppress words")
        style_tokens = {"boost": boost, "suppress": suppress}

        # baseline word rates with NO rules (detached), to measure the shift
        state.update(step="style: measuring baseline")
        base_boost, _ = self._word_rate([s.strip().casefold() for _i, s in boost])
        base_supp, _ = self._word_rate([s.strip().casefold() for _i, s in suppress])
        baseline = {"boost": base_boost, "suppress": base_supp}

        cands = style_candidates(n_layers, rebase_supported)
        scored = []
        for i, cand in enumerate(cands):
            if self.stop_event.is_set():
                raise InterruptedError
            state.update(step=f"style goal: candidate {i + 1}/{len(cands)} ({cand['name']})")
            added = self.apply_style(cand, style_tokens)
            try:
                scored.append(self.score_style(cand, goal, style_tokens, baseline, state))
            finally:
                self.remove_rules(added)
        scored.sort(key=lambda s: s["score"], reverse=True)
        winner = next(c for c in cands if c["name"] == scored[0]["candidate"])
        self.apply_style(winner, style_tokens)  # leave the winner active
        return {
            "goal": goal,
            "skipped_sources": skip_b + skip_s,
            "baseline": {"boost": round(base_boost, 2), "suppress": round(base_supp, 2)},
            "winner": scored[0],
            "scoreboard": scored,
        }

    # --- main loop ----------------------------------------------------------

    def run(self, goals, state):
        self.state = state
        jl = self.manager.jl
        tokenizer = self.manager.tokenizer
        n_layers = len(jl.layers)
        prior_mode = self.interventions.mode
        prior_preserve = self.interventions.preserve_lm_head
        # pure-weights mode this architecture can bake: read projection, or the
        # global projection (W_U abliteration) on write-norm models (Gemma)
        rebase_supported = (self.manager.meta or {}).get("rebase_supported", True)
        self.interventions.set_mode("readthrough" if rebase_supported else "abliteration")

        results = []
        try:
            for goal in goals:
                gtype = goal.get("type")
                if gtype == "style":
                    results.append(self._run_style_goal(
                        goal, tokenizer, n_layers, rebase_supported))
                    continue
                if gtype != "rename":
                    raise ValueError(f"unknown goal type: {gtype}")
                # resolve sources tolerantly (a parsed description may include
                # words the tokenizer can't sensibly represent) and cap them:
                # every source multiplies rules AND vocabulary probes
                sources, skipped = [], []
                for word in goal["sources"][:4]:
                    try:
                        sources.append(resolve_word(tokenizer, word))
                    except ValueError:
                        skipped.append(word)
                skipped += goal["sources"][4:]
                if not sources:
                    raise ValueError(f"no resolvable source word in {goal['sources']}")
                goal_tokens = {
                    "sources": sources,
                    "target": resolve_word(tokenizer, goal["target"]),
                }
                cands = candidates_for(n_layers, rebase_supported)
                scored = []
                for i, cand in enumerate(cands):
                    if self.stop_event.is_set():
                        raise InterruptedError
                    state.update(step=f"goal «{goal['target']}»: candidate "
                                      f"{i + 1}/{len(cands)} ({cand['name']})")
                    added = self.apply_candidate(cand, goal_tokens)
                    try:
                        scored.append(self.score_candidate(cand, goal, goal_tokens, state))
                    finally:
                        self.remove_rules(added)

                # spell-chaser pass on the best head-transformed candidate: for
                # names the model cannot spell, chaining the diverging piece is
                # what closes the gap ("Amaze" → chain «aze»→«etista»)
                exact_scored = [s for s in scored if not s["preserve_lm_head"]]
                chained_ids = None
                if exact_scored and len(goal_tokens["target"][0]) > 1:
                    # base = the exact candidate closest to the goal: highest
                    # identity first, fewest loops, then score
                    base_name = max(
                        exact_scored,
                        key=lambda s: (s["identity"], -s["degenerate_replies"], s["score"]),
                    )["candidate"]
                    base = next(c for c in cands if c["name"] == base_name)
                    chained_ids, chains = self.chase(base, goal_tokens, state)
                    if chains:
                        entry = self.score_candidate(
                            {**base, "name": f"{base_name}+chain"},
                            goal, goal_tokens, state)
                        entry["chains"] = [
                            {"from": c["from"], "to": c["to"]} for c in chains]
                        # the chain rules are already active; count them as the
                        # candidate under evaluation
                        scored.append(entry)
                    else:
                        self.remove_rules(chained_ids)
                        chained_ids = None

                scored.sort(key=lambda s: s["score"], reverse=True)
                winner_entry = scored[0]
                if chained_ids is not None and winner_entry.get("chains"):
                    pass  # chained stack already active — keep it
                else:
                    if chained_ids is not None:
                        self.remove_rules(chained_ids)
                    winner = next(c for c in cands if c["name"] == winner_entry["candidate"])
                    # leave the winner's rules ACTIVE (cumulative across goals)
                    self.apply_candidate(winner, goal_tokens)
                results.append({
                    "goal": goal,
                    "skipped_sources": skipped,
                    "winner": winner_entry,
                    "scoreboard": scored,
                })
            # persist the winner: rules live only in memory, and losing a
            # multi-minute search to a server restart is not acceptable
            preset_name = None
            try:
                label = goals[0].get("target") or (goals[0].get("boost") or ["style"])[0]
                stem = re.sub(r"[^\w.\- ]+", "", str(label)).strip() or "result"
                preset_name = f"wizard-{stem}"
                editing.save_preset(
                    preset_name, self.interventions.summary(),
                    (self.manager.meta or {}).get("model_id"),
                    scale=self.interventions.global_scale,
                    mode=self.interventions.mode,
                    preserve_lm_head=self.interventions.preserve_lm_head,
                )
            except Exception:  # noqa: BLE001 — a preset failure must not fail the run
                preset_name = None
            state.update(state="done", step=None, result=results, preset=preset_name)
        except InterruptedError:
            self.interventions.set_mode(prior_mode)
            self.interventions.set_preserve_lm_head(prior_preserve)
            state.update(state="stopped", step=None, result=results or None)
        return results
