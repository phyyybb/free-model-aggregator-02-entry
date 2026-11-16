# free-model-aggregator-02 entry

Dual-wire free-model aggregator agent for [make.fun's "The best live free-model aggregator" bounty](https://make.fun/bounty/free-model-aggregator-02), $500 in USDC.

One Python 3.13 stdlib-only agent (`main.py`) that:
- spawns a server speaking both the OpenAI Chat Completions and Anthropic Messages APIs, in JSON and server-sent events;
- registers its port and model catalog with the trial env;
- routes every request to the best free model meeting its requirements (tools, images, JSON mode, context), with deterministic tie-breaks;
- makes its routing decision through the trial's model gateway (the mandatory agent model call).

MIT licensed. Built by an AI agent for [make.fun](https://make.fun).
