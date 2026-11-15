# makefun-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/anjay898/makefun-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/anjay898/makefun-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs by routing every request to the best free model across ten free-tier providers: candidates are ranked by the sealed quality table (same-wire first, so a request passes through untouched where it can), failures fail over to the next candidate, and the openai, anthropic and gemini upstream wires are all translated in JSON and in SSE.

## Run it

```sh
MAKEFUN_ENV_URL=<env url> MAKEFUN_PROMPT_FILE=<brief path> MAKEFUN_TRIAL_SEED=<seed> node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=xwewnr0f&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
