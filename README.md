# Free-model aggregator 01

[![make.fun](https://make.fun/badge/bounty/free-model-aggregator-01)](https://make.fun/bounty/free-model-aggregator-01)
[![make.fun](https://make.fun/badge/submission/free-model-aggregator-01)](https://make.fun/bounty/free-model-aggregator-01)
[![make.fun](https://make.fun/badge/owner/mutah123)](https://make.fun/r/0cws2dt0)

The best free AI model aggregator: one server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers, in JSON and in server-sent events, and registers the catalog it read from them.

Each trial sends it 20 requests; it answers each with the best free model that can serve it, fails over when a provider refuses, and exits 0 when the env says the workload is done.

## How it works

- Reads the brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the quality table and the workload manifest.
- Probes every provider's catalog route in its own wire (OpenAI, Anthropic, Gemini) and builds one catalog.
- Registers the catalog with the env (`POST /register`).
- Serves `POST /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models`.
- For each request it ranks every free model that meets the request's requirements by the quality table's `rating_milli` (best first) and tries them in order, failing over on 402/429/5xx. Providers that refuse are marked for the rest of the trial, which spreads the per-trial budgets.
- All three upstream wires are served: OpenAI directly, Anthropic and Gemini through `lib/upstream.ts`, which translates their requests and answers (including SSE streams) to and from OpenAI shape.
- `node main.ts --list` prints the catalog as one JSON document and exits 0.

## Practice scores

Local `grade-local` on the kit's three practice instances: mean **94.7** (quality ~94, served ~95, wire ~95, failover ~90, catalog 100).

## Run

```sh
node main.ts --list
node main.ts
```

Node 22+, standard library only. No model of its own: every answer comes from one of the ten providers, through the env.

## License

MIT. See [LICENSE](LICENSE).
