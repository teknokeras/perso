# perso

**perso is an embedded ABAC policy engine for MCP tool-call authorization — compiled to WebAssembly, with no control plane, no SaaS account, and no network call in the decision path.**

You define who can call which tools, under what conditions, in a plain JSON file. That file compiles into a single portable `.wasm` binary you load directly into your own process — a backend server, an MCP server, an edge function, or a CLI. There is nothing to deploy, register, or authenticate against. The policy engine ships inside your binary, next to your code, and answers Allow/Deny in-process, in microseconds.

The LLM never touches auth. The host owns the role. perso makes the call at the point where the tool call would be forwarded — without leaving the process.

perso is **stateless and per-call by design**: each decision evaluates one tool call against the attributes you pass in. For agentic loops — where the real risk often lives in a *sequence* of individually-innocent calls — the host or gateway accumulates session context and feeds it in as attributes; perso evaluates whatever context you give it, at microsecond cost per call. See [Security model](#security-model) for where this boundary sits and why.

This isn't just asserted. [perso-benchmark](https://github.com/teknokeras/perso-benchmark) measures it directly: in-process WASM evaluation runs at **~1.75µs median**, versus **~50.3µs** for the same decision over a localhost network call to a PDP-style server — roughly **29× faster at the median, ~53× at p99**. That's the best case for the network path (no TLS, no geographic distance, no multi-tenant queueing) — see the benchmark repo for full methodology, hardware specs, and raw results.

**License: MIT.**

---

## What perso is not

This matters more than what perso *is*, because most authorization tools in this space are platforms, and perso deliberately isn't one.

- **Not a SaaS platform.** No control plane, no hosted dashboard, no account to create. Credit where due: Cerbos also ships an embedded WASM PDP that evaluates in-process — but it's exclusive to their commercial Cerbos Hub, which compiles the policy bundles and distributes them from its control plane. Permit.io and Axiomatics likewise require you to run or talk to their service. perso has no account, no bundle CDN, and no control plane at all — the policy file lives in your repo and compiles on your machine.
- **Not a network call per decision.** The policy is compiled into the WASM binary you load. There is no PDP to reach over HTTP, no latency budget consumed by a round trip. Tools that compile Rego to WASM (e.g. OPA's WASM target) get partway here; perso is built around this as the entire design, not an optional deployment mode. [Measured, not just claimed](https://github.com/teknokeras/perso-benchmark).
- **Not multi-tenant.** If you need centralized policy management across many services, many tenants, and a team that isn't comfortable with policy-as-code, you've outgrown perso — go use Permit.io or Cerbos Hub. That's a legitimate, different problem.
- **Not a dashboard or audit platform.** perso returns a decision and a reason string. It does not store history, render charts, or forward to a SIEM by itself. Wire that up at the host level if you need it.
- **Not trying to support every access-control model.** ReBAC and policy hot-reload at scale are explicitly someone else's job. perso does attribute-based authorization over tool calls, well, and stops there.
- **Not sequence-aware on its own.** perso decides one call at a time. Detecting that call #7 is dangerous *because of* calls #1–6 (intent drift, exfiltration chains) requires a context-accumulation layer above the engine — an [AARM](https://arxiv.org/html/2602.09433v1)-style host or gateway. perso is built to be the fast decision core *under* such a layer: accumulate context outside, pass it in as attributes, and per-call evaluation stays cheap enough to run on every single step.

If your honest requirement is "I want tool-call authorization with zero infrastructure, embedded in a process I already control, in a language I'm already using" — that's the gap perso fills. If your requirement is "I want a managed authorization platform my whole org standardizes on" — perso is the wrong tool, and that's fine.

### Is this ABAC or RBAC?

Hybrid, on purpose. Rules are keyed by `(tool, role)` for O(1) dispatch — that part looks like RBAC. But the role is just one attribute among several: conditions read tool-call **arguments**, caller **session attributes**, and **resource attributes** at decision time (`amount ≤ 500`, `mfa_verified` present, `user_id == owner_id`). The role gets you to the rule; the attributes decide the outcome. If you want a label, "role-indexed ABAC" is the accurate one.

---

## Demo

**[perso-demo](https://github.com/teknokeras/perso-demo)** is an interactive web app that shows perso in action.

An LLM (Groq) chats with the user and calls tools against a mock B2B SaaS CRM. perso intercepts every tool call intent before execution and returns Allow or Deny based on the caller's role and runtime attributes — all evaluated in-process, with no external service involved in the decision. The UI shows the decision inline — green for allow, red for deny — alongside the reason from the policy engine.

Try three roles (agent, manager, admin) across seven CRM tools and watch which calls get through and which get blocked — without touching a single line of auth code in the tool implementations, and without standing up anything beyond the demo app itself.

**Roles and what they demonstrate:**

| Role      | Description                                                                              |
|-----------|------------------------------------------------------------------------------------------|
| `agent`   | Front-line support. Can view and update customers, process refunds up to $500.           |
| `manager` | Team lead. Can delete own records, access PII (with MFA), export data (production only), refunds up to $2,000. |
| `admin`   | Full access. All operations including bulk updates (requires MFA + production env).      |

**Tools and permissions:**

| Tool              | agent | manager | admin | Condition                                                      |
|-------------------|-------|---------|-------|----------------------------------------------------------------|
| `view_customer`   | ✅    | ✅      | ✅    | —                                                              |
| `update_customer` | ✅    | ✅      | ✅    | —                                                              |
| `delete_customer` | ❌    | ✅      | ✅    | manager: `user_id == owner_id` only                            |
| `process_refund`  | ✅    | ✅      | ✅    | agent: `amount ≤ $500` · manager/admin: `amount ≤ $2,000`      |
| `access_pii`      | ❌    | ✅      | ✅    | `mfa_verified` must be present                                 |
| `export_data`     | ❌    | ✅      | ✅    | `env == production` only                                       |
| `bulk_update`     | ❌    | ❌      | ✅    | `env == production` + `mfa_verified`                           |

Default action: **Deny**. Anything not explicitly allowed is rejected.

→ **[github.com/teknokeras/perso-demo](https://github.com/teknokeras/perso-demo)**

---

## The problem perso solves

When an LLM calls a tool through MCP, something has to decide whether that call is allowed. Without a policy layer, the choices are bad: either every tool is wide open, or you scatter auth logic across individual tool implementations, or you bolt on coarse-grained role checks with no ability to inspect arguments.

The broader authorization-for-agents space already has answers to "fine-grained ABAC for tool calls" — Cerbos, Permit.io, Cedar-based gateways, and OPA-to-WASM all address this. What's less settled is the deployment shape: most of these are platforms with a control plane, a hosted decision service, or both. perso answers the same authorization question with a different shape: a structured, testable, swappable policy file compiled into a WASM binary that the host calls directly, in-process, with nothing else to run.

"agents can process refunds, but only up to $500", "managers can delete customer records they own", "this tool is blocked unless MFA is verified" — these rules look the same regardless of which engine evaluates them. The differentiator is what you have to stand up to get the answer: with perso, it's nothing beyond the binary you already ship.

---

## Threat model

perso is one layer in an agentic system's defense, not a complete security solution. This section maps perso's actual coverage against the [OWASP Top 10 for Agentic Applications (2026)](https://genai.owasp.org/), so it's clear exactly what's handled, what's partially mitigated, and what's explicitly out of scope.

### Directly mitigated

**ASI02 — Tool Misuse & Exploitation.** This is perso's core job. Every tool call intent is evaluated against an explicit allow-list of (tool, role, condition) before it reaches the real implementation — evaluated in-process, with no dependency on a reachable external service. An agent role calling `process_refund` with `amount: 50000` is denied by the `NumericCheck` condition regardless of what the LLM was convinced to do — the tool's blast radius is bounded by policy, not by how well the model resists manipulation, and not by whether a network call to a PDP succeeded.

**ASI03 — Agent Identity & Privilege Abuse.** perso's per-role expansion model means a given identity's permissions are explicit and inspectable — no implicit inheritance, no "admin-by-default" tool. `FieldEquals` conditions (e.g. `user_id == owner_id`) enforce ownership boundaries so even a valid role can't act outside its own scope. Compromise of one role's context doesn't expand the privilege of the call beyond what that role's rules allow. (Note: today the caller is a *single* subject — see the delegation item on the [Roadmap](#roadmap) for where user-vs-agent principal separation is headed.)

### Partially mitigated

**ASI01 — Agent Goal Hijack.** perso doesn't prevent an LLM from being manipulated into *wanting* to do something harmful — that's a model-behavior problem, not an authorization problem. But it is the last line of defense when a hijacked agent tries to act on that intent: a hijacked agent role still can't call `bulk_update` without `env == production` and `mfa_verified`, no matter what the prompt convinced it to attempt.

**ASI08 — Cascading Agent Failures.** Default-deny limits the blast radius of any single bad decision, since unspecified calls fail closed rather than open. But perso evaluates calls independently: a sequence of individually-allowed calls that compose into a violation (read PII → export externally) is invisible to a stateless per-call check *unless the host passes session context in as attributes* (e.g. a `has_read_pii` flag in `AgentAttributes` that a later rule can reference). Full mitigation needs both that context-passing pattern and the audit/replay layer (in progress — see Roadmap) to reconstruct *why* a sequence of calls went wrong.

**ASI09 — Human-Agent Trust Exploitation.** Every decision includes a structured `reason` string, giving a human reviewer something concrete to check rather than trusting the agent's own account of what it did. This is a partial mitigation — it supports human review, it doesn't automate it.

### Explicitly out of scope

These are real risks in agentic systems that perso does not address, by design — they belong to other layers of the stack:

- **ASI04 — Agentic Supply Chain Compromise** (compromised dependencies, tool definitions, or model weights)
- **ASI05 — Unexpected Code Execution** (sandboxing the LLM's own generated code; perso sandboxes itself via WASM, not the agent's outputs)
- **ASI06 — Memory & Context Poisoning** (corrupted RAG/context stores feeding the model bad information)
- **ASI07 — Insecure Inter-Agent Communication** (multi-agent protocols and trust between agents)
- **ASI10 — Rogue Agents** (detecting an agent that has gone fully off-policy at the behavioral level)

If you're building a production agentic system, perso should be one control among several — pair it with input validation, model-level guardrails, and monitoring for the categories above.

---

## Architecture

```
Browser / Client
      │
      ▼
   Backend  ◄─── owns session, extracts role from JWT/cookie
      │
      │  calls evaluate(tool, args, context) on every tool call
      │  (in-process — no network hop to a policy service)
      ▼
  perso WASM  ──► Allow → forward to MCP
                  Deny  → reject, return error to LLM
      │
      ▼
  MCP Server  (can also embed perso for defense-in-depth)
      │
      ▼
  Core System
```

The LLM returns a tool call intent. The backend intercepts it, builds a context object (role, agent attributes, resource attributes), and asks perso. perso answers in one O(1) map lookup plus optional condition evaluation — entirely within the host process's memory space. The answer is always `{ "decision": "Allow", "reason": "..." }` or `{ "decision": "Deny", "reason": "..." }`.

---

## Repository layout

```
perso/
├── Cargo.toml                  # workspace root — all shared deps declared here
├── LICENSE                     # MIT
├── policies/
│   └── example.json            # ready-to-use example policy
└── crates/
    ├── policy-core/            # shared types, serde AST, parse_policy()
    ├── policy-runtime/         # glob expander, condition evaluator, WASM exports
    ├── policy-compiler/        # CLI: validate and build commands
    └── policy-test/            # integration test suite
```

---

## The four crates

### policy-core

The shared data model. Every other crate depends on it. Contains:

- `Policy` — the top-level document (version, default_action, tools, rules)
- `Rule` — a single access rule: tool name (or glob), roles, optional condition
- `Condition` — a recursive enum: `All`, `Any`, `Not`, `NumericCheck`, `StringCheck`, `FieldPresent`, `FieldEquals`
- `Source` — where a condition reads its data from: `Arguments`, `AgentAttributes`, or `ResourceAttributes`
- `EvaluationRequest` / `EvaluationContext` / `EvaluationResponse` — the evaluation boundary types
- `Decision` — `Allow` or `Deny`
- `parse_policy(json: &str)` — deserialises a JSON string into a `Policy`

No evaluation logic lives here. It is pure types and parsing.

### policy-runtime

The evaluation engine, compiled to `cdylib` for WASM. Three modules:

**`expander`** — runs at init time. Iterates every rule, expands glob patterns (e.g. `crm_tool_*`) against the `tools[]` array in the policy, and builds a flat `HashMap<(tool_name, role), Option<Condition>>`. After init there are zero wildcards in memory. Every lookup at evaluate time is O(1).

**`evaluator`** — runs on every tool call. Takes the pre-built map plus the three JSON input bags (arguments, agent attributes, resource attributes) and recurses through the condition tree. Returns an `EvaluationResponse` with a decision and a human-readable reason.

**`wasm`** — the WASM ABI layer. Exports four functions to the host:

| Export     | Signature                    | Purpose                                                         |
|------------|------------------------------|-----------------------------------------------------------------|
| `alloc`    | `(len: i32) → i32`           | Host calls this to allocate memory before writing input strings |
| `dealloc`  | `(ptr: i32, len: i32)`       | Host calls this to free buffers after reading responses         |
| `init`     | `(ptr: i32, len: i32) → i32` | Load and materialise a policy JSON string                       |
| `evaluate` | `(6× i32) → i32`             | Evaluate a tool call; returns a length-prefixed JSON buffer     |

All strings cross the WASM boundary as `(pointer, length)` pairs into linear memory. Return buffers are length-prefixed: the first 4 bytes are a little-endian `u32` containing the body length, followed by UTF-8 JSON. The host reads the length, reads the body, then calls `dealloc`.

Policy state is held in a `OnceLock<Mutex<PolicyState>>`. Calling `init` again replaces the state, enabling hot-reload without restarting the host.

### policy-compiler

A CLI with two subcommands.

**`validate`** — parses the policy JSON, expands all globs, and reports warnings and errors without producing any output file. Useful in CI.

**`build`** — validates the policy, then invokes `cargo build --release --target wasm32-unknown-unknown -p policy-runtime`, and copies the resulting `.wasm` to the path you specify.

### policy-test

The integration test suite, split into two layers:

**`native_tests`** — calls the Rust API of `policy-runtime` directly. Covers all 18 spec cases. Runs with a plain `cargo test` and no extra setup.

**`wasm_tests`** — loads the compiled `.wasm` binary via `wasmtime` and drives every spec case through the real WASM ABI. Enabled by setting the `PERSO_WASM` environment variable. Skipped gracefully otherwise.

---

## The policy JSON format

```json
{
  "version": "perso-1.0.0",
  "default_action": "Deny",
  "tools": ["view_customer", "update_customer", "delete_customer", "process_refund", "access_pii", "export_data", "bulk_update"],
  "rules": [
    { "tool_name": "view_customer", "roles": ["agent", "manager", "admin"], "condition": null },
    { "tool_name": "process_refund", "roles": ["agent"], "condition": {
        "NumericCheck": { "source": "Arguments", "field": "amount", "op": "Lte", "value": 500.0 }
    }}
  ]
}
```

| Field            | Required | Description                                                    |
|------------------|----------|------------------------------------------------------------------|
| `version`        | yes      | Schema version string, e.g. `"perso-1.0.0"`                   |
| `default_action` | yes      | `"Allow"` or `"Deny"` — applied when no rule matches          |
| `tools`          | yes      | All known tool names. The expansion universe for glob patterns |
| `rules`          | yes      | Ordered list of access rules                                   |

Each rule:

| Field       | Required | Description                                            |
|-------------|----------|--------------------------------------------------------|
| `tool_name` | yes      | Concrete name or glob pattern (`*` wildcard only)      |
| `roles`     | yes      | List of role strings that this rule grants access to   |
| `condition` | yes      | `null` for unconditional access, or a condition object |

### Condition types

**`NumericCheck`** — compare a numeric field to a literal value.

```json
{ "NumericCheck": { "source": "Arguments", "field": "amount", "op": "Lte", "value": 500.0 } }
```

Operators: `Lte`, `Gte`, `Eq`, `Lt`, `Gt`

**`StringCheck`** — check whether a string field is in or not in a list.

```json
{ "StringCheck": { "source": "AgentAttributes", "field": "env", "op": "In", "value": ["production"] } }
```

Operators: `In`, `NotIn`

**`FieldPresent`** — assert a field exists and is not null.

```json
{ "FieldPresent": { "source": "AgentAttributes", "field": "mfa_verified" } }
```

**`FieldEquals`** — assert a field in one source equals a field in another.

```json
{ "FieldEquals": { "source_a": "AgentAttributes", "field_a": "user_id", "source_b": "ResourceAttributes", "field_b": "owner_id" } }
```

**`All`**, **`Any`**, **`Not`** — logical combinators.

```json
{ "All": [ "<condition>", "<condition>" ] }
{ "Any": [ "<condition>", "<condition>" ] }
{ "Not": "<condition>" }
```

**Source values:**

- `"Arguments"` — the tool-call arguments produced by the LLM. Untrusted by definition.
- `"AgentAttributes"` — attributes of the **calling session**, supplied by the host (role claims, `mfa_verified`, `env`, `user_id`, …). **Naming note:** "agent" here means *the caller's session* — which today is a single subject bag, whether the caller is a human user or an AI agent acting on a human's behalf. perso does not yet model the user and the AI agent as two separate principals whose permissions intersect (the confused-deputy problem). That separation is on the [Roadmap](#roadmap); until then, hosts that need it can namespace fields inside this bag (e.g. `user_role`, `agent_trust`) and write conditions against both.
- `"ResourceAttributes"` — attributes of the resource being acted on (e.g. `owner_id`), supplied by the host.

### Glob expansion

Tool names in rules may contain `*` as a wildcard. The `tools[]` array is the expansion universe — a glob is matched against every name in that list at `init` time. No wildcards survive past initialisation.

```json
"tools": ["crm_tool_alpha", "crm_tool_beta", "other_tool"],
"rules": [
  { "tool_name": "crm_tool_*", "roles": ["admin"], "condition": null }
]
```

This expands into two map entries: `(crm_tool_alpha, admin)` and `(crm_tool_beta, admin)`. `other_tool` is unaffected.

---

## Getting started

### Prerequisites

- Rust (stable, 1.75+)
- `wasm32-unknown-unknown` target for compiling the WASM binary:

```
rustup target add wasm32-unknown-unknown
```

### 1. Clone and build

```
git clone https://github.com/teknokeras/perso.git
cd perso
cargo build
```

### 2. Validate your policy

```
cargo run -p policy-compiler -- validate --policy policies/example.json
```

Output:

```
perso: validating policies/example.json
  glob 'crm_tool_*' → 2 tool(s) × 1 role(s) = 2 map entries
ok: 10 rule(s), 13 tool(s), 13 map entries, 0 warning(s)
```

### 3. Build the WASM binary

```
cargo run -p policy-compiler -- build \
  --policy policies/example.json \
  --output dist/policy_runtime.wasm
```

### 4. Run the test suite

```
# Native tests (no WASM binary needed)
cargo test -p policy-test

# Full end-to-end WASM boundary tests
PERSO_WASM=dist/policy_runtime.wasm cargo test -p policy-test
```

### 5. Run all tests across the workspace

```
cargo test
```

No service to start, no account to create, no environment variables for a hosted endpoint. Steps 1–4 are the entire setup.

---

## Embedding in a host (Rust + wasmtime)

```rust
use wasmtime::{Engine, Linker, Module, Store};

// Load once at startup
let engine = Engine::default();
let module = Module::from_file(&engine, "dist/policy_runtime.wasm")?;
let linker = Linker::new(&engine);
let mut store = Store::new(&engine, ());
let instance = linker.instantiate(&mut store, &module)?;

// Initialise with the policy JSON (call again to hot-reload)
let policy_json = std::fs::read_to_string("policies/example.json")?;
// write policy_json into WASM memory via alloc, call init(), read response

// On every tool call:
// 1. Build context: { role, agent_attrs, resource_attrs }
// 2. Write tool_name, args_json, context_json into WASM memory via alloc
// 3. Call evaluate()
// 4. Read the length-prefixed response buffer
// 5. Deserialise: { "decision": "Allow"/"Deny", "reason": "..." }
// 6. Dealloc the response buffer
```

See `crates/policy-test/src/lib.rs` (`mod wasm_tests`) for a complete, working `PersoWasm` harness that demonstrates the full alloc → write → call → read → dealloc cycle.

---

## Embedding in a Node.js host

**Two ways to integrate:**

- **The SDK (recommended):** `npm install @teknokeras/perso-sdk` — wraps the entire ABI below in a typed API, plus audit logging. See [perso-sdk-node](https://github.com/teknokeras/perso-sdk-node).
- **The raw ABI (shown below):** zero dependencies — Node.js has built-in `WebAssembly` support. Use this if you want no packages at all, or as a reference for what the SDK does under the hood.

The WASM binary contains **no policy**. You load `policy_runtime.wasm` (the engine) and `example.json` (the policy) separately, then pass the policy JSON to `init()` at startup.

```javascript
const fs = require('fs');

// ── 1. Load the engine and the policy ────────────────────────────────────────
const wasmBytes  = fs.readFileSync('dist/policy_runtime.wasm');
const policyJson = fs.readFileSync('policies/example.json', 'utf8');

const { instance } = await WebAssembly.instantiate(wasmBytes);
const { alloc, dealloc, init, evaluate, memory } = instance.exports;

// ── 2. ABI helpers ────────────────────────────────────────────────────────────

function writeString(str) {
  const bytes = new TextEncoder().encode(str);
  const ptr   = alloc(bytes.length);
  new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function readResponse(ptr) {
  const view   = new DataView(memory.buffer);
  const len    = view.getUint32(ptr, /*littleEndian=*/true);
  const body   = new Uint8Array(memory.buffer, ptr + 4, len);
  const result = JSON.parse(new TextDecoder().decode(body));
  dealloc(ptr, 4 + len);
  return result;
}

// ── 3. Initialise the engine with the policy (once at startup) ───────────────
const [iPtr, iLen] = writeString(policyJson);
const initResp = readResponse(init(iPtr, iLen));
console.log(initResp);
// { ok: true }

// ── 4. Evaluate a tool call ───────────────────────────────────────────────────
function checkToolCall(toolName, args, role, agentAttrs = {}, resourceAttrs = {}) {
  const [tPtr, tLen] = writeString(toolName);
  const [aPtr, aLen] = writeString(JSON.stringify(args));
  const [cPtr, cLen] = writeString(JSON.stringify({
    role,
    agent_attrs:    agentAttrs,
    resource_attrs: resourceAttrs,
  }));
  return readResponse(evaluate(tPtr, tLen, aPtr, aLen, cPtr, cLen));
}

// Examples from the CRM demo scenario:
console.log(checkToolCall('view_customer', { id: 'C-1042' }, 'agent'));
// { decision: 'Allow', reason: "rule matched tool 'view_customer' for role 'agent'; no condition required" }

console.log(checkToolCall('process_refund', { amount: 200 }, 'agent'));
// { decision: 'Allow', reason: "...condition passed" }

console.log(checkToolCall('process_refund', { amount: 800 }, 'agent'));
// { decision: 'Deny', reason: "NumericCheck failed: amount 800 exceeds 500" }

console.log(checkToolCall('delete_customer', { id: 'C-9001' }, 'manager', { user_id: 'mgr-001' }, { owner_id: 'mgr-002' }));
// { decision: 'Deny', reason: "FieldEquals failed: user_id != owner_id" }

console.log(checkToolCall('bulk_update', {}, 'admin', { env: 'production', mfa_verified: true }));
// { decision: 'Allow', reason: "...all conditions passed" }

// ── 5. Hot-reload the policy ──────────────────────────────────────────────────
const newPolicyJson = fs.readFileSync('policies/updated.json', 'utf8');
const [rPtr, rLen]  = writeString(newPolicyJson);
console.log(readResponse(init(rPtr, rLen)));
// { ok: true }
```

---

## Embedding in a Python host

**Two ways to integrate:**

- **The SDK (recommended):** `pip install perso-sdk` — same shape as the Node SDK, synchronous API. See [perso-sdk-python](https://github.com/teknokeras/perso-sdk-python).
- **The raw ABI (shown below):** just the official Bytecode Alliance wasmtime binding, as a reference for what the SDK does under the hood.

```
pip install wasmtime
```

```python
from wasmtime import Engine, Linker, Module, Store
import json, struct

# ── 1. Load the engine and the policy ────────────────────────────────────────
engine   = Engine()
module   = Module.from_file(engine, 'dist/policy_runtime.wasm')
linker   = Linker(engine)
store    = Store(engine)
instance = linker.instantiate(store, module)

with open('policies/example.json', 'r') as f:
    policy_json = f.read()

alloc   = instance.exports(store)['alloc']
dealloc = instance.exports(store)['dealloc']
init_fn = instance.exports(store)['init']
eval_fn = instance.exports(store)['evaluate']
memory  = instance.exports(store)['memory']

# ── 2. ABI helpers ────────────────────────────────────────────────────────────

def write_string(s: str):
    data = s.encode('utf-8')
    ptr  = alloc(store, len(data))
    memory.write(store, data, ptr)
    return ptr, len(data)

def read_response(ptr: int) -> dict:
    header = bytes(memory.read(store, ptr, ptr + 4))
    length = struct.unpack_from('<I', header)[0]
    body   = bytes(memory.read(store, ptr + 4, ptr + 4 + length))
    dealloc(store, ptr, 4 + length)
    return json.loads(body)

# ── 3. Initialise the engine with the policy (once at startup) ───────────────
ptr, length = write_string(policy_json)
print(read_response(init_fn(store, ptr, length)))
# {'ok': True}

# ── 4. Evaluate a tool call ───────────────────────────────────────────────────
def check_tool_call(tool_name: str, args: dict, role: str,
                    agent_attrs: dict = {}, resource_attrs: dict = {}) -> dict:
    tp, tl = write_string(tool_name)
    ap, al = write_string(json.dumps(args))
    cp, cl = write_string(json.dumps({
        'role':           role,
        'agent_attrs':    agent_attrs,
        'resource_attrs': resource_attrs,
    }))
    return read_response(eval_fn(store, tp, tl, ap, al, cp, cl))

# Examples from the CRM demo scenario:
print(check_tool_call('view_customer', {'id': 'C-1042'}, 'agent'))
# {'decision': 'Allow', 'reason': "rule matched tool 'view_customer' for role 'agent'; no condition required"}

print(check_tool_call('process_refund', {'amount': 200}, 'agent'))
# {'decision': 'Allow', 'reason': "...condition passed"}

print(check_tool_call('process_refund', {'amount': 800}, 'agent'))
# {'decision': 'Deny', 'reason': "NumericCheck failed: amount 800 exceeds 500"}

print(check_tool_call('access_pii', {}, 'manager', {}))
# {'decision': 'Deny', 'reason': "FieldPresent failed: mfa_verified not in agent_attrs"}

print(check_tool_call('bulk_update', {}, 'admin', {'env': 'production', 'mfa_verified': True}))
# {'decision': 'Allow', 'reason': "...all conditions passed"}

# ── 5. Hot-reload the policy ──────────────────────────────────────────────────
with open('policies/updated.json', 'r') as f:
    new_policy_json = f.read()

ptr, length = write_string(new_policy_json)
print(read_response(init_fn(store, ptr, length)))
# {'ok': True}
```

---

## The WASM ABI (for any other host language)

Any runtime with a WASM host — Go (`wasmtime-go` or `wazero`), Java (`wasmtime-java`), Ruby, Elixir, etc. — works the same way. The pattern is always:

1. Call `alloc(len)` → get a pointer into WASM linear memory
2. Write your UTF-8 string at that pointer
3. Call `init(ptr, len)` or `evaluate(tp,tl, ap,al, cp,cl)`
4. Read the 4-byte little-endian `u32` length prefix from the returned pointer
5. Read that many bytes as UTF-8 JSON
6. Call `dealloc(ptr, 4 + len)` to free the buffer

Response shapes:

```json
{ "ok": true }
{ "error": "policy parse error: ..." }
{ "decision": "Allow", "reason": "rule matched tool 'view_customer' for role 'agent'; no condition required" }
{ "decision": "Deny",  "reason": "no rule matched tool 'delete_customer' for role 'agent'; applying default_action" }
```

---

## Hot-reloading the policy

Call `init` again with new policy JSON at any time. The `Mutex<PolicyState>` inside the WASM module is replaced atomically. No restart required, and no coordination with an external service. For multi-threaded hosts, wrap the `Store` in an `Arc<RwLock<>>` so concurrent `evaluate` calls read safely while the reload write is in progress.

---

## Security model

- The LLM never sees or touches the role token. The host extracts the role from its own JWT/session at connection time.
- `default_action: "Deny"` means anything not explicitly allowed is rejected. This is the safe default and the one used in the example policy.
- Conditions are evaluated against three separate JSON bags — arguments, agent attributes, and resource attributes — so the LLM-supplied arguments can never impersonate session data.
- The WASM sandbox means perso has no filesystem, network, or syscall access. It only reads and writes linear memory. This also means a misbehaving policy can't reach out to anything, even by accident.
- **perso is stateless and per-call.** It holds no session history and detects no cross-call patterns. For agentic loops, the host or a gateway should accumulate session context (calls made, data touched, taint flags) and pass it into `AgentAttributes` on each evaluation — rules can then express sequence-aware policies like "deny `export_data` if `has_read_pii` is set." Where accumulation and enforcement should live (in-process vs. a trusted gateway) is a genuine architectural trade-off: in-process fits self-hosted, single-tenant deployments where the host *is* the trust boundary; a gateway fits multi-tenant or lower-trust hosts. perso works in both placements — it is an engine, not a topology.
- For zero-trust deployments, embed perso in both the backend (knows the user role) and the MCP server (knows the service identity). Each layer enforces independently, with no shared external dependency to fail.

See [Threat model](#threat-model) above for how this maps to the OWASP Agentic Applications risk categories.

---

## Example policy walkthrough

```json
{ "tool_name": "process_refund", "roles": ["agent"],
  "condition": { "NumericCheck": { "source": "Arguments", "field": "amount", "op": "Lte", "value": 500.0 } } }
```

Agents can process refunds, but only up to $500. A refund of $800 is denied even for an agent.

```json
{ "tool_name": "delete_customer", "roles": ["manager"],
  "condition": { "FieldEquals": { "source_a": "AgentAttributes", "field_a": "user_id",
                                   "source_b": "ResourceAttributes", "field_b": "owner_id" } } }
```

Managers can delete customer records, but only records they own. Attempting to delete a record owned by another manager is denied via `FieldEquals`.

```json
{ "tool_name": "access_pii", "roles": ["manager", "admin"],
  "condition": { "FieldPresent": { "source": "AgentAttributes", "field": "mfa_verified" } } }
```

Managers and admins can access PII, but only when `mfa_verified` is present in their session attributes.

```json
{ "tool_name": "bulk_update", "roles": ["admin"],
  "condition": { "All": [
    { "StringCheck": { "source": "AgentAttributes", "field": "env", "op": "In", "value": ["production"] } },
    { "FieldPresent": { "source": "AgentAttributes", "field": "mfa_verified" } }
  ]}}
```

Admins can run bulk updates only when the environment is production AND MFA has been verified. Both must pass.

```json
{ "tool_name": "crm_tool_*", "roles": ["admin"], "condition": null }
```

Any tool whose name matches `crm_tool_*` is unconditionally available to admins. The glob is expanded at init time against the `tools[]` array — no wildcard pattern matching at evaluate time.

---

## Test coverage summary

| Crate                      | Tests   | What they cover                                             |
|----------------------------|---------|---------------------------------------------------------------|
| policy-core                | 11      | Parsing every condition type, roundtrip serialisation       |
| policy-runtime (expander)  | 12      | Glob matching edge cases, expansion correctness               |
| policy-runtime (evaluator) | 25      | Every condition type, all logical combinators, default deny   |
| policy-runtime (wasm)      | 19      | Full WASM ABI: alloc/dealloc, init, evaluate, error paths     |
| policy-compiler            | 6       | validate happy/sad paths, edge cases                           |
| policy-test (native)       | 19      | All 18 spec cases + map build                                  |
| policy-test (wasm)         | 19      | Same 18 spec cases through real WASM boundary                  |
| **Total**                  | **111** |                                                               |

---

## Roadmap

- ~~**Latency benchmark**~~ — done. See [perso-benchmark](https://github.com/teknokeras/perso-benchmark) for the full methodology and results: in-process WASM evaluation is ~29× faster at the median than the same decision over a localhost network call, ~53× at p99.
- ~~`perso-sdk-python`~~ — done. See [perso-sdk-python](https://github.com/teknokeras/perso-sdk-python).
- **First-class user↔agent delegation.** Today the caller is a single subject. Planned: separate principal bags for the human user and the AI agent acting on their behalf, with policies able to require conditions on *both* — so an AI agent can never exceed the human's permissions, and vice versa (confused-deputy defense). Until then, the documented workaround is namespacing both principals' fields inside `AgentAttributes`.
- **Session-context conventions.** A documented attribute vocabulary for sequence-aware policies (e.g. `session.has_read_pii`, `session.call_count`) so hosts and gateways that accumulate context can write portable rules against it.
- **Audit/replay layer** for reconstructing multi-step decision chains (referenced under ASI08 above).

---

## License

MIT. See [LICENSE](./LICENSE).

---

## Related repos

| Repo | Description |
|---|---|
| [teknokeras/perso-demo](https://github.com/teknokeras/perso-demo) | Interactive CRM demo showing perso enforcing access control on live LLM tool calls, with zero infrastructure beyond the demo app. Two interchangeable backends (Node and Python) share one frontend. |
| [teknokeras/perso-sdk-node](https://github.com/teknokeras/perso-sdk-node) | Official Node.js SDK (`@teknokeras/perso-sdk`) — wraps the WASM ABI with audit logging |
| [teknokeras/perso-sdk-python](https://github.com/teknokeras/perso-sdk-python) | Official Python SDK (`perso-sdk`) — same shape as the Node SDK, synchronous API |
| [teknokeras/perso-benchmark](https://github.com/teknokeras/perso-benchmark) | Reproducible latency benchmark: in-process WASM vs. localhost network-call PDP, with full methodology and raw results |
