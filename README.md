# perso

**perso** is a policy enforcement engine for MCP (Model Context Protocol) tool calls, compiled to WebAssembly.

It lets you define who can call which tools, under what conditions, in a plain JSON file. That file is compiled into a single portable `.wasm` binary that can run inside any host — a backend server, an MCP server, an edge function, or a CLI — without modification.

The LLM never touches auth. The host owns the role. perso makes the Allow/Deny call in microseconds, at the point where the tool call would be forwarded.

---

## The problem perso solves

When an LLM calls a tool through MCP, something has to decide whether that call is allowed. Without a policy layer, the choices are bad: either every tool is wide open, or you scatter auth logic across individual tool implementations, or you bolt on coarse-grained role checks with no ability to inspect arguments.

perso gives you a third option: a structured, testable, swappable policy file that expresses fine-grained rules — "supervisors can issue refunds, but only up to $500", "admins can edit documents they own or any document", "this tool is blocked unless MFA is verified" — compiled into a WASM binary that the host calls before forwarding anything to MCP.

---

## Architecture

```
Browser / Client
      │
      ▼
   Backend  ◄─── owns session, extracts role from JWT/cookie
      │
      │  calls evaluate(tool, args, context) on every tool call
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

The LLM returns a tool call intent. The backend intercepts it, builds a context object (role, agent attributes, resource attributes), and asks perso. perso answers in one O(1) map lookup plus optional condition evaluation. The answer is always `{ "decision": "Allow", "reason": "..." }` or `{ "decision": "Deny", "reason": "..." }`.

---

## Repository layout

```
perso/
├── Cargo.toml                  # workspace root — all shared deps declared here
├── policies/
│   ├── example.json            # ready-to-use example policy
│   └── policy.schema.json      # JSON Schema for policy validation
└── crates/
    ├── policy-core/            # shared types, serde AST, parse_policy()
    ├── policy-runtime/         # glob expander, condition evaluator, WASM exports
    ├── policy-compiler/        # CLI: validate, build, and validate-and-build commands
    └── policy-test/            # integration test suite
```

---

## The four crates

### policy-core

The shared data model. Every other crate depends on it. Contains:

- `Policy` — the top-level document (version, default_action, tools, roles, rules)
- `Rule` — a single access rule: tool name (or glob), roles, optional condition
- `Condition` — a recursive enum: `All`, `Any`, `Not`, `NumericCheck`, `StringCheck`, `FieldPresent`, `FieldEquals`
- `Source` — where a condition reads its data from: `Arguments`, `AgentAttributes`, or `ResourceAttributes`
- `EvaluationRequest` / `EvaluationContext` / `EvaluationResponse` — the evaluation boundary types
- `Decision` — `Allow` or `Deny`
- `parse_policy(json: &str)` — deserialises a JSON string into a `Policy`

No evaluation logic lives here. It is pure types and parsing.

### policy-runtime

The evaluation engine, compiled to `cdylib` for WASM. Three modules:

**`expander`** — runs at init time. Iterates every rule, expands glob patterns (e.g. `glob_tool_*`) against the `tools[]` array in the policy, and builds a flat `HashMap<(tool_name, role), Option<Condition>>`. After init there are zero wildcards in memory. Every lookup at evaluate time is O(1).

**`evaluator`** — runs on every tool call. Takes the pre-built map plus the three JSON input bags (arguments, agent attributes, resource attributes) and recurses through the condition tree. Returns an `EvaluationResponse` with a decision and a human-readable reason.

**`wasm`** — the WASM ABI layer. Exports four functions to the host:

| Export | Signature | Purpose |
|--------|-----------|---------|
| `alloc` | `(len: i32) → i32` | Host calls this to allocate memory before writing input strings |
| `dealloc` | `(ptr: i32, len: i32)` | Host calls this to free buffers after reading responses |
| `init` | `(ptr: i32, len: i32) → i32` | Load and materialise a policy JSON string |
| `evaluate` | `(6× i32) → i32` | Evaluate a tool call; returns a length-prefixed JSON buffer |

All strings cross the WASM boundary as `(pointer, length)` pairs into linear memory. Return buffers are length-prefixed: the first 4 bytes are a little-endian `u32` containing the body length, followed by UTF-8 JSON. The host reads the length, reads the body, then calls `dealloc`.

Policy state is held in a `OnceLock<Mutex<PolicyState>>`. Calling `init` again replaces the state, enabling hot-reload without restarting the host.

### policy-compiler

A CLI with three subcommands.

**`validate`** — runs the JSON Schema check followed by semantic validation (parse, glob-expand, cross-reference tools and roles). Reports all warnings and errors without producing any output file. Useful in CI.

**`build`** — compiles the policy-runtime engine to WASM without requiring a policy file. Invokes `cargo build --release --target wasm32-unknown-unknown -p policy-runtime` and copies the resulting `.wasm` to the path you specify. The policy JSON is **not** embedded in the binary — it is loaded separately at runtime via `init()`.

**`validate-and-build`** — runs `validate` first and only proceeds to `build` if the policy is valid. The recommended command for most workflows: catches policy errors before spending time on a cargo build.

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
  "tools": ["read_file", "write_file", "glob_tool_alpha", "glob_tool_beta"],
  "roles": ["viewer", "supervisor", "admin"],
  "rules": [
    { "tool_name": "read_file", "roles": ["viewer"], "condition": null },
    { "tool_name": "glob_tool_*", "roles": ["admin"], "condition": null }
  ]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `version` | yes | Schema version string, e.g. `"perso-1.0.0"` |
| `default_action` | yes | `"Allow"` or `"Deny"` — applied when no rule matches |
| `tools` | yes | All known tool names. The expansion universe for glob patterns |
| `roles` | yes | All recognised role names. Every role referenced in rules must appear here |
| `rules` | yes | Ordered list of access rules |

Each rule:

| Field | Required | Description |
|-------|----------|-------------|
| `tool_name` | yes | Concrete name or glob pattern (`*` wildcard only) |
| `roles` | yes | List of role strings that this rule grants access to |
| `condition` | yes | `null` for unconditional access, or a condition object |

### Condition types

**`NumericCheck`** — compare a numeric field to a literal value.
```json
{ "NumericCheck": { "source": "Arguments", "field": "amount", "op": "Lte", "value": 500.0 } }
```
Operators: `Lte`, `Gte`, `Eq`, `Lt`, `Gt`

**`StringCheck`** — check whether a string field is in or not in a list.
```json
{ "StringCheck": { "source": "Arguments", "field": "path", "op": "NotIn", "value": ["/etc/passwd"] } }
```
Operators: `In`, `NotIn`

**`FieldPresent`** — assert a field exists and is not null.
```json
{ "FieldPresent": { "source": "AgentAttributes", "field": "session_token" } }
```

**`FieldEquals`** — assert a field in one source equals a field in another.
```json
{ "FieldEquals": { "source_a": "AgentAttributes", "field_a": "user_id", "source_b": "ResourceAttributes", "field_b": "owner_id" } }
```

**`All`**, **`Any`**, **`Not`** — logical combinators.
```json
{ "All": [ <condition>, <condition> ] }
{ "Any": [ <condition>, <condition> ] }
{ "Not": <condition> }
```

**Source values:** `"Arguments"` (tool call args from the LLM), `"AgentAttributes"` (caller session data), `"ResourceAttributes"` (the resource being acted on).

### Glob expansion

Tool names in rules may contain `*` as a wildcard. The `tools[]` array is the expansion universe — a glob is matched against every name in that list at `init` time. No wildcards survive past initialisation.

```json
"tools": ["glob_tool_alpha", "glob_tool_beta", "other_tool"],
"rules": [
  { "tool_name": "glob_tool_*", "roles": ["admin"], "condition": null }
]
```

This expands into two map entries: `(glob_tool_alpha, admin)` and `(glob_tool_beta, admin)`. `other_tool` is unaffected.

---

## Getting started

### Prerequisites

- Rust (stable, 1.75+)
- `wasm32-unknown-unknown` target for compiling the WASM binary:

```bash
rustup target add wasm32-unknown-unknown
```

### 1. Clone and build

```bash
git clone <repo-url>
cd perso
cargo build
```

### 2. Working with your policy

The `policy-compiler` CLI offers three subcommands depending on what you need:

---

#### Option 1 — Validate only

Parse and semantically check a policy file without producing any output. Use this in CI or whenever you want to verify a policy change before committing.

Two validation layers run in sequence:

1. **JSON Schema** — checks structure, field types, enum values, and duplicate entries against the bundled `policy.schema.json`
2. **Semantic checks** — verifies that every `tool_name` in a rule exists in `tools[]`, every role in a rule exists in `roles[]`, and glob patterns match at least one tool

```bash
cargo run -p policy-compiler -- validate --policy policies/myapp.json
```

Output on success:
```
perso: validating policies/myapp.json
ok: 4 rule(s), 4 tool(s), 4 map entries, 0 warning(s)
```

Output on failure:
```
perso: validating policies/myapp.json
error: rule references tool 'read_folder' which is not listed in tools[]
```

---

#### Option 2 — Build the WASM engine only

Compile the policy-runtime engine to a WASM binary without involving a policy file at all. The policy JSON is **not embedded** in the binary — it is loaded separately at runtime via `init()`.

Use this when the engine itself has changed (Rust code updates) and you already know your policy is valid, or when you want to build the engine once and use it with multiple different policy files.

```bash
cargo run -p policy-compiler -- build --output dist/policy_runtime.wasm
```

Output:
```
perso: building policy-runtime → wasm32-unknown-unknown
note:  policy JSON is not embedded — pass it to init() at runtime
ok: 1234567 bytes → dist/policy_runtime.wasm
```

---

#### Option 3 — Validate then build (recommended)

Validate the policy first and only compile the WASM engine if validation passes. This is the recommended command for most workflows — it catches policy errors before spending time on a cargo build.

```bash
cargo run -p policy-compiler -- validate-and-build \
  --policy policies/myapp.json \
  --output dist/policy_runtime.wasm
```

Output on success:
```
perso: validating policy before build…
perso: validating policies/myapp.json
ok: 4 rule(s), 4 tool(s), 4 map entries, 0 warning(s)
perso: policy valid — proceeding to build…
perso: building policy-runtime → wasm32-unknown-unknown
note:  policy JSON is not embedded — pass it to init() at runtime
ok: 1234567 bytes → dist/policy_runtime.wasm
```

Output when policy is invalid (build is skipped):
```
perso: validating policy before build…
perso: validating policies/myapp.json
error: rule references tool 'read_folder' which is not listed in tools[]
```

---

### 3. Run the test suite

```bash
# Native tests (no WASM binary needed)
cargo test -p policy-test

# Full end-to-end WASM boundary tests
PERSO_WASM=dist/policy_runtime.wasm cargo test -p policy-test
```

### 4. Run all tests across the workspace

```bash
cargo test
```

---

## Policy validation in depth

### JSON Schema (`policy.schema.json`)

The bundled schema is compiled into the `policy-compiler` binary at build time via `include_str!`. No external schema file is needed at runtime. It catches:

- Missing required fields (`version`, `default_action`, `tools`, `roles`, `rules`)
- Wrong `default_action` value (must be `"Allow"` or `"Deny"`)
- `version` not matching the `perso-x.y.z` pattern
- Unknown top-level fields
- Duplicate tool or role names
- Tool and role names with invalid characters
- Malformed condition objects (wrong operator names, missing fields)
- `All` / `Any` combinators with fewer than 2 conditions

### Semantic checks (`run_validate`)

Run after the schema passes, these checks require understanding relationships between fields:

- Concrete `tool_name` in a rule must exist in `tools[]` — hard error
- Every role referenced in a rule must exist in `roles[]` — hard error
- Glob patterns that match zero tools in `tools[]` — warning (build proceeds)
- Rules with an empty `roles[]` array — warning (build proceeds)

The schema and semantic checks are complementary. The schema handles structure; the compiler handles meaning. Neither alone is sufficient.

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
let policy_json = std::fs::read_to_string("policies/myapp.json")?;
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

No extra packages needed — Node.js has built-in `WebAssembly` support.

The WASM binary contains **no policy**. You load `policy_runtime.wasm` (the engine) and your policy JSON separately, then pass the policy JSON to `init()` at startup.

```js
const fs = require('fs');

// ── 1. Load the engine and the policy ────────────────────────────────────────
const wasmBytes  = fs.readFileSync('dist/policy_runtime.wasm');
const policyJson = fs.readFileSync('policies/myapp.json', 'utf8');

const { instance } = await WebAssembly.instantiate(wasmBytes);
const { alloc, dealloc, init, evaluate, memory } = instance.exports;

// ── 2. ABI helpers ────────────────────────────────────────────────────────────

// Write a JS string into WASM linear memory; return [ptr, len].
function writeString(str) {
  const bytes = new TextEncoder().encode(str);
  const ptr   = alloc(bytes.length);
  new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

// Read a length-prefixed response buffer, free it, return parsed JSON object.
// Buffer layout: [u32 LE length][...UTF-8 body...]
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
// Call this on every LLM tool call before forwarding to MCP.
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

// Examples:
console.log(checkToolCall('read_file', {}, 'viewer'));
// { decision: 'Allow', reason: "rule matched tool 'read_file' for role 'viewer'; no condition required" }

console.log(checkToolCall('refund_user', { amount: 200 }, 'supervisor', { session_token: 'tok-abc' }));
// { decision: 'Allow', reason: "...condition passed" }

console.log(checkToolCall('refund_user', { amount: 600 }, 'supervisor', { session_token: 'tok-abc' }));
// { decision: 'Deny', reason: "...condition failed" }

console.log(checkToolCall('dangerous_delete', {}, 'viewer'));
// { decision: 'Deny', reason: "no rule matched...applying default_action" }

// ── 5. Hot-reload the policy ──────────────────────────────────────────────────
// Call init() again at any time with new JSON — no restart needed.
const newPolicyJson = fs.readFileSync('policies/updated.json', 'utf8');
const [rPtr, rLen]  = writeString(newPolicyJson);
console.log(readResponse(init(rPtr, rLen)));
// { ok: true }
```

---

## Embedding in a Python host

Install the official Bytecode Alliance wasmtime binding:

```bash
pip install wasmtime
```

Again: the WASM binary has no policy baked in. Load both files separately and pass the policy JSON string to `init()`.

```python
from wasmtime import Engine, Linker, Module, Store
import json, struct

# ── 1. Load the engine and the policy ────────────────────────────────────────
engine   = Engine()
module   = Module.from_file(engine, 'dist/policy_runtime.wasm')
linker   = Linker(engine)
store    = Store(engine)
instance = linker.instantiate(store, module)

with open('policies/myapp.json', 'r') as f:
    policy_json = f.read()

alloc   = instance.exports(store)['alloc']
dealloc = instance.exports(store)['dealloc']
init_fn = instance.exports(store)['init']
eval_fn = instance.exports(store)['evaluate']
memory  = instance.exports(store)['memory']

# ── 2. ABI helpers ────────────────────────────────────────────────────────────

def write_string(s: str):
    """Write a string into WASM linear memory; return (ptr, len)."""
    data = s.encode('utf-8')
    ptr  = alloc(store, len(data))
    memory.write(store, data, ptr)
    return ptr, len(data)

def read_response(ptr: int) -> dict:
    """Read a length-prefixed response buffer, free it, return parsed dict."""
    header = bytes(memory.read(store, ptr, ptr + 4))
    length = struct.unpack_from('<I', header)[0]      # u32 little-endian
    body   = bytes(memory.read(store, ptr + 4, ptr + 4 + length))
    dealloc(store, ptr, 4 + length)
    return json.loads(body)

# ── 3. Initialise the engine with the policy (once at startup) ───────────────
ptr, length = write_string(policy_json)
resp = read_response(init_fn(store, ptr, length))
print(resp)
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

# Examples:
print(check_tool_call('read_file', {}, 'viewer'))
# {'decision': 'Allow', 'reason': "rule matched tool 'read_file' for role 'viewer'; no condition required"}

print(check_tool_call('refund_user', {'amount': 200}, 'supervisor', {'session_token': 'tok-abc'}))
# {'decision': 'Allow', 'reason': "...condition passed"}

print(check_tool_call('refund_user', {'amount': 600}, 'supervisor', {'session_token': 'tok-abc'}))
# {'decision': 'Deny', 'reason': "...condition failed"}

print(check_tool_call('dangerous_delete', {}, 'viewer'))
# {'decision': 'Deny', 'reason': "no rule matched...applying default_action"}

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
{ "decision": "Allow", "reason": "rule matched tool 'read_file' for role 'viewer'; no condition required" }
{ "decision": "Deny",  "reason": "no rule matched tool 'write_file' for role 'viewer'; applying default_action" }
```

---

## Hot-reloading the policy

Call `init` again with new policy JSON at any time. The `Mutex<PolicyState>` inside the WASM module is replaced atomically. No restart required. For multi-threaded hosts, wrap the `Store` in an `Arc<RwLock<>>` so concurrent `evaluate` calls read safely while the reload write is in progress.

---

## Security model

- The LLM never sees or touches the role token. The host extracts the role from its own JWT/session at connection time.
- `default_action: "Deny"` means anything not explicitly allowed is rejected. This is the safe default and the one used in the example policy.
- Conditions are evaluated against three separate JSON bags — arguments, agent attributes, and resource attributes — so the LLM-supplied arguments can never impersonate session data.
- The WASM sandbox means perso has no filesystem, network, or syscall access. It only reads and writes linear memory.
- For zero-trust deployments, embed perso in both the backend (knows the user role) and the MCP server (knows the service identity). Each layer enforces independently.

---

## Example policy walkthrough

```json
{ "tool_name": "refund_user", "roles": ["supervisor"],
  "condition": { "NumericCheck": { "source": "Arguments", "field": "amount", "op": "Lte", "value": 500.0 } } }
```
Supervisors can issue refunds, but only up to $500. A refund of $600 is denied even for a supervisor.

```json
{ "tool_name": "edit_document", "roles": ["admin"],
  "condition": { "Any": [
    { "FieldEquals": { "source_a": "AgentAttributes", "field_a": "user_id",
                       "source_b": "ResourceAttributes", "field_b": "owner_id" } },
    { "StringCheck": { "source": "AgentAttributes", "field": "role", "op": "In", "value": ["admin"] } }
  ]}}
```
Admins can edit a document if they are the owner, or unconditionally if their session role is admin. Either branch passing is enough.

```json
{ "tool_name": "guarded_tool", "roles": ["supervisor"],
  "condition": { "All": [
    { "StringCheck": { "source": "AgentAttributes", "field": "env", "op": "In", "value": ["production"] } },
    { "FieldPresent": { "source": "AgentAttributes", "field": "mfa_verified" } }
  ]}}
```
Supervisors can use this tool only when the environment is production AND MFA has been verified. Both must pass.

```json
{ "tool_name": "glob_tool_*", "roles": ["admin"], "condition": null }
```
Any tool whose name starts with `glob_tool_` is unconditionally available to admins. The glob is expanded at init time against the `tools[]` array — no wildcard pattern matching at evaluate time.

---

## Test coverage summary

| Crate | Tests | What they cover |
|-------|-------|-----------------|
| policy-core | 11 | Parsing every condition type, roundtrip serialisation |
| policy-runtime (expander) | 12 | Glob matching edge cases, expansion correctness |
| policy-runtime (evaluator) | 25 | Every condition type, all logical combinators, default deny |
| policy-runtime (wasm) | 19 | Full WASM ABI: alloc/dealloc, init, evaluate, error paths |
| policy-compiler | 6 | validate happy/sad paths, edge cases |
| policy-test (native) | 19 | All 18 spec cases + map build |
| policy-test (wasm) | 19 | Same 18 spec cases through real WASM boundary |
| **Total** | **111** | |