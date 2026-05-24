//! Phase 4 — WASM bindings.
//!
//! Exports four functions to the host:
//!   alloc(len)                          → *mut u8
//!   dealloc(ptr, len)
//!   init(ptr, len)                      → *mut u8  (JSON result)
//!   evaluate(tool_ptr, tool_len,
//!            args_ptr, args_len,
//!            ctx_ptr,  ctx_len)         → *mut u8  (JSON result)
//!
//! All strings cross the boundary as (ptr, len) pairs pointing into WASM
//! linear memory. Return values are length-prefixed buffers:
//!   [u32 LE length][...UTF-8 bytes...]
//! The host reads the 4-byte length, then reads that many bytes, then
//! calls dealloc(ptr, 4 + length) to free.

use std::sync::OnceLock;

use policy_core::{parse_policy, DefaultAction, EvaluationContext};
use serde_json::json;

use crate::evaluator::evaluate as core_evaluate;
use crate::expander::{expand_globs, PolicyMap};

// ─── Stored policy state ──────────────────────────────────────────────────────

struct PolicyState {
    map: PolicyMap,
    default_action: DefaultAction,
}

/// Global policy state. Set once by `init()`, readable by `evaluate()`.
/// Re-initialisation replaces the inner value via a Mutex so hot-reload works.
static POLICY: OnceLock<std::sync::Mutex<PolicyState>> = OnceLock::new();

// ─── Memory helpers ───────────────────────────────────────────────────────────

/// Allocate `len` bytes on the WASM heap and return a raw pointer.
/// The host calls this before writing input strings into WASM memory.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    // SAFETY: we leak the vec and hand its pointer to the host.
    // The matching `dealloc` will reconstruct + drop it.
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free a buffer previously returned by `alloc` or one of the export fns.
/// `len` must be the original allocation length (including the 4-byte prefix
/// for return buffers).
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    // SAFETY: ptr was allocated by `alloc` with the same len.
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Read a UTF-8 string from WASM linear memory.
///
/// # Safety
/// `ptr` must point to at least `len` valid bytes of UTF-8 within the WASM
/// linear memory for the lifetime of the returned `&str`.
unsafe fn read_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    let bytes = std::slice::from_raw_parts(ptr, len);
    std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8 input: {e}"))
}

/// Encode a JSON string as a length-prefixed heap buffer and return its pointer.
/// Layout: [u32 LE byte-count][...UTF-8 bytes...]
/// The host must call `dealloc(ptr, 4 + byte-count)` after reading.
fn encode_response(json: &str) -> *mut u8 {
    let bytes = json.as_bytes();
    let len = bytes.len();
    let total = 4 + len;

    let mut buf: Vec<u8> = Vec::with_capacity(total);
    // Write length prefix (little-endian u32)
    buf.extend_from_slice(&(len as u32).to_le_bytes());
    buf.extend_from_slice(bytes);

    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Convenience: serialise any serialisable value and encode it.
fn ok_response<T: serde::Serialize>(val: &T) -> *mut u8 {
    encode_response(&serde_json::to_string(val).unwrap_or_else(|e| {
        json!({ "error": format!("serialization failed: {e}") }).to_string()
    }))
}

fn err_response(msg: &str) -> *mut u8 {
    encode_response(&json!({ "error": msg }).to_string())
}

// ─── Exported WASM functions ──────────────────────────────────────────────────

/// Parse and materialise a policy from JSON.
/// Must be called once before `evaluate`. Can be called again to hot-reload.
///
/// Returns: `{"ok": true}` or `{"error": "..."}` as a length-prefixed buffer.
#[no_mangle]
pub extern "C" fn init(ptr: *const u8, len: usize) -> *mut u8 {
    let json_str = match unsafe { read_str(ptr, len) } {
        Ok(s) => s,
        Err(e) => return err_response(&e),
    };

    let policy = match parse_policy(json_str) {
        Ok(p) => p,
        Err(e) => return err_response(&format!("policy parse error: {e}")),
    };

    let map = expand_globs(&policy);
    let state = PolicyState {
        map,
        default_action: policy.default_action,
    };

    // First call: initialise the OnceLock with a Mutex wrapping the state.
    // Subsequent calls: lock the Mutex and replace the state (hot-reload).
    match POLICY.get_or_init(|| std::sync::Mutex::new(state)) {
        mutex => {
            // If OnceLock was already set, replace the inner state.
            if let Ok(mut guard) = mutex.lock() {
                // Re-parse was already done above; overwrite with fresh state.
                let policy2 = match parse_policy(json_str) {
                    Ok(p) => p,
                    Err(e) => return err_response(&format!("policy re-parse error: {e}")),
                };
                guard.map = expand_globs(&policy2);
                guard.default_action = policy2.default_action;
            }
        }
    }

    encode_response(&json!({ "ok": true }).to_string())
}

/// Evaluate a tool call against the loaded policy.
///
/// Inputs (all UTF-8 JSON strings):
///   tool  — plain string, e.g. `"read_file"`
///   args  — JSON object of tool arguments
///   ctx   — JSON object: `{ "role": "...", "agent_attrs": {...}, "resource_attrs": {...} }`
///
/// Returns: `{"decision":"Allow","reason":"..."}` or `{"decision":"Deny","reason":"..."}`
/// as a length-prefixed buffer. Returns `{"error":"..."}` if init was not called or
/// inputs are malformed.
#[no_mangle]
pub extern "C" fn evaluate(
    tool_ptr: *const u8, tool_len: usize,
    args_ptr: *const u8, args_len: usize,
    ctx_ptr:  *const u8, ctx_len:  usize,
) -> *mut u8 {
    // ── Read inputs ──────────────────────────────────────────────────────────
    let tool_name = match unsafe { read_str(tool_ptr, tool_len) } {
        Ok(s) => s,
        Err(e) => return err_response(&format!("tool_name: {e}")),
    };

    let args_str = match unsafe { read_str(args_ptr, args_len) } {
        Ok(s) => s,
        Err(e) => return err_response(&format!("arguments: {e}")),
    };

    let ctx_str = match unsafe { read_str(ctx_ptr, ctx_len) } {
        Ok(s) => s,
        Err(e) => return err_response(&format!("context: {e}")),
    };

    // ── Parse JSON inputs ─────────────────────────────────────────────────────
    let arguments: serde_json::Value = match serde_json::from_str(args_str) {
        Ok(v) => v,
        Err(e) => return err_response(&format!("arguments JSON: {e}")),
    };

    let ctx: EvaluationContext = match serde_json::from_str(ctx_str) {
        Ok(v) => v,
        Err(e) => return err_response(&format!("context JSON: {e}")),
    };

    // ── Access policy state ───────────────────────────────────────────────────
    let policy_mutex = match POLICY.get() {
        Some(m) => m,
        None => return err_response("policy not initialised; call init() first"),
    };

    let guard = match policy_mutex.lock() {
        Ok(g) => g,
        Err(_) => return err_response("policy state mutex poisoned"),
    };

    // ── Evaluate ──────────────────────────────────────────────────────────────
    let response = core_evaluate(
        tool_name,
        &ctx.role,
        &arguments,
        &ctx.agent_attrs,
        &ctx.resource_attrs,
        &guard.map,
        &guard.default_action,
    );

    ok_response(&response)
}

// ─── Tests (host-side, no WASM runtime needed) ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const POLICY_JSON: &str = include_str!("../../../policies/example.json");

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Write a string into a Vec, call f with (ptr, len), return result.
    fn with_str<F: FnOnce(*const u8, usize) -> *mut u8>(s: &str, f: F) -> String {
        let bytes = s.as_bytes();
        let result_ptr = f(bytes.as_ptr(), bytes.len());
        read_response(result_ptr)
    }

    /// Read a length-prefixed response buffer back into a String, then dealloc.
    fn read_response(ptr: *mut u8) -> String {
        unsafe {
            let len_bytes = std::slice::from_raw_parts(ptr, 4);
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            let body = std::slice::from_raw_parts(ptr.add(4), len);
            let s = std::str::from_utf8(body).unwrap().to_string();
            dealloc(ptr, 4 + len);
            s
        }
    }

    fn call_init(policy: &str) -> Value {
        let json_str = with_str(policy, |p, l| init(p, l));
        serde_json::from_str(&json_str).unwrap()
    }

    fn call_evaluate(tool: &str, args: &str, ctx: &str) -> Value {
        let tool_b = tool.as_bytes();
        let args_b = args.as_bytes();
        let ctx_b  = ctx.as_bytes();
        let ptr = evaluate(
            tool_b.as_ptr(), tool_b.len(),
            args_b.as_ptr(), args_b.len(),
            ctx_b.as_ptr(),  ctx_b.len(),
        );
        let json_str = read_response(ptr);
        serde_json::from_str(&json_str).unwrap()
    }

    fn ctx(role: &str) -> String {
        json!({ "role": role, "agent_attrs": {}, "resource_attrs": {} }).to_string()
    }

    fn ctx_with_agent(role: &str, agent: serde_json::Value) -> String {
        json!({ "role": role, "agent_attrs": agent, "resource_attrs": {} }).to_string()
    }

    fn ctx_with_resource(role: &str, agent: serde_json::Value, resource: serde_json::Value) -> String {
        json!({ "role": role, "agent_attrs": agent, "resource_attrs": resource }).to_string()
    }

    // ── alloc / dealloc roundtrip ─────────────────────────────────────────────

    #[test]
    fn alloc_dealloc_roundtrip() {
        let ptr = alloc(64);
        assert!(!ptr.is_null());
        dealloc(ptr, 64);
    }

    // ── init ──────────────────────────────────────────────────────────────────

    #[test]
    fn init_valid_policy_returns_ok() {
        let resp = call_init(POLICY_JSON);
        assert_eq!(resp["ok"], true);
    }

    #[test]
    fn init_invalid_json_returns_error() {
        let resp = call_init("not json at all {{");
        assert!(resp["error"].is_string());
    }

    #[test]
    fn init_hot_reload_second_call_ok() {
        call_init(POLICY_JSON);
        let resp = call_init(POLICY_JSON);
        assert_eq!(resp["ok"], true);
    }

    // ── evaluate — basic allow / deny ─────────────────────────────────────────

    #[test]
    fn evaluate_role_match_no_condition_allows() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("read_file", "{}", &ctx("viewer"));
        assert_eq!(resp["decision"], "Allow");
    }

    #[test]
    fn evaluate_role_mismatch_denies() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("read_file", "{}", &ctx("admin"));
        assert_eq!(resp["decision"], "Deny");
    }

    #[test]
    fn evaluate_unknown_tool_denies() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("no_such_tool", "{}", &ctx("admin"));
        assert_eq!(resp["decision"], "Deny");
    }

    // ── evaluate — NumericCheck ───────────────────────────────────────────────

    #[test]
    fn evaluate_numeric_lte_pass() {
        call_init(POLICY_JSON);
        let args = json!({ "amount": 200 }).to_string();
        let resp = call_evaluate("refund_user", &args, &ctx("supervisor"));
        assert_eq!(resp["decision"], "Allow");
    }

    #[test]
    fn evaluate_numeric_lte_fail() {
        call_init(POLICY_JSON);
        let args = json!({ "amount": 600 }).to_string();
        let resp = call_evaluate("refund_user", &args, &ctx("supervisor"));
        assert_eq!(resp["decision"], "Deny");
    }

    // ── evaluate — StringCheck ────────────────────────────────────────────────

    #[test]
    fn evaluate_string_not_in_safe_allows() {
        call_init(POLICY_JSON);
        let args = json!({ "path": "/home/user/file.txt" }).to_string();
        let resp = call_evaluate("read_restricted", &args, &ctx("viewer"));
        assert_eq!(resp["decision"], "Allow");
    }

    #[test]
    fn evaluate_string_not_in_blocked_denies() {
        call_init(POLICY_JSON);
        let args = json!({ "path": "/etc/shadow" }).to_string();
        let resp = call_evaluate("read_restricted", &args, &ctx("viewer"));
        assert_eq!(resp["decision"], "Deny");
    }

    // ── evaluate — FieldPresent ───────────────────────────────────────────────

    #[test]
    fn evaluate_field_present_allows() {
        call_init(POLICY_JSON);
        let agent = json!({ "session_token": "tok123" });
        let resp = call_evaluate("sensitive_tool", "{}", &ctx_with_agent("supervisor", agent));
        assert_eq!(resp["decision"], "Allow");
    }

    #[test]
    fn evaluate_field_present_missing_denies() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("sensitive_tool", "{}", &ctx("supervisor"));
        assert_eq!(resp["decision"], "Deny");
    }

    // ── evaluate — glob expansion ─────────────────────────────────────────────

    #[test]
    fn evaluate_glob_tool_alpha_admin_allows() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("glob_tool_alpha", "{}", &ctx("admin"));
        assert_eq!(resp["decision"], "Allow");
    }

    #[test]
    fn evaluate_glob_tool_beta_admin_allows() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("glob_tool_beta", "{}", &ctx("admin"));
        assert_eq!(resp["decision"], "Allow");
    }

    #[test]
    fn evaluate_glob_tool_wrong_role_denies() {
        call_init(POLICY_JSON);
        let resp = call_evaluate("glob_tool_alpha", "{}", &ctx("viewer"));
        assert_eq!(resp["decision"], "Deny");
    }

    // ── evaluate — FieldEquals via Any ────────────────────────────────────────

    #[test]
    fn evaluate_field_equals_owner_match_allows() {
        call_init(POLICY_JSON);
        let agent    = json!({ "user_id": "u7" });
        let resource = json!({ "owner_id": "u7" });
        let c = ctx_with_resource("admin", agent, resource);
        let resp = call_evaluate("edit_document", "{}", &c);
        assert_eq!(resp["decision"], "Allow");
    }

    // ── evaluate — error paths ────────────────────────────────────────────────

    #[test]
    fn evaluate_bad_args_json_returns_error() {
        call_init(POLICY_JSON);
        let tool_b = b"read_file";
        let bad_args = b"not-json{{";
        let ctx_b = ctx("viewer").into_bytes();
        let ptr = evaluate(
            tool_b.as_ptr(), tool_b.len(),
            bad_args.as_ptr(), bad_args.len(),
            ctx_b.as_ptr(), ctx_b.len(),
        );
        let json_str = read_response(ptr);
        let v: Value = serde_json::from_str(&json_str).unwrap();
        assert!(v["error"].is_string());
    }

    #[test]
    fn evaluate_bad_context_json_returns_error() {
        call_init(POLICY_JSON);
        let tool_b = b"read_file";
        let args_b = b"{}";
        let bad_ctx = b"not-json{{";
        let ptr = evaluate(
            tool_b.as_ptr(), tool_b.len(),
            args_b.as_ptr(), args_b.len(),
            bad_ctx.as_ptr(), bad_ctx.len(),
        );
        let json_str = read_response(ptr);
        let v: Value = serde_json::from_str(&json_str).unwrap();
        assert!(v["error"].is_string());
    }
}
