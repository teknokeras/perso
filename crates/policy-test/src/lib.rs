// perso - policy-test  Phase 6: Integration test suite.
//
// Two test layers:
//
// 1. native_tests  - calls policy-runtime's public Rust API directly.
//    Covers all 18 spec cases. Runs with: cargo test -p policy-test
//
// 2. wasm_tests    - loads the compiled policy_runtime.wasm via wasmtime
//    and drives every spec case through the real WASM ABI.
//    Enable by setting the PERSO_WASM env var:
//
//      PERSO_WASM=dist/policy_runtime.wasm cargo test -p policy-test
//
//    Build the artifact first with:
//      cargo run -p policy-compiler -- build \
//          --policy policies/example.json \
//          --output dist/policy_runtime.wasm

// ─── Shared helpers ───────────────────────────────────────────────────────────

pub mod helpers {
    use policy_core::{DefaultAction, Decision, EvaluationResponse};
    use policy_runtime::expander::expand_globs;
    use policy_runtime::expander::PolicyMap;
    use serde_json::{json, Value};

    pub const POLICY_JSON: &str = include_str!("../../../policies/example.json");

    pub fn load_map() -> (PolicyMap, DefaultAction) {
        let policy = policy_core::parse_policy(POLICY_JSON).expect("example.json must parse");
        let default = policy.default_action.clone();
        (expand_globs(&policy), default)
    }

    pub fn empty() -> Value { json!({}) }

    pub fn is_allow(resp: &EvaluationResponse) -> bool { resp.decision == Decision::Allow }
    pub fn is_deny(resp: &EvaluationResponse)  -> bool { resp.decision == Decision::Deny  }
}

// ─── Native integration tests ─────────────────────────────────────────────────

#[cfg(test)]
mod native_tests {
    use super::helpers::*;
    use policy_runtime::evaluator::evaluate;
    use serde_json::json;

    // 1. Load policy.json, init — no error
    #[test]
    fn spec_01_policy_parses_and_map_builds() {
        let (map, _) = load_map();
        assert!(!map.is_empty(), "PolicyMap must not be empty after init");
    }

    // 2. Role match, no condition → Allow
    #[test]
    fn spec_02_role_match_no_condition_allows() {
        let (map, def) = load_map();
        let resp = evaluate("read_file", "viewer", &empty(), &empty(), &empty(), &map, &def);
        assert!(is_allow(&resp), "read_file+viewer should Allow, got: {:?}", resp);
    }

    // 3. Role mismatch → Deny (default_action)
    #[test]
    fn spec_03_role_mismatch_denies() {
        let (map, def) = load_map();
        let resp = evaluate("read_file", "admin", &empty(), &empty(), &empty(), &map, &def);
        assert!(is_deny(&resp), "read_file+admin should Deny, got: {:?}", resp);
    }

    // 4. NumericCheck: amount=200, Lte 500 → Allow
    #[test]
    fn spec_04_numeric_check_pass_allows() {
        let (map, def) = load_map();
        let args = json!({ "amount": 200.0 });
        let resp = evaluate("refund_user", "supervisor", &args, &empty(), &empty(), &map, &def);
        assert!(is_allow(&resp), "refund_user amount=200 should Allow, got: {:?}", resp);
    }

    // 5. NumericCheck: amount=600, Lte 500 → Deny
    #[test]
    fn spec_05_numeric_check_fail_denies() {
        let (map, def) = load_map();
        let args = json!({ "amount": 600.0 });
        let resp = evaluate("refund_user", "supervisor", &args, &empty(), &empty(), &map, &def);
        assert!(is_deny(&resp), "refund_user amount=600 should Deny, got: {:?}", resp);
    }

    // 6. StringCheck NotIn: safe path → Allow
    #[test]
    fn spec_06_string_not_in_safe_path_allows() {
        let (map, def) = load_map();
        let args = json!({ "path": "/home/user/report.txt" });
        let resp = evaluate("read_restricted", "viewer", &args, &empty(), &empty(), &map, &def);
        assert!(is_allow(&resp), "read_restricted safe path should Allow, got: {:?}", resp);
    }

    // 7. StringCheck NotIn: /etc/passwd → Deny
    #[test]
    fn spec_07_string_not_in_blocked_path_denies() {
        let (map, def) = load_map();
        let args = json!({ "path": "/etc/passwd" });
        let resp = evaluate("read_restricted", "viewer", &args, &empty(), &empty(), &map, &def);
        assert!(is_deny(&resp), "read_restricted /etc/passwd should Deny, got: {:?}", resp);
    }

    // 8. FieldPresent: field present → Allow
    #[test]
    fn spec_08_field_present_allows() {
        let (map, def) = load_map();
        let agent = json!({ "session_token": "tok-abc" });
        let resp = evaluate("sensitive_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert!(is_allow(&resp), "sensitive_tool with session_token should Allow, got: {:?}", resp);
    }

    // 9. FieldPresent: field missing → Deny
    #[test]
    fn spec_09_field_present_missing_denies() {
        let (map, def) = load_map();
        let resp = evaluate("sensitive_tool", "supervisor", &empty(), &empty(), &empty(), &map, &def);
        assert!(is_deny(&resp), "sensitive_tool without session_token should Deny, got: {:?}", resp);
    }

    // 10. FieldEquals: values match → Allow
    #[test]
    fn spec_10_field_equals_match_allows() {
        let (map, def) = load_map();
        let agent    = json!({ "user_id": "u99" });
        let resource = json!({ "owner_id": "u99" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert!(is_allow(&resp), "edit_document user==owner should Allow, got: {:?}", resp);
    }

    // 11. FieldEquals: values mismatch, second Any branch also fails → Deny
    #[test]
    fn spec_11_field_equals_mismatch_denies() {
        let (map, def) = load_map();
        let agent    = json!({ "user_id": "u1", "role": "viewer" });
        let resource = json!({ "owner_id": "u2" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert!(is_deny(&resp), "edit_document mismatched fields should Deny, got: {:?}", resp);
    }

    // 12. All: all conditions pass → Allow
    #[test]
    fn spec_12_all_conditions_pass_allows() {
        let (map, def) = load_map();
        let agent = json!({ "env": "production", "mfa_verified": true });
        let resp = evaluate("guarded_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert!(is_allow(&resp), "guarded_tool all-pass should Allow, got: {:?}", resp);
    }

    // 13. All: one condition fails → Deny
    #[test]
    fn spec_13_all_one_fail_denies() {
        let (map, def) = load_map();
        let agent = json!({ "env": "staging", "mfa_verified": true });
        let resp = evaluate("guarded_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert!(is_deny(&resp), "guarded_tool env=staging should Deny, got: {:?}", resp);
    }

    // 14. Any: one condition passes → Allow
    #[test]
    fn spec_14_any_one_pass_allows() {
        let (map, def) = load_map();
        let agent    = json!({ "user_id": "u1", "role": "admin" });
        let resource = json!({ "owner_id": "u99" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert!(is_allow(&resp), "edit_document any-one-pass should Allow, got: {:?}", resp);
    }

    // 15. Any: all conditions fail → Deny
    #[test]
    fn spec_15_any_all_fail_denies() {
        let (map, def) = load_map();
        let agent    = json!({ "user_id": "u1", "role": "viewer" });
        let resource = json!({ "owner_id": "u99" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert!(is_deny(&resp), "edit_document all-any-fail should Deny, got: {:?}", resp);
    }

    // 16a. Not: blocked_role → Deny
    #[test]
    fn spec_16a_not_blocked_role_denies() {
        let (map, def) = load_map();
        let agent = json!({ "role": "blocked_role" });
        let resp = evaluate("open_tool", "viewer", &empty(), &agent, &empty(), &map, &def);
        assert!(is_deny(&resp), "open_tool with blocked_role should Deny, got: {:?}", resp);
    }

    // 16b. Not: normal role → Allow
    #[test]
    fn spec_16b_not_normal_role_allows() {
        let (map, def) = load_map();
        let agent = json!({ "role": "viewer" });
        let resp = evaluate("open_tool", "viewer", &empty(), &agent, &empty(), &map, &def);
        assert!(is_allow(&resp), "open_tool with viewer should Allow, got: {:?}", resp);
    }

    // 17. Glob expansion: glob_tool_alpha + glob_tool_beta with admin → Allow
    #[test]
    fn spec_17_glob_expansion_allows() {
        let (map, def) = load_map();
        let ra = evaluate("glob_tool_alpha", "admin", &empty(), &empty(), &empty(), &map, &def);
        let rb = evaluate("glob_tool_beta",  "admin", &empty(), &empty(), &empty(), &map, &def);
        assert!(is_allow(&ra), "glob_tool_alpha+admin should Allow, got: {:?}", ra);
        assert!(is_allow(&rb), "glob_tool_beta+admin should Allow, got: {:?}", rb);
    }

    // 18. Unknown tool → Deny (default_action)
    #[test]
    fn spec_18_unknown_tool_denies() {
        let (map, def) = load_map();
        let resp = evaluate("totally_unknown_tool", "admin", &empty(), &empty(), &empty(), &map, &def);
        assert!(is_deny(&resp), "unknown tool should Deny via default_action, got: {:?}", resp);
    }
}

// ─── WASM boundary integration tests ─────────────────────────────────────────

#[cfg(test)]
mod wasm_tests {
    // Loads the compiled policy_runtime.wasm via wasmtime and drives every
    // spec case through the real WASM ABI (alloc / dealloc / init / evaluate).
    //
    // Set PERSO_WASM=<path-to-wasm> to enable. Skipped silently otherwise.
    //
    // Build the WASM first:
    //   cargo run -p policy-compiler -- build \
    //       --policy policies/example.json \
    //       --output dist/policy_runtime.wasm
    //
    // Then run:
    //   PERSO_WASM=dist/policy_runtime.wasm cargo test -p policy-test

    use std::path::PathBuf;
    use anyhow::Result;
    use wasmtime::{Engine, Linker, Module, Store};
    use serde_json::{json, Value};

    use super::helpers::POLICY_JSON;

    // ── WASM harness ──────────────────────────────────────────────────────────

    struct PersoWasm {
        store:    Store<()>,
        instance: wasmtime::Instance,
    }

    impl PersoWasm {
        fn load(wasm_path: &PathBuf) -> Result<Self> {
            let engine   = Engine::default();
            let module   = Module::from_file(&engine, wasm_path)?;
            let linker   = Linker::new(&engine);
            let mut store = Store::new(&engine, ());
            let instance  = linker.instantiate(&mut store, &module)?;
            Ok(Self { store, instance })
        }

        // Write bytes into WASM linear memory via alloc; return (ptr, len).
        fn write_bytes(&mut self, data: &[u8]) -> Result<(i32, i32)> {
            let alloc = self.instance
                .get_typed_func::<i32, i32>(&mut self.store, "alloc")?;
            let len = data.len() as i32;
            let ptr = alloc.call(&mut self.store, len)?;
            let memory = self.instance
                .get_memory(&mut self.store, "memory")
                .ok_or_else(|| anyhow::anyhow!("no 'memory' export"))?;
            memory.write(&mut self.store, ptr as usize, data)?;
            Ok((ptr, len))
        }

        // Read a length-prefixed response buffer, dealloc it, return JSON string.
        fn read_response(&mut self, ptr: i32) -> Result<String> {
            let memory = self.instance
                .get_memory(&mut self.store, "memory")
                .ok_or_else(|| anyhow::anyhow!("no 'memory' export"))?;
            let mut len_buf = [0u8; 4];
            memory.read(&self.store, ptr as usize, &mut len_buf)?;
            let body_len = u32::from_le_bytes(len_buf) as usize;
            let mut body = vec![0u8; body_len];
            memory.read(&self.store, ptr as usize + 4, &mut body)?;
            let json_str = String::from_utf8(body)?;
            let dealloc = self.instance
                .get_typed_func::<(i32, i32), ()>(&mut self.store, "dealloc")?;
            dealloc.call(&mut self.store, (ptr, (4 + body_len) as i32))?;
            Ok(json_str)
        }

        // Call init(policy_json) and return the parsed JSON response.
        fn init(&mut self, policy_json: &str) -> Result<Value> {
            let (ptr, len) = self.write_bytes(policy_json.as_bytes())?;
            let init_fn = self.instance
                .get_typed_func::<(i32, i32), i32>(&mut self.store, "init")?;
            let result_ptr = init_fn.call(&mut self.store, (ptr, len))?;
            let json_str = self.read_response(result_ptr)?;
            Ok(serde_json::from_str(&json_str)?)
        }

        // Call evaluate(tool, args_json, ctx_json) and return the parsed JSON response.
        fn evaluate(&mut self, tool: &str, args: &str, ctx: &str) -> Result<Value> {
            let (tp, tl) = self.write_bytes(tool.as_bytes())?;
            let (ap, al) = self.write_bytes(args.as_bytes())?;
            let (cp, cl) = self.write_bytes(ctx.as_bytes())?;
            let eval_fn = self.instance
                .get_typed_func::<(i32,i32,i32,i32,i32,i32), i32>(
                    &mut self.store, "evaluate")?;
            let result_ptr = eval_fn.call(&mut self.store, (tp,tl,ap,al,cp,cl))?;
            let json_str = self.read_response(result_ptr)?;
            Ok(serde_json::from_str(&json_str)?)
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn wasm_path() -> Option<PathBuf> {
        std::env::var("PERSO_WASM").ok().map(PathBuf::from)
    }

    macro_rules! wasm_test {
        ($name:ident, $body:expr) => {
            #[test]
            fn $name() {
                let Some(path) = wasm_path() else {
                    eprintln!("SKIP {}: set PERSO_WASM=<path> to enable", stringify!($name));
                    return;
                };
                let mut w = PersoWasm::load(&path)
                    .expect("failed to load WASM module");
                let init_resp = w.init(POLICY_JSON).expect("init failed");
                assert_eq!(init_resp["ok"], true, "init must return ok:true");
                #[allow(clippy::redundant_closure_call)]
                ($body)(&mut w);
            }
        };
    }

    fn ctx(role: &str) -> String {
        json!({ "role": role, "agent_attrs": {}, "resource_attrs": {} }).to_string()
    }
    fn ctx_agent(role: &str, agent: Value) -> String {
        json!({ "role": role, "agent_attrs": agent, "resource_attrs": {} }).to_string()
    }
    fn ctx_full(role: &str, agent: Value, resource: Value) -> String {
        json!({ "role": role, "agent_attrs": agent, "resource_attrs": resource }).to_string()
    }

    // ── All 18 spec tests through the real WASM boundary ─────────────────────

    wasm_test!(wasm_01_init_succeeds, |_w: &mut PersoWasm| {
        // init is called and asserted in the macro itself
    });

    wasm_test!(wasm_02_role_match_no_condition_allows, |w: &mut PersoWasm| {
        let r = w.evaluate("read_file", "{}", &ctx("viewer")).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_03_role_mismatch_denies, |w: &mut PersoWasm| {
        let r = w.evaluate("read_file", "{}", &ctx("admin")).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_04_numeric_check_pass_allows, |w: &mut PersoWasm| {
        let args = json!({ "amount": 200 }).to_string();
        let r = w.evaluate("refund_user", &args, &ctx("supervisor")).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_05_numeric_check_fail_denies, |w: &mut PersoWasm| {
        let args = json!({ "amount": 600 }).to_string();
        let r = w.evaluate("refund_user", &args, &ctx("supervisor")).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_06_string_not_in_safe_allows, |w: &mut PersoWasm| {
        let args = json!({ "path": "/home/user/doc.txt" }).to_string();
        let r = w.evaluate("read_restricted", &args, &ctx("viewer")).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_07_string_not_in_blocked_denies, |w: &mut PersoWasm| {
        let args = json!({ "path": "/etc/passwd" }).to_string();
        let r = w.evaluate("read_restricted", &args, &ctx("viewer")).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_08_field_present_allows, |w: &mut PersoWasm| {
        let c = ctx_agent("supervisor", json!({ "session_token": "tok123" }));
        let r = w.evaluate("sensitive_tool", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_09_field_present_missing_denies, |w: &mut PersoWasm| {
        let r = w.evaluate("sensitive_tool", "{}", &ctx("supervisor")).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_10_field_equals_match_allows, |w: &mut PersoWasm| {
        let c = ctx_full("admin",
            json!({ "user_id": "u42" }),
            json!({ "owner_id": "u42" }));
        let r = w.evaluate("edit_document", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_11_field_equals_mismatch_denies, |w: &mut PersoWasm| {
        let c = ctx_full("admin",
            json!({ "user_id": "u1", "role": "viewer" }),
            json!({ "owner_id": "u99" }));
        let r = w.evaluate("edit_document", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_12_all_pass_allows, |w: &mut PersoWasm| {
        let c = ctx_agent("supervisor", json!({ "env": "production", "mfa_verified": true }));
        let r = w.evaluate("guarded_tool", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_13_all_one_fail_denies, |w: &mut PersoWasm| {
        let c = ctx_agent("supervisor", json!({ "env": "staging", "mfa_verified": true }));
        let r = w.evaluate("guarded_tool", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_14_any_one_pass_allows, |w: &mut PersoWasm| {
        let c = ctx_full("admin",
            json!({ "user_id": "u1", "role": "admin" }),
            json!({ "owner_id": "u99" }));
        let r = w.evaluate("edit_document", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_15_any_all_fail_denies, |w: &mut PersoWasm| {
        let c = ctx_full("admin",
            json!({ "user_id": "u1", "role": "viewer" }),
            json!({ "owner_id": "u99" }));
        let r = w.evaluate("edit_document", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_16a_not_blocked_role_denies, |w: &mut PersoWasm| {
        let c = ctx_agent("viewer", json!({ "role": "blocked_role" }));
        let r = w.evaluate("open_tool", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Deny");
    });

    wasm_test!(wasm_16b_not_normal_role_allows, |w: &mut PersoWasm| {
        let c = ctx_agent("viewer", json!({ "role": "viewer" }));
        let r = w.evaluate("open_tool", "{}", &c).unwrap();
        assert_eq!(r["decision"], "Allow");
    });

    wasm_test!(wasm_17_glob_expansion_allows, |w: &mut PersoWasm| {
        let ra = w.evaluate("glob_tool_alpha", "{}", &ctx("admin")).unwrap();
        let rb = w.evaluate("glob_tool_beta",  "{}", &ctx("admin")).unwrap();
        assert_eq!(ra["decision"], "Allow", "glob_tool_alpha");
        assert_eq!(rb["decision"], "Allow", "glob_tool_beta");
    });

    wasm_test!(wasm_18_unknown_tool_denies, |w: &mut PersoWasm| {
        let r = w.evaluate("totally_unknown_tool", "{}", &ctx("admin")).unwrap();
        assert_eq!(r["decision"], "Deny");
    });
}
