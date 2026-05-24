//! Phase 3b — Condition evaluator + top-level evaluate().

use serde_json::Value;

use policy_core::{
    Condition, Decision, DefaultAction, EvaluationContext, EvaluationResponse,
    FieldEquals, FieldPresent, NumericCheck, NumericOp, Source, StringCheck, StringOp,
};

use crate::expander::PolicyMap;

// ─── Evaluation context wrapper ───────────────────────────────────────────────

/// Holds the three JSON bags used during condition evaluation.
/// Mirrors `policy_core::EvaluationContext` but owned and ready for lookup.
pub struct RuntimeContext {
    pub arguments: Value,
    pub agent_attrs: Value,
    pub resource_attrs: Value,
}

impl From<&EvaluationContext> for RuntimeContext {
    fn from(ctx: &EvaluationContext) -> Self {
        Self {
            arguments: ctx.agent_attrs.clone(),   // will be overridden per call
            agent_attrs: ctx.agent_attrs.clone(),
            resource_attrs: ctx.resource_attrs.clone(),
        }
    }
}

// ─── Source resolver ──────────────────────────────────────────────────────────

/// Return the JSON bag that corresponds to `source`.
pub fn resolve_source<'a>(source: &Source, ctx: &'a FullContext<'_>) -> &'a Value {
    match source {
        Source::Arguments => ctx.arguments,
        Source::AgentAttributes => ctx.agent_attrs,
        Source::ResourceAttributes => ctx.resource_attrs,
    }
}

/// Temporary struct that holds all three bags as references for a single
/// evaluate call — avoids cloning Values on the hot path.
pub struct FullContext<'a> {
    pub arguments: &'a Value,
    pub agent_attrs: &'a Value,
    pub resource_attrs: &'a Value,
}

// ─── Condition evaluator ──────────────────────────────────────────────────────

/// Recursively evaluate a `Condition` tree against the provided context.
pub fn evaluate_condition(condition: &Condition, ctx: &FullContext<'_>) -> bool {
    match condition {
        // ── Logical combinators ──────────────────────────────────────────────
        Condition::All(children) => children.iter().all(|c| evaluate_condition(c, ctx)),
        Condition::Any(children) => children.iter().any(|c| evaluate_condition(c, ctx)),
        Condition::Not(inner) => !evaluate_condition(inner, ctx),

        // ── NumericCheck ─────────────────────────────────────────────────────
        Condition::NumericCheck(NumericCheck { source, field, op, value }) => {
            let bag = resolve_source(source, ctx);
            match bag.get(field).and_then(Value::as_f64) {
                None => false, // field missing or not numeric → deny
                Some(actual) => match op {
                    NumericOp::Lte => actual <= *value,
                    NumericOp::Gte => actual >= *value,
                    NumericOp::Eq  => (actual - value).abs() < f64::EPSILON,
                    NumericOp::Lt  => actual <  *value,
                    NumericOp::Gt  => actual >  *value,
                },
            }
        }

        // ── StringCheck ──────────────────────────────────────────────────────
        Condition::StringCheck(StringCheck { source, field, op, value: list }) => {
            let bag = resolve_source(source, ctx);
            match bag.get(field).and_then(Value::as_str) {
                None => false, // field missing or not a string → deny
                Some(actual) => {
                    let in_list = list.iter().any(|v| v == actual);
                    match op {
                        StringOp::In    =>  in_list,
                        StringOp::NotIn => !in_list,
                    }
                }
            }
        }

        // ── FieldPresent ─────────────────────────────────────────────────────
        Condition::FieldPresent(FieldPresent { source, field }) => {
            let bag = resolve_source(source, ctx);
            match bag.get(field) {
                None => false,
                Some(Value::Null) => false, // null counts as absent
                Some(_) => true,
            }
        }

        // ── FieldEquals ──────────────────────────────────────────────────────
        Condition::FieldEquals(FieldEquals { source_a, field_a, source_b, field_b }) => {
            let bag_a = resolve_source(source_a, ctx);
            let bag_b = resolve_source(source_b, ctx);
            match (bag_a.get(field_a), bag_b.get(field_b)) {
                (Some(a), Some(b)) => a == b,
                _ => false, // either field missing → deny
            }
        }
    }
}

// ─── Top-level evaluate ───────────────────────────────────────────────────────

/// Main entry point called for every tool invocation.
///
/// 1. Look up `(tool_name, role)` in the pre-built `PolicyMap`.
/// 2. Miss → apply `default_action`.
/// 3. Hit, no condition → Allow.
/// 4. Hit, has condition → evaluate recursively.
pub fn evaluate(
    tool_name: &str,
    role: &str,
    arguments: &Value,
    agent_attrs: &Value,
    resource_attrs: &Value,
    map: &PolicyMap,
    default_action: &DefaultAction,
) -> EvaluationResponse {
    let key = (tool_name.to_string(), role.to_string());

    match map.get(&key) {
        None => {
            let decision = match default_action {
                DefaultAction::Allow => Decision::Allow,
                DefaultAction::Deny  => Decision::Deny,
            };
            EvaluationResponse {
                decision,
                reason: format!(
                    "no rule matched tool '{}' for role '{}'; applying default_action",
                    tool_name, role
                ),
            }
        }

        Some(None) => EvaluationResponse {
            decision: Decision::Allow,
            reason: format!(
                "rule matched tool '{}' for role '{}'; no condition required",
                tool_name, role
            ),
        },

        Some(Some(condition)) => {
            let ctx = FullContext { arguments, agent_attrs, resource_attrs };
            if evaluate_condition(condition, &ctx) {
                EvaluationResponse {
                    decision: Decision::Allow,
                    reason: format!(
                        "rule matched tool '{}' for role '{}'; condition passed",
                        tool_name, role
                    ),
                }
            } else {
                EvaluationResponse {
                    decision: Decision::Deny,
                    reason: format!(
                        "rule matched tool '{}' for role '{}'; condition failed",
                        tool_name, role
                    ),
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expander::expand_globs;
    use policy_core::{parse_policy, DefaultAction};
    use serde_json::json;

    fn load_map() -> (PolicyMap, DefaultAction) {
        let policy = parse_policy(include_str!("../../../policies/example.json")).unwrap();
        let default = policy.default_action.clone();
        (expand_globs(&policy), default)
    }

    fn empty() -> Value { json!({}) }

    // ── evaluate() integration against example policy ─────────────────────────

    #[test]
    fn role_match_no_condition_allows() {
        let (map, def) = load_map();
        let resp = evaluate("read_file", "viewer", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn role_mismatch_denies_default() {
        let (map, def) = load_map();
        let resp = evaluate("read_file", "admin", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    #[test]
    fn unknown_tool_denies_default() {
        let (map, def) = load_map();
        let resp = evaluate("nonexistent_tool", "admin", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    // ── NumericCheck ──────────────────────────────────────────────────────────

    #[test]
    fn numeric_lte_pass() {
        let (map, def) = load_map();
        let args = json!({ "amount": 200.0 });
        let resp = evaluate("refund_user", "supervisor", &args, &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn numeric_lte_fail() {
        let (map, def) = load_map();
        let args = json!({ "amount": 600.0 });
        let resp = evaluate("refund_user", "supervisor", &args, &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    #[test]
    fn numeric_lte_boundary_exact() {
        let (map, def) = load_map();
        let args = json!({ "amount": 500.0 });
        let resp = evaluate("refund_user", "supervisor", &args, &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn numeric_missing_field_denies() {
        let (map, def) = load_map();
        let resp = evaluate("refund_user", "supervisor", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    // ── StringCheck ───────────────────────────────────────────────────────────

    #[test]
    fn string_not_in_safe_path_allows() {
        let (map, def) = load_map();
        let args = json!({ "path": "/home/user/doc.txt" });
        let resp = evaluate("read_restricted", "viewer", &args, &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn string_not_in_blocked_path_denies() {
        let (map, def) = load_map();
        let args = json!({ "path": "/etc/passwd" });
        let resp = evaluate("read_restricted", "viewer", &args, &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    #[test]
    fn string_in_operator_allows_match() {
        // open_tool has Not(StringCheck In ["blocked_role"]) — so "viewer" (not blocked) → Allow
        let (map, def) = load_map();
        let agent = json!({ "role": "viewer" });
        let resp = evaluate("open_tool", "viewer", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    // ── FieldPresent ──────────────────────────────────────────────────────────

    #[test]
    fn field_present_allows_when_present() {
        let (map, def) = load_map();
        let agent = json!({ "session_token": "abc123" });
        let resp = evaluate("sensitive_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn field_present_denies_when_missing() {
        let (map, def) = load_map();
        let resp = evaluate("sensitive_tool", "supervisor", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    #[test]
    fn field_present_denies_when_null() {
        let (map, def) = load_map();
        let agent = json!({ "session_token": null });
        let resp = evaluate("sensitive_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    // ── FieldEquals ───────────────────────────────────────────────────────────

    #[test]
    fn field_equals_match_allows() {
        let (map, def) = load_map();
        // edit_document: Any[FieldEquals(agent.user_id == resource.owner_id), ...]
        let agent    = json!({ "user_id": "u42" });
        let resource = json!({ "owner_id": "u42" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn field_equals_mismatch_falls_to_second_any_branch() {
        let (map, def) = load_map();
        // user_id != owner_id, but role=="admin" in AgentAttributes → second Any branch passes
        let agent    = json!({ "user_id": "u1", "role": "admin" });
        let resource = json!({ "owner_id": "u2" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn field_equals_both_branches_fail_denies() {
        let (map, def) = load_map();
        let agent    = json!({ "user_id": "u1", "role": "viewer" });
        let resource = json!({ "owner_id": "u2" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    // ── All ───────────────────────────────────────────────────────────────────

    #[test]
    fn all_both_pass_allows() {
        let (map, def) = load_map();
        // guarded_tool: All[env In ["production"], FieldPresent mfa_verified]
        let agent = json!({ "env": "production", "mfa_verified": true });
        let resp = evaluate("guarded_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn all_one_fail_denies() {
        let (map, def) = load_map();
        let agent = json!({ "env": "staging", "mfa_verified": true });
        let resp = evaluate("guarded_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    #[test]
    fn all_both_fail_denies() {
        let (map, def) = load_map();
        let agent = json!({ "env": "staging" });
        let resp = evaluate("guarded_tool", "supervisor", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    // ── Any ───────────────────────────────────────────────────────────────────

    #[test]
    fn any_all_fail_denies() {
        let (map, def) = load_map();
        // edit_document Any: no matching user_id, role is not admin
        let agent    = json!({ "user_id": "u1", "role": "viewer" });
        let resource = json!({ "owner_id": "u99" });
        let resp = evaluate("edit_document", "admin", &empty(), &agent, &resource, &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    // ── Not ───────────────────────────────────────────────────────────────────

    #[test]
    fn not_negation_blocked_role_denies() {
        let (map, def) = load_map();
        // open_tool: Not(role In ["blocked_role"]) — blocked_role → Deny
        let agent = json!({ "role": "blocked_role" });
        let resp = evaluate("open_tool", "viewer", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }

    #[test]
    fn not_negation_normal_role_allows() {
        let (map, def) = load_map();
        let agent = json!({ "role": "viewer" });
        let resp = evaluate("open_tool", "viewer", &empty(), &agent, &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    // ── Glob expansion end-to-end ─────────────────────────────────────────────

    #[test]
    fn glob_tool_alpha_admin_allows() {
        let (map, def) = load_map();
        let resp = evaluate("glob_tool_alpha", "admin", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn glob_tool_beta_admin_allows() {
        let (map, def) = load_map();
        let resp = evaluate("glob_tool_beta", "admin", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Allow);
    }

    #[test]
    fn glob_tool_alpha_viewer_denies() {
        let (map, def) = load_map();
        let resp = evaluate("glob_tool_alpha", "viewer", &empty(), &empty(), &empty(), &map, &def);
        assert_eq!(resp.decision, Decision::Deny);
    }
}
