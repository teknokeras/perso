//! perso — policy-core
//! Shared types, serde parsing, and the policy AST.
//! No evaluation logic lives here — only the data model.

use serde::{Deserialize, Serialize};

// ─── Top-level policy document ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Semver-style schema version, e.g. "perso-1.0.0"
    pub version: String,

    /// What to do when no rule matches (almost always Deny)
    pub default_action: DefaultAction,

    /// Canonical list of all known tool names.
    /// Used as the expansion universe for glob patterns in rules.
    pub tools: Vec<String>,

    /// Ordered list of access rules.
    pub rules: Vec<Rule>,
}

// ─── Default action ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefaultAction {
    Allow,
    Deny,
}

// ─── Rule ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Concrete tool name or glob pattern (e.g. "glob_tool_*").
    /// Globs are expanded at init time against `Policy::tools`.
    pub tool_name: String,

    /// Roles that this rule grants access to.
    pub roles: Vec<String>,

    /// Optional condition tree. `None` means unconditional allow for matched roles.
    pub condition: Option<Condition>,
}

// ─── Condition AST ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Condition {
    /// All child conditions must be true.
    All(Vec<Condition>),

    /// At least one child condition must be true.
    Any(Vec<Condition>),

    /// Negates the inner condition.
    Not(Box<Condition>),

    /// Compare a numeric field to a literal value.
    NumericCheck(NumericCheck),

    /// Compare a string field against a list of values.
    StringCheck(StringCheck),

    /// Assert that a field exists (non-null) in the given source.
    FieldPresent(FieldPresent),

    /// Assert that a field in source A equals a field in source B.
    FieldEquals(FieldEquals),
}

// ─── Condition leaf types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericCheck {
    pub source: Source,
    pub field: String,
    pub op: NumericOp,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringCheck {
    pub source: Source,
    pub field: String,
    pub op: StringOp,
    /// The set of strings to compare against.
    pub value: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPresent {
    pub source: Source,
    pub field: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEquals {
    pub source_a: Source,
    pub field_a: String,
    pub source_b: Source,
    pub field_b: String,
}

// ─── Source enum ─────────────────────────────────────────────────────────────

/// Where to look up a field when evaluating a condition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    /// The JSON arguments passed to the tool call by the LLM.
    Arguments,
    /// Attributes attached to the calling agent / session (e.g. role, user_id).
    AgentAttributes,
    /// Attributes of the resource being acted upon (e.g. owner_id).
    ResourceAttributes,
}

// ─── Operators ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericOp {
    /// Less than or equal
    Lte,
    /// Greater than or equal
    Gte,
    /// Equal
    Eq,
    /// Less than
    Lt,
    /// Greater than
    Gt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringOp {
    /// Field value is in the list
    In,
    /// Field value is NOT in the list
    NotIn,
}

// ─── Evaluation types (request / response) ───────────────────────────────────

/// Passed to `evaluate()` after deserialization from the WASM host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRequest {
    pub tool_name: String,
    /// Raw JSON object — tool arguments from the LLM.
    pub arguments: serde_json::Value,
    /// Raw JSON object — `{ role, agent_attrs, resource_attrs }` built by host.
    pub context: EvaluationContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationContext {
    pub role: String,
    #[serde(default)]
    pub agent_attrs: serde_json::Value,
    #[serde(default)]
    pub resource_attrs: serde_json::Value,
}

/// Returned by `evaluate()`, serialized to JSON across the WASM boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResponse {
    pub decision: Decision,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Allow,
    Deny,
}

// ─── Parse helper ─────────────────────────────────────────────────────────────

/// Deserialize a `Policy` from a JSON string.
pub fn parse_policy(json: &str) -> Result<Policy, serde_json::Error> {
    serde_json::from_str(json)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = include_str!("../../../policies/example.json");

    #[test]
    fn parse_example_policy() {
        let policy = parse_policy(EXAMPLE).expect("example.json must parse");
        assert_eq!(policy.version, "perso-1.0.0");
        assert_eq!(policy.default_action, DefaultAction::Deny);
        assert_eq!(policy.tools.len(), 13);
        assert_eq!(policy.rules.len(), 10);
    }

    #[test]
    fn rule_with_null_condition_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let read_file = policy.rules.iter().find(|r| r.tool_name == "read_file").unwrap();
        assert!(read_file.condition.is_none());
        assert_eq!(read_file.roles, vec!["viewer"]);
    }

    #[test]
    fn numeric_check_condition_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let refund = policy.rules.iter().find(|r| r.tool_name == "refund_user").unwrap();
        match refund.condition.as_ref().unwrap() {
            Condition::NumericCheck(nc) => {
                assert_eq!(nc.field, "amount");
                assert_eq!(nc.op, NumericOp::Lte);
                assert!((nc.value - 500.0).abs() < f64::EPSILON);
                assert_eq!(nc.source, Source::Arguments);
            }
            other => panic!("expected NumericCheck, got {other:?}"),
        }
    }

    #[test]
    fn string_check_not_in_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let rule = policy.rules.iter().find(|r| r.tool_name == "read_restricted").unwrap();
        match rule.condition.as_ref().unwrap() {
            Condition::StringCheck(sc) => {
                assert_eq!(sc.op, StringOp::NotIn);
                assert!(sc.value.contains(&"/etc/passwd".to_string()));
            }
            other => panic!("expected StringCheck, got {other:?}"),
        }
    }

    #[test]
    fn field_present_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let rule = policy.rules.iter().find(|r| r.tool_name == "sensitive_tool").unwrap();
        match rule.condition.as_ref().unwrap() {
            Condition::FieldPresent(fp) => {
                assert_eq!(fp.field, "session_token");
                assert_eq!(fp.source, Source::AgentAttributes);
            }
            other => panic!("expected FieldPresent, got {other:?}"),
        }
    }

    #[test]
    fn field_equals_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let rule = policy.rules.iter().find(|r| r.tool_name == "edit_document").unwrap();
        match rule.condition.as_ref().unwrap() {
            Condition::Any(children) => {
                match &children[0] {
                    Condition::FieldEquals(fe) => {
                        assert_eq!(fe.field_a, "user_id");
                        assert_eq!(fe.field_b, "owner_id");
                    }
                    other => panic!("expected FieldEquals, got {other:?}"),
                }
            }
            other => panic!("expected Any, got {other:?}"),
        }
    }

    #[test]
    fn all_condition_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let rule = policy.rules.iter().find(|r| r.tool_name == "guarded_tool").unwrap();
        match rule.condition.as_ref().unwrap() {
            Condition::All(children) => assert_eq!(children.len(), 2),
            other => panic!("expected All, got {other:?}"),
        }
    }

    #[test]
    fn not_condition_deserializes() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let rule = policy.rules.iter().find(|r| r.tool_name == "open_tool").unwrap();
        match rule.condition.as_ref().unwrap() {
            Condition::Not(_) => {}
            other => panic!("expected Not, got {other:?}"),
        }
    }

    #[test]
    fn glob_rule_present() {
        let policy = parse_policy(EXAMPLE).unwrap();
        let rule = policy.rules.iter().find(|r| r.tool_name == "glob_tool_*").unwrap();
        assert_eq!(rule.roles, vec!["admin"]);
        assert!(rule.condition.is_none());
    }

    #[test]
    fn roundtrip_evaluation_response() {
        let resp = EvaluationResponse {
            decision: Decision::Allow,
            reason: "role matched, no condition".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: EvaluationResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.decision, Decision::Allow);
    }

    #[test]
    fn evaluation_context_default_attrs() {
        let json = r#"{"role":"viewer","agent_attrs":{},"resource_attrs":{}}"#;
        let ctx: EvaluationContext = serde_json::from_str(json).unwrap();
        assert_eq!(ctx.role, "viewer");
    }
}
