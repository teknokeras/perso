//! Phase 3a — Glob expander + HashMap builder.
//!
//! After `init()` there are zero wildcards in memory.
//! Every pattern in `rules[].tool_name` is resolved against `policy.tools[]`
//! and expanded into one concrete (tool, role) entry per match.

use std::collections::HashMap;

use policy_core::{Condition, Policy};

/// Key: (concrete_tool_name, role)
/// Value: optional condition tree (None = unconditional allow for that pair)
pub type PolicyMap = HashMap<(String, String), Option<Condition>>;

/// Expand every rule's tool_name glob against `policy.tools`, then build
/// a flat `PolicyMap` ready for O(1) lookup at evaluate time.
///
/// Glob syntax: only `*` is supported, matching any sequence of characters.
/// The `*` may appear anywhere in the pattern (prefix, suffix, middle, or alone).
///
/// Concrete (non-glob) tool names that are not present in `policy.tools` are
/// still inserted — the tools list is the expansion universe for globs, not a
/// whitelist for concrete names.
pub fn expand_globs(policy: &Policy) -> PolicyMap {
    let mut map: PolicyMap = HashMap::new();

    for rule in &policy.rules {
        let matched_tools: Vec<&str> = if is_glob(&rule.tool_name) {
            policy
                .tools
                .iter()
                .filter(|t| glob_matches(&rule.tool_name, t))
                .map(String::as_str)
                .collect()
        } else {
            // Concrete name — use as-is regardless of tools list
            vec![rule.tool_name.as_str()]
        };

        for tool in matched_tools {
            for role in &rule.roles {
                // ✅ NEW: skip roles not declared in policy.roles[]
                if !policy.roles.contains(role) {
                    continue;
                }
                // Later rules win if the same (tool, role) pair appears twice.
                // For this design, first-match is fine since the spec doesn't
                // define precedence; we use insert (last-write wins).
                map.insert((tool.to_string(), role.clone()), rule.condition.clone());
            }
        }
    }

    map
}

/// Returns true if `pattern` contains at least one `*` wildcard.
fn is_glob(pattern: &str) -> bool {
    pattern.contains('*')
}

/// Simple glob matcher — only `*` wildcard supported.
///
/// Splits the pattern on `*` into segments. The first segment is anchored to
/// the start of `text`, the last to the end, and middle segments are found
/// left-to-right. An empty segment (from adjacent `*`s or leading/trailing
/// `*`) is skipped. `*` always matches zero or more characters.
pub fn glob_matches(pattern: &str, text: &str) -> bool {
    // Fast path: no wildcard → exact match
    if !pattern.contains('*') {
        return pattern == text;
    }

    let parts: Vec<&str> = pattern.split('*').collect();
    let mut remaining = text;

    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 {
            // First segment must match the start of the remaining text
            if !remaining.starts_with(part) {
                return false;
            }
            remaining = &remaining[part.len()..];
        } else if i == parts.len() - 1 {
            // Last segment must match the end — but only consume from what's left,
            // so `*` between two segments can match the empty string.
            if remaining.len() < part.len() {
                return false;
            }
            if !remaining.ends_with(part) {
                return false;
            }
            // Consume the suffix so nothing is "left over"
            remaining = &remaining[..remaining.len() - part.len()];
        } else {
            // Middle segment — find leftmost occurrence in remaining
            match remaining.find(part) {
                Some(pos) => remaining = &remaining[pos + part.len()..],
                None => return false,
            }
        }
    }

    true
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use policy_core::{DefaultAction, Rule};

    fn make_policy(tools: Vec<&str>, roles: Vec<&str>, rules: Vec<Rule>) -> Policy {
        Policy {
            version: "perso-1.0.0".into(),
            default_action: DefaultAction::Deny,
            tools: tools.into_iter().map(str::to_string).collect(),
            roles: roles.into_iter().map(str::to_string).collect(),
            rules,
        }
    }

    fn simple_rule(tool_name: &str, roles: Vec<&str>) -> Rule {
        Rule {
            tool_name: tool_name.into(),
            roles: roles.into_iter().map(str::to_string).collect(),
            condition: None,
        }
    }

    // ── glob_matches unit tests ───────────────────────────────────────────────

    #[test]
    fn glob_exact_match() {
        assert!(glob_matches("read_file", "read_file"));
        assert!(!glob_matches("read_file", "write_file"));
    }

    #[test]
    fn glob_suffix_wildcard() {
        assert!(glob_matches("glob_tool_*", "glob_tool_alpha"));
        assert!(glob_matches("glob_tool_*", "glob_tool_beta"));
        assert!(!glob_matches("glob_tool_*", "other_tool"));
    }

    #[test]
    fn glob_prefix_wildcard() {
        assert!(glob_matches("*_tool", "my_tool"));
        assert!(glob_matches("*_tool", "other_tool"));
        assert!(!glob_matches("*_tool", "my_toolbox"));
    }

    #[test]
    fn glob_star_only_matches_everything() {
        assert!(glob_matches("*", "anything"));
        assert!(glob_matches("*", ""));
        assert!(glob_matches("*", "glob_tool_alpha"));
    }

    #[test]
    fn glob_middle_wildcard() {
        assert!(glob_matches("read_*_file", "read_big_file"));
        assert!(glob_matches("read_*_file", "read_very_large_file"));
        // "read_file" does NOT match "read_*_file": the literal "_" after the
        // wildcard must still appear — * matches zero or more chars but cannot
        // absorb surrounding literals.
        assert!(!glob_matches("read_*_file", "read_file"));
        assert!(!glob_matches("read_*_file", "write_big_file"));
    }

    #[test]
    fn glob_multiple_wildcards() {
        assert!(glob_matches("a_*_b_*_c", "a_X_b_Y_c"));
        assert!(!glob_matches("a_*_b_*_c", "a_X_b_Y_d"));
    }

    // ── expand_globs unit tests ───────────────────────────────────────────────

    #[test]
    fn expand_concrete_rule() {
        let policy = make_policy(
            vec!["read_file", "write_file"],
            vec!["viewer"],
            vec![simple_rule("read_file", vec!["viewer"])],
        );
        let map = expand_globs(&policy);
        assert!(map.contains_key(&("read_file".into(), "viewer".into())));
        assert!(!map.contains_key(&("write_file".into(), "viewer".into())));
    }

    #[test]
    fn expand_glob_rule_hits_two_tools() {
        let policy = make_policy(
            vec!["glob_tool_alpha", "glob_tool_beta", "other_tool"],
            vec!["admin"],
            vec![simple_rule("glob_tool_*", vec!["admin"])],
        );
        let map = expand_globs(&policy);
        assert!(map.contains_key(&("glob_tool_alpha".into(), "admin".into())));
        assert!(map.contains_key(&("glob_tool_beta".into(), "admin".into())));
        assert!(!map.contains_key(&("other_tool".into(), "admin".into())));
    }

    #[test]
    fn expand_glob_multiple_roles() {
        let policy = make_policy(
            vec!["tool_a", "tool_b"],
            vec!["viewer", "admin"],
            vec![simple_rule("tool_*", vec!["viewer", "admin"])],
        );
        let map = expand_globs(&policy);
        assert!(map.contains_key(&("tool_a".into(), "viewer".into())));
        assert!(map.contains_key(&("tool_a".into(), "admin".into())));
        assert!(map.contains_key(&("tool_b".into(), "viewer".into())));
        assert!(map.contains_key(&("tool_b".into(), "admin".into())));
    }

    #[test]
    fn expand_no_glob_match_produces_empty_map() {
        let policy = make_policy(
            vec!["other_tool"],
            vec!["admin"],
            vec![simple_rule("glob_tool_*", vec!["admin"])],
        );
        let map = expand_globs(&policy);
        assert!(map.is_empty());
    }

    #[test]
    fn expand_concrete_not_in_tools_still_inserted() {
        // Concrete names bypass the tools[] universe check
        let policy = make_policy(
            vec![], // empty tools list
            vec!["viewer"],
            vec![simple_rule("read_file", vec!["viewer"])],
        );
        let map = expand_globs(&policy);
        assert!(map.contains_key(&("read_file".into(), "viewer".into())));
    }

    #[test]
    fn expand_full_example_policy() {
        let json = include_str!("../../../policies/example.json");
        let policy = policy_core::parse_policy(json).unwrap();
        let map = expand_globs(&policy);

        // Concrete rules
        assert!(map.contains_key(&("read_file".into(), "viewer".into())));
        assert!(map.contains_key(&("write_file".into(), "supervisor".into())));
        assert!(map.contains_key(&("dangerous_delete".into(), "admin".into())));

        // Glob expansion: glob_tool_* → admin for both alpha and beta
        assert!(map.contains_key(&("glob_tool_alpha".into(), "admin".into())));
        assert!(map.contains_key(&("glob_tool_beta".into(), "admin".into())));

        // Glob should NOT produce a key for the pattern itself
        assert!(!map.contains_key(&("glob_tool_*".into(), "admin".into())));

        // open_tool has three roles
        assert!(map.contains_key(&("open_tool".into(), "viewer".into())));
        assert!(map.contains_key(&("open_tool".into(), "supervisor".into())));
        assert!(map.contains_key(&("open_tool".into(), "admin".into())));
    }

    #[test]
    fn expand_unknown_role_is_skipped() {
        let policy = make_policy(
            vec!["read_file"],
            vec!["viewer"], // "ghost" is not declared
            vec![simple_rule("read_file", vec!["viewer", "ghost"])],
        );
        let map = expand_globs(&policy);
        // known role is inserted
        assert!(map.contains_key(&("read_file".into(), "viewer".into())));
        // unknown role is silently skipped by the expander
        assert!(!map.contains_key(&("read_file".into(), "ghost".into())));
    }
}
