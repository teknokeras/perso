//! perso — policy-compiler
//! Phase 5: CLI with two subcommands:
//!   validate --policy <path>                      parse + glob-expand, report errors
//!   build    --policy <path> --output <path>      compile to .wasm

use std::path::PathBuf;
use std::process::{Command, ExitCode};

use clap::{Parser, Subcommand};

use policy_core::parse_policy;
use policy_runtime::expander::expand_globs;

use boon::{Compiler, Schemas};
use serde_json::Value;

// ─── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "policy-compiler",
    about   = "perso — compile and validate MCP policy files",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse a policy JSON file and report any errors. No WASM is produced.
    Validate {
        #[arg(short, long)]
        policy: PathBuf,
    },

    /// Compile the policy-runtime engine to WASM.
    /// The policy JSON is NOT embedded — load it separately via init() at runtime.
    Build {
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Validate a policy JSON file, then compile the engine to WASM only if valid.
    /// The policy JSON is NOT embedded — load it separately via init() at runtime.
    ValidateAndBuild {
        #[arg(short, long)]
        policy: PathBuf,

        #[arg(short, long)]
        output: PathBuf,
    },
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let cli = Cli::parse();

    let ok = match cli.command {
        Cmd::Validate { policy } => run_validate(&policy),
        Cmd::Build { output } => run_build(&output),
        Cmd::ValidateAndBuild { policy, output } => {
            println!("perso: validating policy before build…");
            if !run_validate(&policy) {
                return ExitCode::FAILURE;
            }
            println!("perso: policy valid — proceeding to build…");
            run_build(&output)
        }
    };

    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

// ─── validate ─────────────────────────────────────────────────────────────────

// Embed the schema at compile time — no external file dependency at runtime
const POLICY_SCHEMA: &str = include_str!("../../../policies/policy.schema.json");

fn validate_schema(json: &str) -> Result<(), String> {
    let instance: Value = serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;

    let schema_value: Value = serde_json::from_str(POLICY_SCHEMA).expect("bundled schema is valid");

    let mut schemas = Schemas::new();
    let mut compiler = Compiler::new();

    compiler
        .add_resource("policy.schema.json", schema_value)
        .map_err(|e| format!("schema compile error: {e}"))?;

    let schema_id = compiler
        .compile("policy.schema.json", &mut schemas)
        .map_err(|e| format!("schema compile error: {e}"))?;

    schemas
        .validate(&instance, schema_id)
        .map_err(|e| format!("{e}"))
}

/// Returns `true` on success, `false` on any hard error.
fn run_validate(policy_path: &PathBuf) -> bool {
    println!("perso: validating {}", policy_path.display());

    // 1. Read file
    let json = match std::fs::read_to_string(policy_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e}", policy_path.display());
            return false;
        }
    };

    // 2. ✅ NEW: JSON Schema check — catches structural/type errors first
    if let Err(e) = validate_schema(&json) {
        eprintln!("error: policy failed schema validation: {e}");
        return false;
    }

    // 3. Semantic parse — now safe to call, structure is guaranteed correct
    let policy = match parse_policy(&json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: JSON parse failed: {e}");
            return false;
        }
    };

    // 4. Basic field checks
    if policy.version.is_empty() {
        eprintln!("error: 'version' field is empty");
        return false;
    }
    if policy.tools.is_empty() {
        eprintln!("warning: 'tools' array is empty — no globs can be expanded");
    }
    if policy.rules.is_empty() {
        eprintln!("warning: 'rules' array is empty — all calls will hit default_action");
    }
    if policy.roles.is_empty() {
        eprintln!("warning: 'roles' array is empty — no rules can ever match");
    }

    // 5. Glob expansion — report unmatched patterns and empty-role rules
    let map = expand_globs(&policy);
    let mut warnings = 0usize;

    for rule in &policy.rules {
        for role in &rule.roles {
            if !policy.roles.contains(role) {
                eprintln!(
                    "error: rule for '{}' references role '{}' which is not listed in roles[]",
                    rule.tool_name, role
                );
                return false;
            }
        }

        if rule.tool_name.contains('*') {
            let matched = policy
                .tools
                .iter()
                .filter(|t| policy_runtime::expander::glob_matches(&rule.tool_name, t))
                .count();

            if matched == 0 {
                eprintln!(
                    "warning: glob '{}' matched 0 tools in tools[]",
                    rule.tool_name
                );
                warnings += 1;
            } else {
                println!(
                    "  glob '{}' → {} tool(s) × {} role(s) = {} map entries",
                    rule.tool_name,
                    matched,
                    rule.roles.len(),
                    matched * rule.roles.len()
                );
            }
        } else {
            // ✅ NEW: concrete name must exist in tools[]
            if !policy.tools.contains(&rule.tool_name) {
                eprintln!(
                    "error: rule references tool '{}' which is not listed in tools[]",
                    rule.tool_name
                );
                return false; // hard error, not a warning
            }
        }

        if rule.roles.is_empty() {
            eprintln!(
                "warning: rule for '{}' has an empty roles array — it will never match",
                rule.tool_name
            );
            warnings += 1;
        }
    }

    println!(
        "ok: {} rule(s), {} tool(s), {} map entries, {} warning(s)",
        policy.rules.len(),
        policy.tools.len(),
        map.len(),
        warnings
    );

    true
}

// ─── build ────────────────────────────────────────────────────────────────────

/// Returns `true` on success, `false` on any hard error.
fn run_build(output_path: &PathBuf) -> bool {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&manifest_dir)
        .to_path_buf();

    println!(
        "perso: building policy-runtime → wasm32-unknown-unknown (workspace: {})",
        workspace_root.display()
    );
    println!("note:  policy JSON is not embedded — pass it to init() at runtime");

    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "policy-runtime",
        ])
        .current_dir(&workspace_root)
        .status();

    match status {
        Err(e) => {
            eprintln!("error: failed to invoke cargo: {e}");
            eprintln!("hint:  ensure cargo is on PATH");
            return false;
        }
        Ok(s) if !s.success() => {
            eprintln!("error: cargo build failed (exit {})", s);
            eprintln!("hint:  install the WASM target: rustup target add wasm32-unknown-unknown");
            return false;
        }
        Ok(_) => {}
    }

    let wasm_src = workspace_root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("policy_runtime.wasm");

    if !wasm_src.exists() {
        eprintln!(
            "error: expected WASM artifact not found at {}",
            wasm_src.display()
        );
        return false;
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "error: cannot create output directory '{}': {e}",
                    parent.display()
                );
                return false;
            }
        }
    }

    match std::fs::copy(&wasm_src, output_path) {
        Ok(bytes) => {
            println!("ok: {} bytes → {}", bytes, output_path.display());
            true
        }
        Err(e) => {
            eprintln!("error: copy failed: {e}");
            false
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn example_policy_path() -> PathBuf {
        // CARGO_MANIFEST_DIR is crates/policy-compiler at test time
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/example.json")
    }

    #[test]
    fn validate_example_policy_succeeds() {
        assert!(run_validate(&example_policy_path()));
    }

    #[test]
    fn validate_missing_file_fails() {
        assert!(!run_validate(&PathBuf::from(
            "/nonexistent/path/policy.json"
        )));
    }

    #[test]
    fn validate_bad_json_fails() {
        let dir = std::env::temp_dir();
        let path = dir.join("perso_bad_policy.json");
        fs::write(&path, b"not valid json {{{{").unwrap();
        assert!(!run_validate(&path));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn validate_empty_version_fails() {
        let dir = std::env::temp_dir();
        let path = dir.join("perso_empty_version.json");
        let json = r#"{
            "version": "",
            "default_action": "Deny",
            "tools": ["read_file"],
            "rules": []
        }"#;
        fs::write(&path, json).unwrap();
        assert!(!run_validate(&path));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn validate_unmatched_glob_warns_but_succeeds() {
        let dir = std::env::temp_dir();
        let path = dir.join("perso_unmatched_glob.json");
        let json = r#"{
            "version": "perso-1.0.0",
            "default_action": "Deny",
            "tools": ["read_file"],
            "rules": [
                { "tool_name": "nonexistent_*", "roles": ["admin"], "condition": null }
            ]
        }"#;
        fs::write(&path, json).unwrap();
        // Unmatched glob is a warning, not a hard error
        assert!(run_validate(&path));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn validate_empty_roles_warns_but_succeeds() {
        let dir = std::env::temp_dir();
        let path = dir.join("perso_empty_roles.json");
        let json = r#"{
            "version": "perso-1.0.0",
            "default_action": "Deny",
            "tools": ["read_file"],
            "rules": [
                { "tool_name": "read_file", "roles": [], "condition": null }
            ]
        }"#;
        fs::write(&path, json).unwrap();
        assert!(run_validate(&path));
        let _ = fs::remove_file(&path);
    }
}
