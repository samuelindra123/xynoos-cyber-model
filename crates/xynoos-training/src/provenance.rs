use std::{path::Path, process::Command};

pub const SOURCE_REPOSITORY_ID: &str = "samuelindra123/xynoos-cyber-model";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub repository_id: String,
    pub revision: String,
}

/// Confirm that the checkout used to build the runtime is the clean checkout
/// named by the immutable experiment configuration.
pub fn attest_source_checkout(
    path: &Path,
    build_repository_id: &str,
    build_revision: &str,
    configured_repository_id: &str,
    configured_revision: &str,
) -> Result<SourceIdentity, String> {
    if configured_repository_id != SOURCE_REPOSITORY_ID
        || build_repository_id != SOURCE_REPOSITORY_ID
        || configured_repository_id != build_repository_id
    {
        return Err("repository identity does not match the canonical project".into());
    }
    if !is_revision(build_revision)
        || !is_revision(configured_revision)
        || configured_revision != build_revision
    {
        return Err("configuration source revision does not match the training binary".into());
    }

    let root = git(path, &["rev-parse", "--show-toplevel"])?;
    let remote = git(Path::new(&root), &["remote", "get-url", "origin"])?;
    let actual_repository_id = normalize_github_remote(&remote)
        .ok_or_else(|| "origin is not an unambiguous GitHub repository URL".to_owned())?;
    if actual_repository_id != configured_repository_id {
        return Err("checkout origin does not match the configured repository identity".into());
    }
    let actual_revision = git(Path::new(&root), &["rev-parse", "HEAD"])?;
    if actual_revision != build_revision {
        return Err("checkout revision does not match the training binary revision".into());
    }
    let status = git(
        Path::new(&root),
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        return Err("source checkout is dirty or contains untracked files".into());
    }
    let ignored_sources = git(
        Path::new(&root),
        &[
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
        ],
    )?;
    if !ignored_sources.is_empty() {
        return Err("source checkout contains ignored untracked implementation files".into());
    }
    Ok(SourceIdentity {
        repository_id: actual_repository_id,
        revision: actual_revision,
    })
}

fn git(path: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(arguments)
        .output()
        .map_err(|error| format!("could not execute git for source identity: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed while verifying source identity",
            arguments.join(" ")
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| "git returned invalid source identity text".into())
}

fn is_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn normalize_github_remote(value: &str) -> Option<String> {
    let value = value.trim().trim_end_matches('/');
    let path = value
        .strip_prefix("https://github.com/")
        .or_else(|| value.strip_prefix("http://github.com/"))
        .or_else(|| value.strip_prefix("ssh://git@github.com/"))
        .or_else(|| value.strip_prefix("git@github.com:"))?;
    let path = path.strip_suffix(".git").unwrap_or(path);
    let (owner, repository) = path.split_once('/')?;
    if owner.is_empty()
        || repository.is_empty()
        || repository.contains('/')
        || owner
            .chars()
            .any(|character| matches!(character, '?' | '#' | '@' | ':'))
        || repository
            .chars()
            .any(|character| matches!(character, '?' | '#' | '@' | ':'))
    {
        return None;
    }
    Some(format!("{owner}/{repository}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
    };

    #[test]
    fn repository_identity_normalizes_only_supported_github_remotes() {
        for remote in [
            "https://github.com/samuelindra123/xynoos-cyber-model.git",
            "https://github.com/samuelindra123/xynoos-cyber-model",
            "git@github.com:samuelindra123/xynoos-cyber-model.git",
            "ssh://git@github.com/samuelindra123/xynoos-cyber-model.git",
        ] {
            assert_eq!(
                normalize_github_remote(remote).as_deref(),
                Some(SOURCE_REPOSITORY_ID)
            );
        }
        assert_eq!(
            normalize_github_remote("https://github.com/someone-else/xynoos-cyber-model.git")
                .as_deref(),
            Some("someone-else/xynoos-cyber-model")
        );
        for remote in [
            "https://github.com/samuelindra123/xynoos-cyber-model/tree/main",
            "https://example.test/samuelindra123/xynoos-cyber-model.git",
        ] {
            assert_eq!(normalize_github_remote(remote), None);
        }
    }

    #[test]
    fn checkout_attestation_checks_origin_revision_tracked_files_and_untracked_files() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "xyprovenance-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let git = |arguments: &[&str]| {
            let output = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(arguments)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "git {:?}: {}",
                arguments,
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        };
        git(&["init", "--quiet"]);
        git(&["config", "user.name", "XYNOOS Test"]);
        git(&["config", "user.email", "test@example.invalid"]);
        fs::write(root.join("tracked.txt"), "attested\n").unwrap();
        git(&["add", "tracked.txt"]);
        git(&["commit", "--quiet", "-m", "fixture source revision"]);
        git(&[
            "remote",
            "add",
            "origin",
            "https://github.com/samuelindra123/xynoos-cyber-model.git",
        ]);
        let revision = git(&["rev-parse", "HEAD"]);

        let identity = attest_source_checkout(
            &root,
            SOURCE_REPOSITORY_ID,
            &revision,
            SOURCE_REPOSITORY_ID,
            &revision,
        )
        .unwrap();
        assert_eq!(identity.repository_id, SOURCE_REPOSITORY_ID);
        assert_eq!(identity.revision, revision);

        fs::write(root.join("untracked.txt"), "must be rejected\n").unwrap();
        assert!(
            attest_source_checkout(
                &root,
                SOURCE_REPOSITORY_ID,
                &revision,
                SOURCE_REPOSITORY_ID,
                &revision,
            )
            .unwrap_err()
            .contains("untracked")
        );
        fs::remove_file(root.join("untracked.txt")).unwrap();

        fs::write(root.join("tracked.txt"), "modified\n").unwrap();
        assert!(
            attest_source_checkout(
                &root,
                SOURCE_REPOSITORY_ID,
                &revision,
                SOURCE_REPOSITORY_ID,
                &revision,
            )
            .unwrap_err()
            .contains("dirty")
        );
        fs::write(root.join("tracked.txt"), "attested\n").unwrap();

        fs::create_dir_all(root.join("crates/xynoos-training/src")).unwrap();
        fs::write(
            root.join(".git/info/exclude"),
            "/crates/xynoos-training/src/hidden.rs\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/xynoos-training/src/hidden.rs"),
            "ignored implementation file\n",
        )
        .unwrap();
        assert!(
            attest_source_checkout(
                &root,
                SOURCE_REPOSITORY_ID,
                &revision,
                SOURCE_REPOSITORY_ID,
                &revision,
            )
            .unwrap_err()
            .contains("ignored untracked")
        );
        fs::remove_dir_all(root.join("crates")).unwrap();

        assert!(
            attest_source_checkout(
                &root,
                SOURCE_REPOSITORY_ID,
                &revision,
                "someone-else/xynoos-cyber-model",
                &revision,
            )
            .is_err()
        );
        assert!(
            attest_source_checkout(
                &root,
                SOURCE_REPOSITORY_ID,
                &revision,
                SOURCE_REPOSITORY_ID,
                &"0".repeat(40),
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
