use std::{env, path::Path, process::Command};

fn output(directory: &Path, arguments: &[&str]) -> String {
    let result = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("could not execute git for training provenance: {error}"));
    if !result.status.success() {
        panic!(
            "could not attest training build provenance: git {} failed",
            arguments.join(" ")
        );
    }
    String::from_utf8(result.stdout)
        .unwrap_or_else(|_| panic!("git returned non-UTF-8 training provenance"))
        .trim()
        .to_owned()
}

fn repository_id(remote: &str) -> String {
    let remote = remote.trim().trim_end_matches('/');
    let path = remote
        .strip_prefix("https://github.com/")
        .or_else(|| remote.strip_prefix("http://github.com/"))
        .or_else(|| remote.strip_prefix("ssh://git@github.com/"))
        .or_else(|| remote.strip_prefix("git@github.com:"))
        .unwrap_or_else(|| panic!("training build origin must be a GitHub repository URL"));
    path.strip_suffix(".git").unwrap_or(path).to_owned()
}

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo");
    let crate_path = Path::new(&crate_dir);
    let repository_root = output(crate_path, &["rev-parse", "--show-toplevel"]);
    let repository_path = Path::new(&repository_root);
    let revision = output(repository_path, &["rev-parse", "HEAD"]);
    let remote = output(repository_path, &["remote", "get-url", "origin"]);
    let repository = repository_id(&remote);
    if repository != "samuelindra123/xynoos-cyber-model"
        || revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!("training binary must be built from the canonical XYNOOS Cyber Git revision");
    }
    let head_path = output(
        repository_path,
        &["rev-parse", "--path-format=absolute", "--git-path", "HEAD"],
    );
    let index_path = output(
        repository_path,
        &["rev-parse", "--path-format=absolute", "--git-path", "index"],
    );
    println!("cargo:rerun-if-changed={head_path}");
    println!("cargo:rerun-if-changed={index_path}");
    let symbolic_head = Command::new("git")
        .arg("-C")
        .arg(repository_path)
        .args(["symbolic-ref", "-q", "HEAD"])
        .output()
        .expect("could not inspect training build branch");
    if symbolic_head.status.success() {
        let reference = String::from_utf8_lossy(&symbolic_head.stdout)
            .trim()
            .to_owned();
        let reference_path = output(
            repository_path,
            &[
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                &reference,
            ],
        );
        println!("cargo:rerun-if-changed={reference_path}");
    }

    println!("cargo:rustc-env=XYNOOS_BUILD_SOURCE_REVISION={revision}");
    println!("cargo:rustc-env=XYNOOS_BUILD_REPOSITORY_ID={repository}");
    println!("cargo:rerun-if-changed=build.rs");
}
