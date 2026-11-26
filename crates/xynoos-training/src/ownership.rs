use std::{
    fs::{self, File, OpenOptions},
    io,
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

#[derive(Debug, PartialEq, Eq)]
pub enum LockError {
    Contended,
    RecoveryFailure(io::ErrorKind),
}

/// A kernel-backed exclusive lock held for the lifetime of this value.
///
/// The lock file is intentionally persistent. Removing a locked path can let
/// another process create and lock a different inode, splitting ownership.
#[derive(Debug)]
pub struct AdvisoryLock {
    file: File,
}

impl AdvisoryLock {
    pub fn acquire(path: &Path) -> Result<Self, LockError> {
        validate_lock_path(path)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .map_err(recovery_failure)?;
        if !file.metadata().map_err(recovery_failure)?.is_file() {
            return Err(LockError::RecoveryFailure(io::ErrorKind::InvalidInput));
        }
        validate_lock_path(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { file }),
            Err(std::fs::TryLockError::WouldBlock) => Err(LockError::Contended),
            Err(std::fs::TryLockError::Error(error)) => Err(recovery_failure(error)),
        }
    }

    /// Probe whether an owner currently holds the kernel lock without waiting.
    pub fn is_held(path: &Path) -> Result<bool, LockError> {
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(recovery_failure(error)),
            Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
                return Err(LockError::RecoveryFailure(io::ErrorKind::InvalidInput));
            }
            Ok(_) => {}
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(recovery_failure)?;
        validate_lock_path(path)?;
        match file.try_lock() {
            Ok(()) => {
                file.unlock().map_err(recovery_failure)?;
                Ok(false)
            }
            Err(std::fs::TryLockError::WouldBlock) => Ok(true),
            Err(std::fs::TryLockError::Error(error)) => Err(recovery_failure(error)),
        }
    }
}

impl Drop for AdvisoryLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn validate_lock_path(path: &Path) -> Result<(), LockError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(LockError::RecoveryFailure(io::ErrorKind::InvalidInput)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(recovery_failure(error)),
    }
}

fn recovery_failure(error: io::Error) -> LockError {
    LockError::RecoveryFailure(error.kind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::MetadataExt,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn root() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "xyownership-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn kernel_lock_is_exclusive_released_and_keeps_its_inode() {
        let root = root();
        let path = root.join(".run.lock");
        let first = AdvisoryLock::acquire(&path).unwrap();
        let inode = fs::metadata(&path).unwrap();
        assert!(AdvisoryLock::is_held(&path).unwrap());
        assert!(matches!(
            AdvisoryLock::acquire(&path),
            Err(LockError::Contended)
        ));
        drop(first);
        assert!(!AdvisoryLock::is_held(&path).unwrap());
        let second = AdvisoryLock::acquire(&path).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().ino(), inode.ino());
        drop(second);
        fs::remove_dir_all(root).unwrap();
    }
}
