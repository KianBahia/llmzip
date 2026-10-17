"""Exception hierarchy. The CLI turns any LlmzipError into a clean message + exit 1."""


class LlmzipError(Exception):
    pass


class UsageError(Exception):
    """Bad command-line usage (exit code 2)."""


class Cancelled(LlmzipError):
    """Raised from a progress callback to stop a running job; nothing has been written."""


class ContainerError(LlmzipError):
    """The .lz file is malformed, incompatible or corrupt."""


class BadMagic(ContainerError):
    pass


class UnsupportedVersion(ContainerError):
    pass


class Truncated(ContainerError):
    pass


class CorruptContainer(ContainerError):
    """Structurally invalid: unknown mode, inconsistent counts, trailing data, bad zstd frame."""


class FingerprintMismatch(ContainerError):
    pass


class ChecksumMismatch(ContainerError):
    pass
