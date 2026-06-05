class AutoClipperError(Exception):
    """Base exception untuk semua error domain."""


class PipelineError(AutoClipperError):
    """Error saat memproses clip."""


class ModelLoadError(AutoClipperError):
    """Gagal load ML model."""


class FFmpegError(AutoClipperError):
    """FFmpeg subprocess gagal."""

    def __init__(self, msg: str, returncode: int, stderr: str = ""):
        super().__init__(msg)
        self.returncode = returncode
        self.stderr = stderr


class LLMError(AutoClipperError):
    """LLM API call gagal."""


class TranscriptUnavailableError(AutoClipperError):
    """Transcript tidak tersedia untuk video ini."""
