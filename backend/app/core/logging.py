import structlog
from pydantic_settings import BaseSettings
import logging


class LogConfig(BaseSettings):
    def _build_structlog_processors(self):
        processors = [
            # Combines variables defined in the execution local global
            # context with variables bound to a specific logger.
            structlog.contextvars.merge_contextvars,
            # Adds a variable indicating the log level.
            structlog.processors.add_log_level,
            # Adds the stack trace to exceptions
            structlog.processors.StackInfoRenderer(),
            # Adds a timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        return processors
    
    def get_logger(self) -> structlog.stdlib.BoundLogger:
        """
        Fetch a new logger configured based on the current class state.
        """
        structlog.configure(
            processors=self._build_structlog_processors(),
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            # This is the logger that ultimately renders the output.
            # Many types of loggers are supported including the stdlib logger.
            # PrintLogger is a simple logger which just writes to stdout.
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )
        # Use structlog.stdlib.get_logger used type hinting
        return structlog.get_logger()
    

            