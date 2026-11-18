import logging
import json
import sys
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            # Exclude raw image buffers or large binary arrays
            filtered_extra = {
                k: v for k, v in record.extra.items()
                if not any(token in k.lower() for token in ["byte", "image", "embedding", "face"])
            }
            log_data.update(filtered_extra)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    
    root_logger = logging.getLogger("pixelshield")
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    root_logger.propagate = False
    return root_logger

logger = setup_logging()
