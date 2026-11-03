# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Discovery event logging."""

import json
import os
from typing import Any


class DiscoveryLogger:
    """Logs discovery events to JSONL file."""

    def __init__(self, log_path: str):
        """
        Initialize logger.

        Args:
            log_path: Path to events.jsonl file
        """
        self.log_path = log_path

        # Create directory if needed
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_event(self, event_type: str, data: dict[str, Any]):
        """
        Log an event.

        Args:
            event_type: Type of event ("hypothesis_proposed", "experiment_planned", etc.)
            data: Event data
        """
        event = {
            "event_type": event_type,
            **data
        }

        # Append to log file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
