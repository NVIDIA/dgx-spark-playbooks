#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def __init__(self, **kwargs):
        self.default_keys = {
            'timestamp': 'asctime',
            'level': 'levelname',
            'message': 'message'
        }
        self.default_keys.update(kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        """
        log_record = {}
        
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        if isinstance(record.msg, dict):
            log_record['message'] = record.msg.get('message', '')
            for key, value in record.msg.items():
                if key != 'message':
                    log_record[key] = value
        else:
            log_record['message'] = record.getMessage()
        
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        for key, value in record.__dict__.items():
            if key not in ['msg', 'args', 'exc_info', 'exc_text', 'stack_info', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 'levelname', 
                          'levelno', 'pathname', 'filename', 'module', 'name', 'thread', 
                          'threadName', 'processName', 'process']:
                log_record[key] = value
        
        return json.dumps(log_record)

def setup_logger(name: str = 'backend', 
                 level: int = logging.INFO, 
                 log_file: Optional[str] = 'app.log') -> logging.Logger:
    """
    Set up a JSON logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (None for no file logging)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.propagate = False
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()


def log_request(request_data: Dict[str, Any], endpoint: str) -> None:
    """
    Log an API request with structured data.
    """
    logger.info({
        'message': f'API request to {endpoint}',
        'endpoint': endpoint,
        'request_data': request_data
    })


def log_response(response_data: Dict[str, Any], endpoint: str, status_code: int = 200) -> None:
    """
    Log an API response with structured data.
    """
    logger.info({
        'message': f'API response from {endpoint}',
        'endpoint': endpoint,
        'status_code': status_code,
        'response_data': response_data
    })


def log_error(error: Exception, endpoint: str = None, request_data: Dict[str, Any] = None) -> None:
    """
    Log an error with structured data.
    """
    error_data = {
        'message': f'Error: {str(error)}',
        'error_type': error.__class__.__name__,
    }
    
    if endpoint:
        error_data['endpoint'] = endpoint
    
    if request_data:
        error_data['request_data'] = request_data
        
    logger.error(error_data, exc_info=True) 