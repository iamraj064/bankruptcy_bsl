import os
import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "app.log")

class CustomLogger:
    def __init__(self, name):
        self.name = name

    def info(self, message):
        self.log("INFO", message)

    def error(self, message):
        self.log("ERROR", message)

    def warn(self, message):
        self.log("WARNING", message)

    def log(self, level, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"{timestamp} | {level} | {self.name} | {message}\n"
        
        # Write to app.log
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass
            
        # Also print to standard console for visual debugging
        print(log_line, end="")

def setup_logger(name):
    return CustomLogger(name)
