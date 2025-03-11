import subprocess
from datetime import date

from fastapi.templating import Jinja2Templates

# Configure templates
templates = Jinja2Templates(directory="src/frontend/templates")


# Add custom filters
def format_date(value):
    if isinstance(value, date):
        return value.strftime("%b %d, %Y")
    return value


templates.env.filters["date"] = format_date


# Get current git commit hash
def get_git_commit_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


# Add global context variables
templates.env.globals["git_version"] = get_git_commit_hash()
