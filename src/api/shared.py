from fastapi.templating import Jinja2Templates
from datetime import date

# Configure templates
templates = Jinja2Templates(directory="src/frontend/templates")


# Add custom filters
def format_date(value):
    if isinstance(value, date):
        return value.strftime("%b %d, %Y")
    return value


templates.env.filters["date"] = format_date
