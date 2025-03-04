# Frontend Implementation Plan for OS Legal Explorer

## 1. Feature Overview

### Single-Page Application
- **Search Box for Legal Cases:**  
  - A search input in the hero section that calls the CourtListener API endpoint (`/api/rest/v4/search/`) to retrieve the associated `cluster_id` for legal cases.
  - After that's returned, we can query our local database for that cluster_id to get the opinion text.
  - Add search filters for jurisdiction, date range, and court level.
- **Recent Cases List:**  
  - Display a list of the most recently added court cases pulled from the backend.
  - Include pagination for browsing through more cases.
  - Show key metadata: case name, court, date, and a brief excerpt.
- **Dynamic Case Detail Section:**
  - **Case Text Display:**  
    - Show the full text of the selected court opinion in a section that appears when a case is selected.
    - (future) Implement text highlighting for citations and key terms.
    - (future) Add a table of contents for navigating long opinions.
    - (future) Include a "copy citation" feature for academic/legal reference.
  - **Citation Mapping Visualization:**  
    - Use d3.js to render an interactive network visualization of the citation map (i.e., how cases cite each other).
    - Include filtering options to focus on specific time periods or courts.
    - Provide zoom and pan capabilities for exploring large citation networks.
    - Add tooltips showing case summaries when hovering over nodes.
    - Implement a "focus mode" to highlight direct connections to the current case.
    - Based on the relationship TREATMENT, make the connecting lines different colors, i.e. green for positive, yellow for caution, red for negative, and gray for neutral.
- **Feedback Mechanisms:**
  - Add UI elements for users to report missing citations or opinions.
  - Include a general feedback form for user suggestions.
  


---

## 2. Guiding Implementation Steps

### Step 1: Initial Setup & Directory Structure 
- **Create Frontend Structure:**  
  - Create the directory structure for frontend assets.
  ```
  mkdir -p src/frontend/templates src/frontend/static/{css,js,images}
  ```
  - Set up the build system for Tailwind CSS:
  ```
  npm init -y
  npm install -D tailwindcss postcss autoprefixer
  npx tailwindcss init -p
  ```
  - Configure Tailwind CSS in `tailwind.config.js`:
  ```javascript
  module.exports = {
    content: ["./src/frontend/templates/**/*.html", "./src/frontend/static/js/**/*.js"],
    theme: {
      extend: {},
    },
    plugins: [],
  }
  ```
  - Create a base CSS file at `src/frontend/static/css/global.css`:

- **Configure FastAPI for Frontend:**
  - Uncomment and update the template and static file configuration in `src/api/main.py`.
  - Set up Jinja2 templates with proper directory paths.
  - Create a base template with common layout elements.

### Step 2: Single-Page Application Implementation
- **Create Base Layout Template:**
  - Develop a base.html template with common elements (header, footer, navigation).
  - Implement responsive design using Tailwind CSS.
  - Set up meta tags and basic SEO elements.
  
- **Design the Single-Page Template:**
  - Create `src/frontend/templates/index.html` extending the base template.
  - Design a clean, modern interface with a prominent search box in the hero section.
  - Add sections for recent cases and a dynamic case detail area.
  - Implement responsive layout for different screen sizes.
  
- **Implement Search Functionality:**
  - Create a search component with HTML, CSS, and JavaScript.
  - Implement debouncing for search input to prevent excessive API calls.
  - Add a loading indicator for search results.
  - Create a backend endpoint in FastAPI that proxies the CourtListener API:
  ```python
  @app.get("/api/search")
  async def search_cases(query: str, jurisdiction: Optional[str] = None, 
                         start_date: Optional[str] = None, end_date: Optional[str] = None):
      # Call CourtListener API and process results
      # Return formatted data for frontend consumption
  ```
  - Implement error handling for API failures.
  - Add search filters with dropdowns for jurisdiction, date range, etc.
    - use courtlistener API docs to pass along the filtering.

- **Recent Cases List:**
  - Create a backend endpoint to fetch recent cases:
  ```python
  @app.get("/api/recent-cases")
  async def get_recent_cases(limit: int = 10, offset: int = 0):
      # Query Neo4j database for recent cases
      # Return formatted case data
  ```
  - Design a card-based UI for displaying case information.
  - Implement pagination controls.
  - Add a "load more" button for infinite scrolling.

- **Dynamic Case Detail Section:**
  - Create a section that appears when a case is selected:
    - Case metadata (court, date, judges, etc.)
    - Full opinion text
    - Citation visualization
    - Related cases
  - Implement a responsive layout that works well on different screen sizes.
  - Add smooth transitions between the search results view and case detail view.
  
- **Case Text Display:**
  - Create a backend endpoint to fetch case details:
  ```python
  @app.get("/api/case/{case_id}")
  async def get_case_details(case_id: str):
      # Fetch case data from database
      # Return formatted case with full text and metadata
  ```
  - Implement text formatting for legal opinions.
  - (FUTURE) Add syntax highlighting for citations within the text.
  - (FUTURE) Create a table of contents based on opinion sections.
  - (FUTURE) Add a "copy citation" button with proper legal citation format.
  
- **Citation Mapping with d3.js:**
  - Create a backend endpoint for citation network data:
  ```python
  @app.get("/api/case/{case_id}/citations")
  async def get_citation_network(case_id: str, depth: int = 1):
      # Query Neo4j for citation relationships
      # Return nodes and edges in a format suitable for d3.js
  ```
  - Develop a d3.js visualization module in `src/frontend/static/js/citation_map.js`.
  - Implement a force-directed graph layout.
  - Add interactive features:
    - Zoom and pan controls
    - Node selection and highlighting
    - Filtering options by court, date, etc.
  - Create a legend explaining node and edge types.
  - Citation networks should never be larger than 100-200 nodes, so do not think we will have to worry about performance much.

### Step 3: User Feedback Mechanisms
- **Design Feedback UI Components:**
  - Create modal dialogs for different feedback types that overlay the single-page application.
  - Implement form validation for feedback submissions.
  
- **Implement Feedback Endpoints:**
  - Create backend endpoints for each feedback type:
  ```python
  @app.post("/api/feedback/citation/missing")
  async def report_missing_citation(feedback: MissingCitationFeedback):
      # Process and store feedback
      # Return confirmation
  
  @app.post("/api/feedback/opinion/missing")
  async def report_missing_opinion(feedback: MissingOpinionFeedback):
      # Process and store feedback
      # Return confirmation
  
  @app.post("/api/feedback/general")
  async def submit_general_feedback(feedback: GeneralFeedback):
      # Process and store feedback
      # Return confirmation
  ```
  - Just store feedback as csv's or json for now, database later.
  
- **Deployment Preparation:**
  - Update Dockerfile and docker-compose.yml to include frontend build steps.
  - Configure Caddy for serving static files and proxying API requests.
  - Set up Cloudflared tunnel for secure external access.
  
- **Monitoring and Analytics:**
  - (FUTURE) Implement basic analytics to track user behavior.
  - (FUTURE) Set up error logging and monitoring.

---

## 3. Detailed Directory Structure & File Roles

```
repo/
├── neo4j_bulk_import/            # Existing backend code related to Neo4j operations
├── src/
│   ├── api/                     # FastAPI backend, endpoints, and business logic
│   │   ├── main.py             # App entry point, includes mounting static files and templates
│   │   ├── routers/            # API route definitions
│   │   │   ├── search.py       # Search-related endpoints
│   │   │   ├── cases.py        # Case detail endpoints
│   │   │   └── feedback.py     # User feedback endpoints
│   │   └── services/           # Business logic and external API interactions
│   │       ├── courtlistener.py # CourtListener API client
│   │       └── citation.py     # Citation processing service
│   ├── citation/                # Citation parsing and resolution utilities
│   └── frontend/                # Frontend assets and templates
│       ├── templates/           # Jinja2 templates
│       │   ├── base.html       # Base template with common elements
│       │   ├── index.html      # Single-page application template
│       │   └── components/     # Reusable template components
│       │       ├── search.html # Search box component
│       │       ├── pagination.html # Pagination controls
│       │       ├── case_detail.html # Case detail component
│       │       └── feedback.html # Feedback forms
│       └── static/              # Frontend static assets
│           ├── css/             # CSS files
│           │   ├── global.css     # Main Tailwind CSS entry point
│           │   └── components/  # Component-specific styles
│           ├── js/              # JavaScript files
│           │   ├── app.js       # Main application logic
│           │   ├── search.js    # Search functionality
│           │   ├── citation_map.js # d3.js visualization
│           │   └── utils/       # Utility functions
│           │       ├── api.js   # API client functions
│           │       └── dom.js   # DOM manipulation helpers
│           └── images/          # Static images
├── Dockerfile                   # Updated to include Node.js/Tailwind build steps
├── docker-compose.yml           # May include services for both backend and frontend build
├── Makefile                     # Includes targets for building and watching frontend assets
├── package.json                 # Node.js dependencies and scripts
├── tailwind.config.js           # Tailwind CSS configuration
└── README.md                    # Updated project documentation and roadmap
```

### File Roles Detail:
- **`src/api/main.py`:**  
  - Initialize FastAPI, configure CORS, mount static files, and include routes.
  - Set up Jinja2 templates for server-side rendering.

- **`src/api/routers/*.py`:**
  - Modular API endpoints organized by feature area.
  - Each router handles a specific set of related functionality.

- **`src/frontend/templates/base.html`:**
  - Base template with common HTML structure, meta tags, and layout.
  - Includes navigation, header, footer, and script/style imports.

- **`src/frontend/templates/index.html`:**
  - Single-page application layout extending base.html.
  - Contains search box, recent cases list, and dynamic case detail section.

- **`src/frontend/templates/components/case_detail.html`:**
  - Component template for case details.
  - Displays case metadata, full opinion text, and citation visualization.
  - Designed to be loaded dynamically when a case is selected.

- **`src/frontend/static/css/global.css`:**
  - Main CSS file with Tailwind directives and custom styles.
  - Imports component-specific styles as needed.

- **`src/frontend/static/js/app.js`:**
  - Main application logic for the single-page application.
  - Handles state management and transitions between views.
  - Coordinates the loading and display of different components.

- **`src/frontend/static/js/search.js`:**
  - Handles search input, API calls, and results display.
  - Implements debouncing, loading states, and error handling.

- **`src/frontend/static/js/citation_map.js`:**
  - d3.js module for rendering the citation network visualization.
  - Handles user interactions, filtering, and dynamic data updates.

- **`src/frontend/static/js/utils/api.js`:**
  - Utility functions for making API calls.
  - Handles authentication, error handling, and response parsing.

- **`Makefile`:**
  - Contains targets for frontend development:
    - `frontend-build`: Compile Tailwind CSS and bundle JS assets.
    - `frontend-watch`: Watch for changes and rebuild automatically.
    - `frontend-test`: Run frontend tests.

- **`package.json`:**
  - Defines Node.js dependencies and scripts for frontend development.
  - Includes build, watch, and test commands.

- **`tailwind.config.js`:**
  - Configuration for Tailwind CSS, including custom theme settings.
  - Defines content paths for purging unused styles in production.

---

## 4. API Endpoints Overview

### Search Endpoints
- **/api/search**: Acts as a proxy to the CourtListener API. It processes the user's search input, fetches results from the external API, extracts vital data (e.g., cluster_id), and returns it to the frontend.
  - Parameters: query (string), jurisdiction (optional), start_date (optional), end_date (optional)
  - Returns: List of matching cases with metadata

### Case Data Endpoints
- **/api/recent-cases**: Queries the Neo4j database for the most recently added opinions.
  - Parameters: limit (default: 10), offset (default: 0)
  - Returns: List of recent cases with metadata

- **/api/case/{case_id}**: Retrieves detailed information for a specific case, including the full opinion text and metadata.
  - Parameters: case_id (string)
  - Returns: Complete case data including text and metadata

- **/api/case/{case_id}/citations**: Retrieves the citation network for a specific case.
  - Parameters: case_id (string), depth (default: 1)
  - Returns: Nodes and edges representing the citation network

### Feedback Endpoints
- **/api/feedback/citation/missing**: Accepts feedback reporting missing citations for a given case.
  - Parameters: case_id, expected_citation, description
  - Returns: Confirmation of submission

- **/api/feedback/citation**: Collects user feedback related to the accuracy and quality of citation mappings.
  - Parameters: case_id, citation_id, feedback_type, description
  - Returns: Confirmation of submission

- **/api/feedback/opinion/missing**: Allows users to report cases where the full opinion text is missing.
  - Parameters: case_id, description
  - Returns: Confirmation of submission

- **/api/feedback/general**: Serves as a catch-all for other feedback regarding the platform.
  - Parameters: feedback_type, description, email (optional)
  - Returns: Confirmation of submission

---

## 5. Development Timeline and Milestones

### Week 1: Setup and Infrastructure
- Complete directory structure setup
- Configure build tools (Tailwind, etc.)
- Set up base templates and styling
- Implement FastAPI integration with templates

### Week 2: Single-Page Application Core
- Complete search functionality
- Implement recent cases list
- Design and implement responsive layout
- Create dynamic case detail section

### Week 3: Case Detail and Visualization
- Implement case text display with formatting
- Create citation network visualization
- Add interactive features to visualization
- Implement smooth transitions between views

### Week 4: Feedback Mechanisms and Deployment
- Implement feedback forms and UI
- Create backend endpoints for feedback
- Set up feedback storage
- Prepare for deployment

---

## 4. Next Steps Recap

### Frontend
- **HTML/CSS/JavaScript**: Core web technologies
- **Tailwind CSS**: Utility-first CSS framework for styling
- **DaisyUI**: For css components that integrate with TailwindCSS.
- **HTMX**: For dynamic content without complex JavaScript
- **d3.js**: For interactive data visualizations
- **Alpine.js** (optional): Lightweight JavaScript framework for enhanced interactivity

### Backend
- **FastAPI**: Python web framework for API endpoints
- **Jinja2**: Template engine for server-side rendering
- **Neo4j**: Graph database for storing and querying citation networks
- **PostgreSQL**: Relational database for structured data

### Build Tools
- **Node.js/npm**: For managing frontend dependencies
- **PostCSS**: For processing CSS with Tailwind
- **Docker**: For containerization and deployment
- **Caddy**: Web server for production deployment
- **Cloudflared**: For secure tunneling

---


### Enhanced Visualization
- Timeline view of citations
- Court hierarchy visualization
- Citation sentiment analysis






