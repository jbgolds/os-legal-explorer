# OS Legal Explorer - Frontend Implementation TODO

## Setup & Infrastructure
- [x] Create frontend directory structure
  - [x] Create `src/frontend/templates` directory
  - [x] Create `src/frontend/static/{css,js,images}` directories
- [x] Set up build system for Tailwind CSS
  - [x] Follow https://daisyui.com/docs/install/cli/
- [x] Configure Tailwind CSS in `tailwind.config.js`
- [x] Create base CSS file at `src/frontend/static/css/global.css`
- [x] Configure FastAPI for frontend
  - [x] Uncomment and update template and static file configuration in `src/api/main.py`
  - [x] Set up Jinja2 templates with proper directory paths

## Single-Page Application Implementation
- [x] Create base layout template
  - [x] Create `src/frontend/templates/base.html` template with common elements
    - [x] Add header with navigation
    - [x] Add footer with project information
    - [x] Include meta tags and basic SEO elements
    - [x] Set up responsive layout with Tailwind CSS
    - [x] Include script and style imports
  - [x] Implement responsive design using Tailwind CSS and DaisyUI
- [x] Design the single-page template
  - [x] Create `src/frontend/templates/index.html` extending the base template
  - [x] Design interface with prominent search box in hero section
  - [x] Add sections for recent cases
  - [x] Add dynamic case detail section that appears when a case is selected
  - [x] Implement responsive layout for different screen sizes
- [x] Implement search functionality
  - [x] Create search component with HTML, CSS, and JavaScript
  - [x] Implement debouncing for search input using JavaScript
  - [x] Add loading indicator for search results
  - [x] Create backend endpoint in FastAPI for search
    - [x] Implement proxy to CourtListener API (`/api/rest/v4/search/`)
    - [x] Add support for search filters (jurisdiction, date range, court level)
    - [x] Handle response formatting for frontend
  - [x] Implement error handling for API failures
  - [x] Add search filters (jurisdiction, date range, etc.)
- [x] Implement recent cases list
  - [x] Create backend endpoint to fetch recent cases
    - [x] Create `/api/recent-cases` endpoint with pagination support
    - [x] Query Neo4j database for recent cases (TODO: Currently using CourtListener API as a temporary solution)
    - [x] Format response with case metadata (name, court, date, excerpt)
  - [x] Design card-based UI for displaying case information
  - [x] Implement pagination controls
  - [x] Add "load more" button for infinite scrolling
- [x] Implement case detail section
  - [x] Create dynamic section that appears when a case is selected
  - [x] Implement case text display
    - [x] Create backend endpoint to fetch case details
      - [x] Create `/api/case/{case_id}` endpoint
      - [x] Fetch case data from Neo4j database (TODO: Currently using CourtListener API as a temporary solution)
      - [x] Return formatted case with full text and metadata
    - [x] Implement text formatting for legal opinions
    - [ ] (Future) Add syntax highlighting for citations within text
    - [ ] (Future) Create table of contents based on opinion sections
  - [x] Implement citation mapping with d3.js
    - [x] Create backend endpoint for citation network data
      - [x] Create `/api/case/{case_id}/citations` endpoint
      - [x] Query Neo4j for citation relationships (TODO: Currently using placeholder data)
      - [x] Return nodes and edges in format suitable for d3.js
    - [x] Develop d3.js visualization module in `src/frontend/static/js/citation_map.js`
    - [x] Implement force-directed graph layout
    - [x] Add interactive features (zoom, pan, filtering)
      - [x] Add zoom and pan controls
      - [x] Implement node selection and highlighting
      - [x] Add filtering options by court, date, etc.
    - [x] Create legend explaining node and edge types
    - [x] Implement color coding for different citation treatments
      - [x] Green for positive citations
      - [x] Yellow for cautionary citations
      - [x] Red for negative citations
      - [x] Gray for neutral citations

## User Feedback Mechanisms
- [x] Design feedback UI components
  - [x] Create modal dialogs for different feedback types
    - [x] Missing citation feedback form
    - [x] Missing opinion feedback form
    - [x] General feedback form
  - [x] Implement form validation
- [x] Implement feedback endpoints
  - [x] Create backend endpoints for each feedback type
    - [x] `/api/feedback/citation/missing` endpoint
    - [x] `/api/feedback/opinion/missing` endpoint
    - [x] `/api/feedback/general` endpoint
  - [x] Set up storage for feedback data (CSV/JSON files initially)

## JavaScript Implementation
- [x] Create utility modules
  - [x] Create `src/frontend/static/js/utils/api.js` for API client functions
  - [x] Create `src/frontend/static/js/utils/dom.js` for DOM manipulation helpers
- [x] Implement search functionality in `src/frontend/static/js/search.js`
  - [x] Handle search input and API calls
  - [x] Implement debouncing and loading states
  - [x] Display and format search results
- [x] Implement dynamic content loading in `src/frontend/static/js/app.js`
  - [x] Handle case selection and detail view display
  - [x] Manage state transitions between search results and case details
  - [x] Implement smooth animations for transitions
- [x] Implement citation visualization in `src/frontend/static/js/citation_map.js`
  - [x] Create d3.js force-directed graph
  - [x] Handle data loading and transformation
  - [x] Implement interactive features

## UI Enhancement with DaisyUI
- [x] Update base.html with DaisyUI components
- [x] Update index.html with DaisyUI components
- [x] Create and enhance component templates
  - [x] Create search_results.html with DaisyUI components
  - [x] Create case_detail.html with DaisyUI components
  - [x] Create search_filters.html with DaisyUI components
  - [x] Create recent_cases.html with DaisyUI components
  - [x] Create citation_network.html with DaisyUI components
  - [x] Create feedback_form.html with DaisyUI components
- [x] Add custom CSS styling in global.css

## Deployment Preparation
- [x] Update Dockerfile and docker-compose.yml
  - [x] Add Node.js/Tailwind build steps
  - [ ] Configure multi-stage build for production
- [ ] Configure Caddy for serving static files
- [ ] Set up Cloudflared tunnel for secure external access

## Future Enhancements
- [ ] Timeline view of citations
- [ ] Court hierarchy visualization
- [ ] Citation sentiment analysis
- [ ] User accounts and saved searches
- [ ] Export functionality for citation networks
- [ ] Add syntax highlighting for citations within text
- [ ] Create table of contents based on opinion sections
- [ ] Add "copy citation" feature for academic/legal reference 