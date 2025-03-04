# OS Legal Explorer - Frontend Implementation TODO

## Setup & Infrastructure
- [x] Create frontend directory structure
  - [x] Create `src/frontend/templates` directory
  - [x] Create `src/frontend/static/{css,js,images}` directories
- [x] Set up build system for Tailwind CSS
  - [x] Follow https://daisyui.com/docs/install/cli/
- [x] Configure Tailwind CSS in `tailwind.config.js`
- [x] Create base CSS file at `src/frontend/static/css/main.css`
- [x] Configure FastAPI for frontend
  - [x] Uncomment and update template and static file configuration in `src/api/main.py`
  - [x] Set up Jinja2 templates with proper directory paths

## Home Page Implementation
- [x] Create base layout template
  - [x] Create `src/frontend/templates/base.html` template with common elements
    - [x] Add header with navigation
    - [x] Add footer with project information
    - [x] Include meta tags and basic SEO elements
    - [x] Set up responsive layout with Tailwind CSS
    - [x] Include script and style imports
  - [x] Implement responsive design using Tailwind CSS and DaisyUI
- [x] Design the homepage template
  - [x] Create `src/frontend/templates/home.html` extending the base template
  - [x] Design interface with prominent search box
  - [x] Add sections for recent cases
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

## Court Case Detail Page
- [x] Create detail page template
  - [x] Develop `src/frontend/templates/case_detail.html` with sections for:
    - [x] Case metadata (court, date, judges, etc.)
    - [x] Full opinion text
    - [x] Citation visualization
    - [x] Related cases
  - [x] Implement responsive layout
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
    - [ ] Add filtering options by court, date, etc.
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
- [x] Implement citation visualization in `src/frontend/static/js/citation_map.js`
  - [x] Create d3.js force-directed graph
  - [x] Handle data loading and transformation
  - [x] Implement interactive features

## Deployment Preparation
- [x] Update Dockerfile and docker-compose.yml
  - [x] Add Node.js/Tailwind build steps
  - [ ] Configure multi-stage build for production
- [ ] Configure Caddy for serving static files
- [ ] Set up Cloudflared tunnel for secure external access

## Next Steps (Immediate Focus)
1. ~~Create backend API endpoints for search and recent cases~~
   - ~~Implement CourtListener API proxy for search~~
   - ~~Create endpoint for fetching recent cases from Neo4j~~
2. ~~Implement the case detail page~~
   - ~~Create template with responsive layout~~
   - ~~Implement case text display with formatting~~
3. ~~Develop the citation mapping visualization with d3.js~~
   - ~~Create backend endpoint for citation network data~~
   - ~~Implement force-directed graph with interactive features~~
4. ~~Add user feedback mechanisms~~
   - ~~Design modal dialogs for different feedback types~~
   - ~~Create backend endpoints for storing feedback~~
5. ~~Update Dockerfile and docker-compose.yml for deployment~~
   - ~~Add Node.js/Tailwind build steps~~
   - Configure multi-stage build for production
6. Enhance the citation mapping visualization
   - Add filtering options by court, date, etc.
   - Implement more interactive features
7. Integrate with Neo4j database for citation data
   - Update endpoints to use Neo4j instead of placeholder data
   - Implement more sophisticated citation network queries

## Future Enhancements
- [ ] Timeline view of citations
- [ ] Court hierarchy visualization
- [ ] Citation sentiment analysis
- [ ] User accounts and saved searches
- [ ] Export functionality for citation networks 