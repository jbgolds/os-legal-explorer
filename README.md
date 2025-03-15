# OS Legal Explorer

An open-source tool for exploring legal cases and their citation networks, via LLM analysis of court opinions.

View live site [here](https://law.jbgolds.com/)
## Features

- **Search for Legal Cases**: Find court opinions by case name, citation, or keywords.
- **Read Court Opinions**: Access the full text of court opinions with citation highlighting.
- **Explore Citation Networks**: Visualize how cases cite each other with an interactive network graph.
- **No Accounts or Signups**: Just a simple frontend.


## Technology Stack

- **Frontend**:
  - HTML/CSS/JavaScript
  - Tailwind CSS for styling
  - DaisyUI for UI components
  - HTMX for dynamic content
  - Alpine.js for interactivity
  - D3.js for citation network visualization

- **Backend**:
  - FastAPI
  - Neo4j (Graph database for citation networks)
  - PostgreSQL 


> [!IMPORTANT]  
>  Note: this project assumes that you have the courtlistener database cloned and running locally. See [CourtListener's Bulk Data Page](https://www.courtlistener.com/help/api/bulk-data/) for more information.

## Getting Started

### Prerequisites

- Docker
- Courtlistener database access, E2E script coming soon.


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jacobrosenthal/os-legal-explorer.git
   cd os-legal-explorer
   ```

2. Build and start the containers:
   ```bash
   docker compose up -d
   ```

3. Access the application:
   Open your browser and navigate to `http://localhost:8000`


## Acknowledgments

- [CourtListener](https://www.courtlistener.com/) for such great data and mission. Would be great to integrate this into their [codebase](https://github.com/freelawproject/courtlistener).


## License

COMING SOON, SETTING UP MIT LICENSE
