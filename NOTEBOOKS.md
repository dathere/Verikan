# Accessing Generated Notebooks

The Data Concierge API now saves generated Jupyter notebooks to disk and provides endpoints to access them.

## Notebook Storage

Generated notebooks are saved in the `notebooks/` directory at the root of the project.

## API Endpoints

### 1. Query with Notebook Generation

When making a query, include `"include_notebook": true` (default):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the unemployment rate in Texas?", "include_notebook": true}'
```

The response will include a `notebook_url` field like:
```json
{
  "query_id": "32bd10aa-a3f6-438a-8a34-6a14bb2f52f9",
  "notebook_url": "/api/v1/notebooks/32bd10aa-a3f6-438a-8a34-6a14bb2f52f9",
  ...
}
```

### 2. List All Notebooks

Get a list of all generated notebooks:

```bash
curl http://127.0.0.1:8000/api/v1/notebooks
```

Response:
```json
{
  "count": 1,
  "notebooks": [
    {
      "query_id": "32bd10aa-a3f6-438a-8a34-6a14bb2f52f9",
      "filename": "32bd10aa-a3f6-438a-8a34-6a14bb2f52f9.ipynb",
      "created": 1768294028.2749467,
      "size_bytes": 26811,
      "download_url": "/api/v1/notebooks/32bd10aa-a3f6-438a-8a34-6a14bb2f52f9"
    }
  ]
}
```

### 3. Download a Specific Notebook

Download a notebook by query ID:

```bash
curl -O -J http://127.0.0.1:8000/api/v1/notebooks/32bd10aa-a3f6-438a-8a34-6a14bb2f52f9
```

Or access it in your browser:
```
http://127.0.0.1:8000/api/v1/notebooks/32bd10aa-a3f6-438a-8a34-6a14bb2f52f9
```

## Opening Notebooks

### Option 1: Direct File Access
Open the notebook directly from the file system:
```bash
jupyter lab notebooks/32bd10aa-a3f6-438a-8a34-6a14bb2f52f9.ipynb
```

### Option 2: Download via API
Download using the API endpoint and open in your Jupyter environment:
```bash
curl -o my_notebook.ipynb http://127.0.0.1:8000/api/v1/notebooks/32bd10aa-a3f6-438a-8a34-6a14bb2f52f9
jupyter lab my_notebook.ipynb
```

### Option 3: VS Code
Open the notebook directly in VS Code:
1. Navigate to the `notebooks/` directory in the file explorer
2. Click on the `.ipynb` file to open it in VS Code's notebook editor

## Notebook Contents

Each generated notebook includes:
- Query metadata and documentation
- Standard imports (pandas, numpy, visualization libraries)
- Data retrieval code (reproducible API calls)
- Data processing and analysis
- Visualizations
- Statistical computations
- Citations and references

All code is fully executable and can be modified to extend the analysis.

## Notes

- Notebooks are stored locally in the `notebooks/` directory
- The directory is git-ignored to prevent committing large notebook files
- Each notebook is named with its query ID for easy tracking
- Notebooks persist between API restarts
