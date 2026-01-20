# WebLDM - Web-based Lifetime Density Analysis

A web application for performing Lifetime Density Analysis on time-resolved spectroscopic data.

Based on [PyLDM](https://github.com/gadorlhiac/PyLDM) by Dorlhiac GF, Fare C, van Thor JJ.

## Features

- **Data Upload**: Upload CSV files with time-resolved spectroscopy data
- **SVD Analysis**: Singular Value Decomposition for data exploration
- **Global Analysis**: Fit lifetimes to weighted left singular vectors
- **LDA (Lifetime Density Analysis)**: 
  - Tikhonov (L2) regularization
  - LASSO (L1) regularization
  - Elastic Net regularization
  - Truncated SVD
- **Interactive Plots**: Explore results with Plotly.js interactive visualizations

## Quick Start with Docker

The easiest way to run WebLDM is with Docker Compose:

```bash
docker compose up -d
```

The application will be available at http://localhost:3000

To stop:
```bash
docker compose down
```

## Manual Installation

### Prerequisites

- Python 3.8+
- Node.js 18+

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running Manually (Development)

### Start the Backend (FastAPI)

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run.py
```

The API will be available at http://localhost:8000
- API documentation: http://localhost:8000/api/docs

### Start the Frontend (React)

```bash
cd frontend
npm run dev
```

The web application will be available at http://localhost:5173

## Data Format

Upload CSV files with the following format:

```
0.0,wavelength1,wavelength2,wavelength3,...
time1,signal11,signal12,signal13,...
time2,signal21,signal22,signal23,...
...
```

- First row: Wavelengths (first cell is ignored)
- First column: Time points
- Remaining cells: Signal intensity matrix

Example datasets are provided in the `example_data/` directory.

## Usage

1. Upload your CSV data file
2. View the raw data contour plot
3. Set IRF parameters (FWHM) if needed
4. Explore SVD decomposition
5. Run Global Analysis to extract lifetimes
6. Run LDA to get the Lifetime Density Map

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload CSV data |
| `/api/data/{session_id}` | GET | Get data info |
| `/api/bounds` | POST | Update data bounds |
| `/api/irf` | POST | Update IRF parameters |
| `/api/chirp/fit` | POST | Fit chirp correction |
| `/api/svd/{session_id}` | GET | Get SVD results |
| `/api/ga` | POST | Run Global Analysis |
| `/api/lda` | POST | Run LDA |
| `/api/tsvd` | POST | Run Truncated SVD |

## Citation

If you use this software, please cite the original PyLDM paper:

> Dorlhiac GF, Fare C, van Thor JJ. (2017) PyLDM - An open source package for lifetime density analysis of time-resolved spectroscopic data. PLoS Comput Biol. 2017 May 22;13(5):e1005528. https://doi.org/10.1371/journal.pcbi.1005528

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

The original PyLDM code is Copyright (C) 2016 Gabriel Dorlhiac, Clyde Fare, licensed under GPLv3.
