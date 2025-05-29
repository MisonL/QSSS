# 量化交易策略系统 (Quantitative Stock Analysis System)

## Project Overview

This project is a quantitative stock analysis system designed for the A-share market. It combines a Python backend for data fetching, financial indicator calculation, and machine learning-based stock selection, with a Next.js web application for user interaction and results display. The system is containerized using Docker for easy deployment and is designed to operate in a pure memory mode, making it suitable for platforms with ephemeral file systems like Hugging Face Spaces and ModelScope.

## Features

*   **Python Backend:**
    *   Fetches A-share stock data (historical prices, daily fundamentals, annual financial reports).
    *   Calculates a variety of technical and fundamental indicators.
    *   Utilizes a LightGBM machine learning model for stock trend prediction.
    *   Implements a strategy to select promising stocks based on a combination of factors.
*   **Key Indicators Used:**
    *   **Technical:** Moving Averages (MA), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Volatility, Momentum, Volume Ratios.
    *   **Fundamental (Daily):** P/E Ratio (PE), P/B Ratio (PB), P/S Ratio (PS), Dividend Yield Ratio, Total Market Value.
    *   **Fundamental (Annual):** Earnings Per Share (EPS), Book Value Per Share (BVPS), Return On Equity (ROE), Debt-to-Asset Ratio, Gross Profit Margin, Net Profit Margin.
*   **Next.js Frontend:**
    *   Provides a web interface to trigger the analysis and display selected stocks and their key metrics.
    *   Communicates with the Python backend via a Next.js API route.
*   **Docker Support:**
    *   Includes a `Dockerfile` for building a containerized version of the application.
    *   Facilitates deployment on platforms like Hugging Face Spaces, ModelScope, or any Docker-compatible environment.
*   **Pure Memory Mode:**
    *   Designed to operate primarily in memory, minimizing reliance on a persistent file system for core operations (e.g., data caching is in-memory).

## Project Structure

```
.
├── app.py                  # Main Python application entry point (strategy execution)
├── Dockerfile              # For building and deploying the application
├── requirements.txt        # Python backend dependencies
├── src/                    # Python backend source code
│   ├── core/               # Core logic: strategy, data fetching, indicators, ML model
│   │   ├── data_fetcher.py
│   │   ├── indicators.py
│   │   ├── model.py
│   │   └── strategy.py
│   └── utils/              # Helper utilities (e.g., retry decorators)
│       └── helpers.py
├── webapp/                 # Next.js frontend application
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/run-strategy/route.ts # API route to trigger Python script
│   │   │   └── page.tsx                # Main page component
│   │   └── ...             # Other Next.js files (components, public, etc.)
│   ├── package.json
│   └── next.config.js
└── tests/                  # Unit and integration tests
    ├── api/
    └── core/
```

## Prerequisites

*   **Python:** 3.10 or higher.
*   **Node.js:** v18 or higher (for `webapp` development and building).
*   **Docker:** Required if building or running the application using Docker.
*   **pip:** For Python package management.
*   **npm:** For Node.js package management.

## Installation & Setup (Local Development)

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Python Backend Setup:**
    *   It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        # Windows:
        # venv\Scripts\activate
        # macOS/Linux:
        source venv/bin/activate
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        *(For users in mainland China, consider using a local pip mirror for faster downloads, e.g., `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`)*

3.  **Next.js Frontend Setup:**
    ```bash
    cd webapp
    npm install
    cd .. 
    ```

## How to Run (Local Development)

The system is designed to be primarily accessed via its web interface.

1.  **Start the Next.js Development Server:**
    This server will handle API requests and forward them to the Python backend.
    ```bash
    cd webapp
    npm run dev
    ```
    The web application will typically be available at `http://localhost:3000`. Open this URL in your browser. The page will automatically fetch and display the strategy results by calling the backend API.

2.  **Alternative: Running Python Script Directly (for CLI output or debugging):**
    You can also run the Python backend script directly from the project root:
    *   For detailed console output (including logs, statistics, and tables):
        ```bash
        python app.py
        ```
    *   For JSON output (this is what the Next.js API route uses):
        ```bash
        python app.py --json-output
        ```

## How to Run (Docker)

1.  **Build the Docker Image:**
    From the project root directory:
    ```bash
    docker build -t quant-analysis-app .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 3000:3000 quant-analysis-app
    ```
    This will start the Next.js application, which serves the frontend and the API route that executes the Python backend.

3.  **Access the Web Interface:**
    Open `http://localhost:3000` in your web browser.

## Deployment

The provided `Dockerfile` is designed for easy deployment on various platforms supporting Docker containers, such as:
*   Hugging Face Spaces
*   ModelScope
*   Other cloud platforms (AWS, GCP, Azure, etc.)

The application runs on port `3000` within the container, which should be mapped to an external port.

## Web Interface

The web interface provides a user-friendly way to:
*   Automatically trigger the execution of the quantitative analysis strategy upon loading.
*   View the selected stocks along with their key performance indicators and financial metrics in a tabular format.
*   See loading and error states during the data fetching and processing.

## Pure Memory Mode

The system is designed to operate primarily in memory. Data caching (e.g., for fetched stock data) is done in-memory within the Python process. This ensures compatibility with platforms that have ephemeral or read-only file systems, common in serverless or containerized deployment environments.

## Author

-   Developer: Mison
-   Contact: 1360962086@qq.com

## License

MIT License
