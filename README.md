# Compatibility-Model

This project implements a compatibility model using a Siamese neural network architecture, designed to evaluate the compatibility between items (such as images) for use in recommendation systems or similarity search. The application exposes a REST API using Flask and stores/retrieves embeddings using Milvus, a vector database.

## Project Structure

- `main.py`: Flask API server, handles requests for adding, recommending, and deleting items.
- `model.py`: Siamese neural network model definition using EfficientNet.
- `milvus_helper.py`: Helper class for interacting with Milvus.
- `best_siamese_model.pth`: Pre-trained model weights.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Containerizes the Flask app.
- `docker-compose.yaml`: Orchestrates the Flask app and Milvus database.

## Requirements

- Python 3.11+
- Docker & Docker Compose (recommended for easy setup)

## Setup & Usage

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Compatibility-Model
```

### 2. (Recommended) Run with Docker Compose

Build and start both the Flask app and Milvus database:

```bash
docker-compose up --build
```

- The Flask API will be available at `http://localhost:5000`
- Milvus will be available at `localhost:19530`

### 3. (Alternative) Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask server:

```bash
python main.py
```

## API Endpoints

- `GET /api/v1/items/recommendations/<item_id>?top_k=5`  
  Get top-k recommended item IDs for a given item.
- `POST /api/v1/items`  
  Add a new item (expects form data: `id`, `category`, and an image file).
- `DELETE /api/v1/items/<item_id>`  
  Delete an item by ID.
- `GET /health`  
  Health check endpoint.

## Environment Variables

- `MILVUS_URI`
- `MILVUS_TOKEN`
