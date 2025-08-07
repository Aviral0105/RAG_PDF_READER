# RAG PDF Reader API - Deployment Guide

This guide will help you deploy your RAG PDF Reader API publicly.

## 🚀 Quick Start

### 1. Local Development

```bash
# Navigate to app directory
cd app

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_KEY="your-secret-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# Run the API
python api.py
```

### 2. Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t rag-pdf-api .
docker run -p 8000:8000 -e API_KEY=your-key -e OPENAI_API_KEY=your-openai-key rag-pdf-api
```

## 🌐 Public Deployment Options

### Option 1: Railway (Recommended for quick deployment)

1. **Sign up** at [Railway.app](https://railway.app)
2. **Connect your GitHub repository**
3. **Set environment variables**:
   - `API_KEY`: Your secret API key
   - `OPENAI_API_KEY`: Your OpenAI API key
4. **Deploy** - Railway will automatically detect the Dockerfile and deploy

### Option 2: Render

1. **Sign up** at [Render.com](https://render.com)
2. **Create a new Web Service**
3. **Connect your GitHub repository**
4. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. **Set environment variables**:
   - `API_KEY`: Your secret API key
   - `OPENAI_API_KEY`: Your OpenAI API key

### Option 3: Heroku

1. **Install Heroku CLI**
2. **Create Heroku app**:
   ```bash
   heroku create your-rag-api-name
   ```
3. **Set environment variables**:
   ```bash
   heroku config:set API_KEY=your-secret-api-key
   heroku config:set OPENAI_API_KEY=your-openai-api-key
   ```
4. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 4: DigitalOcean App Platform

1. **Sign up** at [DigitalOcean](https://digitalocean.com)
2. **Create App** from your GitHub repository
3. **Configure**:
   - Source: GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. **Set environment variables** in the dashboard

### Option 5: AWS/GCP/Azure

For cloud providers, you can use:
- **AWS**: Elastic Beanstalk or ECS
- **GCP**: Cloud Run or App Engine
- **Azure**: App Service or Container Instances

## 🔧 Environment Variables

Create a `.env` file or set these in your deployment platform:

```env
# Required
API_KEY=your-secret-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Optional
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

## 📡 API Usage

### Endpoint: `POST /hackrx/run`

**Headers:**
```
Content-Type: application/json
Accept: application/json
Authorization: Bearer your-api-key
```

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key points mentioned?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The main topic is...",
    "The key points are..."
  ]
}
```

## 🔒 Security Considerations

1. **API Key**: Use a strong, unique API key
2. **Rate Limiting**: Consider implementing rate limiting
3. **CORS**: Configure CORS if needed for web applications
4. **HTTPS**: Always use HTTPS in production
5. **Input Validation**: The API validates all inputs

## 📊 Monitoring

### Health Check
```bash
curl https://your-api-domain.com/health
```

### API Documentation
Visit `https://your-api-domain.com/docs` for interactive API documentation.

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Large PDFs may require more memory
3. **Timeout**: Processing large documents may take time
4. **API Key Issues**: Verify your OpenAI API key is valid

### Logs:
Check your deployment platform's logs for detailed error information.

## 📈 Scaling

For high traffic:
1. **Horizontal Scaling**: Deploy multiple instances
2. **Caching**: Consider caching processed PDFs
3. **CDN**: Use CDN for static assets
4. **Database**: Consider using a database for storing processed documents

## 🔄 Updates

To update your API:
1. Push changes to your repository
2. Your deployment platform will automatically redeploy
3. Or manually trigger a redeploy from the platform dashboard
