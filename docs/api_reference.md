# Theta AI API Reference

This document provides detailed information about the Theta AI API endpoints, enabling programmatic interaction with the model.

## API Overview

The Theta AI API allows developers to integrate Theta's capabilities into other applications through a RESTful interface. The API supports both synchronous and streaming responses, context management, and various configuration options.

## Authentication

API requests require authentication using an API key:

```http
Authorization: Bearer YOUR_API_KEY
```

API keys are managed by system administrators and must be securely stored. Contact your system administrator to obtain an API key with appropriate permissions.

## API Endpoints

### Text Generation

#### POST /api/v1/generate

Generate a text response from Theta AI.

**Request Format:**

```json
{
  "prompt": "What are the best practices for cloud security?",
  "max_tokens": 500,
  "temperature": 0.7,
  "context_id": "optional-conversation-id"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | The input text to generate a response for |
| `max_tokens` | integer | No | Maximum number of tokens to generate (default: 1024) |
| `temperature` | float | No | Controls randomness (0.0-1.0, default: 0.7) |
| `context_id` | string | No | Conversation ID for maintaining context across requests |
| `stream` | boolean | No | If true, response is streamed (default: false) |
| `system_prompt` | string | No | Custom system prompt to use for this generation |

**Response Format:**

```json
{
  "id": "gen_12345",
  "response": "Cloud security best practices include...",
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 156,
    "total_tokens": 163
  },
  "context_id": "conv_67890"
}
```

**Example Request:**

```bash
curl -X POST https://your-theta-server/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "What are the best practices for cloud security?",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

#### POST /api/v1/stream

Stream a response token by token for real-time display.

**Request Format:**
Same as `/api/v1/generate`, but responses are streamed using server-sent events.

**Response Format:**
A series of server-sent events with JSON payloads:

```
event: token
data: {"token": "Cloud", "index": 0}

event: token
data: {"token": " security", "index": 1}

event: done
data: {"id": "gen_12345", "usage": {"prompt_tokens": 7, "completion_tokens": 156, "total_tokens": 163}, "context_id": "conv_67890"}
```

### Context Management

#### GET /api/v1/contexts/{context_id}

Retrieve a conversation context by ID.

**Response Format:**

```json
{
  "context_id": "conv_67890",
  "turns": [
    {
      "role": "user",
      "content": "What are the best practices for cloud security?"
    },
    {
      "role": "assistant",
      "content": "Cloud security best practices include..."
    }
  ],
  "created_at": "2025-09-13T12:45:00Z",
  "updated_at": "2025-09-13T12:46:20Z"
}
```

#### DELETE /api/v1/contexts/{context_id}

Delete a conversation context.

**Response:**
HTTP 204 No Content

### Knowledge Base Search

#### POST /api/v1/search

Search the Theta AI knowledge base for relevant information.

**Request Format:**

```json
{
  "query": "RTX 3060 specifications",
  "top_k": 5
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The search query |
| `top_k` | integer | No | Number of results to return (default: 3) |
| `filters` | object | No | Optional filters to apply to search |

**Response Format:**

```json
{
  "results": [
    {
      "score": 0.92,
      "source": "hardware_knowledge.json",
      "content": "The RTX 3060 has 12GB GDDR6 memory, ray tracing capabilities, and good 1080p/1440p gaming performance."
    },
    // Additional results...
  ],
  "query": "RTX 3060 specifications",
  "total_found": 8,
  "search_time_ms": 53
}
```

## API Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (invalid or missing API key) |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Server error |

## Rate Limiting

API calls are subject to rate limiting:

- 60 requests per minute per API key
- 10,000 tokens per minute per API key

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1631546520
```

## SDK Libraries

Official client libraries are available for:

- Python: `pip install theta-ai-client`
- JavaScript: `npm install theta-ai-js`

### Python Example

```python
from theta_ai_client import ThetaClient

client = ThetaClient(api_key="YOUR_API_KEY")

response = client.generate(
    prompt="Explain the difference between RAM and ROM",
    max_tokens=300
)

print(response.text)
```

### JavaScript Example

```javascript
import { ThetaClient } from 'theta-ai-js';

const client = new ThetaClient({ apiKey: 'YOUR_API_KEY' });

async function generateText() {
  const response = await client.generate({
    prompt: 'Explain the difference between RAM and ROM',
    maxTokens: 300
  });
  
  console.log(response.text);
}

generateText();
```

## API Versioning

The API follows semantic versioning (v1, v2, etc.) in the URL path. Breaking changes will only be introduced in new major versions. The current version is v1.

## Best Practices

1. **Maintain Context**: Use `context_id` for multi-turn conversations to maintain context
2. **Error Handling**: Implement proper error handling for all API calls
3. **Streaming**: Use streaming for real-time interfaces and better user experience
4. **Rate Limiting**: Design your application with rate limits in mind
5. **Retry Logic**: Implement exponential backoff for failed requests

## Support

For API-related issues or questions, please:

- Check the [troubleshooting guide](./troubleshooting.md)
- Open an issue in the GitHub repository
- Contact the support team at [frostlinesolutions.com](mailto:frostlinesolutionsllc@gmail.com)
