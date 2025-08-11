"""Load testing for ollama-proxy using Locust."""

import json
import random
import time
from locust import HttpUser, task, between


class OllamaProxyUser(HttpUser):
    """Simulates a user interacting with the Ollama Proxy."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        self.models = []
        self.get_available_models()
    
    def get_available_models(self):
        """Get available models from the API."""
        try:
            response = self.client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                self.models = [model["name"] for model in data.get("models", [])]
            else:
                # Fallback models for testing
                self.models = ["gemini-pro", "gpt-4", "claude-3-sonnet"]
        except Exception:
            # Fallback models for testing
            self.models = ["gemini-pro", "gpt-4", "claude-3-sonnet"]
    
    @task(3)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health")
    
    @task(2)
    def get_version(self):
        """Test version endpoint."""
        self.client.get("/api/version")
    
    @task(2)
    def list_models(self):
        """Test listing models."""
        self.client.get("/api/tags")
    
    @task(1)
    def show_model(self):
        """Test showing model details."""
        if self.models:
            model = random.choice(self.models)
            self.client.post("/api/show", json={"name": model})
    
    @task(5)
    def chat_completion(self):
        """Test chat completion endpoint."""
        if not self.models:
            return
        
        model = random.choice(self.models)
        messages = [
            {"role": "user", "content": self.get_random_prompt()}
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": random.uniform(0.1, 1.0),
                "max_tokens": random.randint(50, 200)
            }
        }
        
        with self.client.post("/api/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400:
                # Model not found or invalid request
                response.failure("Bad request - possibly invalid model")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(3)
    def generate_completion(self):
        """Test generate completion endpoint."""
        if not self.models:
            return
        
        model = random.choice(self.models)
        prompt = self.get_random_prompt()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": random.uniform(0.1, 1.0),
                "max_tokens": random.randint(50, 200)
            }
        }
        
        with self.client.post("/api/generate", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400:
                response.failure("Bad request - possibly invalid model")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def embeddings(self):
        """Test embeddings endpoint."""
        if not self.models:
            return
        
        # Use a model that might support embeddings
        model = random.choice(self.models)
        text = self.get_random_text_for_embedding()
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        with self.client.post("/api/embeddings", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400:
                # Model might not support embeddings
                response.failure("Bad request - model might not support embeddings")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def metrics(self):
        """Test metrics endpoint."""
        self.client.get("/metrics")
    
    def get_random_prompt(self):
        """Get a random prompt for testing."""
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about nature.",
            "How do neural networks work?",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "What is the difference between AI and ML?",
            "Explain the concept of blockchain.",
            "How does the internet work?",
            "What is climate change?",
            "Describe the solar system.",
            "What is the theory of relativity?",
            "How do computers process information?",
            "What is the importance of biodiversity?",
            "Explain the water cycle.",
        ]
        return random.choice(prompts)
    
    def get_random_text_for_embedding(self):
        """Get random text for embedding testing."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Climate change is one of the most pressing issues of our time.",
            "The human brain contains approximately 86 billion neurons.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "The internet has revolutionized how we communicate and access information.",
            "Quantum computers use quantum mechanical phenomena to process information.",
            "Biodiversity is essential for maintaining healthy ecosystems.",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "Artificial intelligence has the potential to transform many industries.",
        ]
        return random.choice(texts)


class HighLoadUser(OllamaProxyUser):
    """High-load user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Much shorter wait times
    
    @task(10)
    def rapid_health_checks(self):
        """Rapid health checks for stress testing."""
        self.client.get("/health")
    
    @task(8)
    def rapid_chat_requests(self):
        """Rapid chat requests for stress testing."""
        if not self.models:
            return
        
        model = random.choice(self.models)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {"max_tokens": 10}  # Short responses for speed
        }
        
        self.client.post("/api/chat", json=payload)


class LightUser(OllamaProxyUser):
    """Light user for baseline testing."""
    
    wait_time = between(5, 10)  # Longer wait times
    
    @task(5)
    def occasional_health_check(self):
        """Occasional health checks."""
        self.client.get("/health")
    
    @task(3)
    def occasional_chat(self):
        """Occasional chat requests."""
        if not self.models:
            return
        
        model = random.choice(self.models)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "How are you?"}],
            "stream": False
        }
        
        self.client.post("/api/chat", json=payload)


class ErrorTestUser(HttpUser):
    """User that tests error conditions."""
    
    wait_time = between(1, 2)
    
    @task(3)
    def test_invalid_model(self):
        """Test with invalid model names."""
        payload = {
            "model": "nonexistent-model-12345",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        with self.client.post("/api/chat", json=payload, catch_response=True) as response:
            if response.status_code == 400:
                response.success()
            else:
                response.failure(f"Expected 400, got {response.status_code}")
    
    @task(2)
    def test_malformed_request(self):
        """Test with malformed requests."""
        payload = {
            "invalid_field": "invalid_value"
        }
        
        with self.client.post("/api/chat", json=payload, catch_response=True) as response:
            if response.status_code in [400, 422]:  # Bad request or validation error
                response.success()
            else:
                response.failure(f"Expected 400/422, got {response.status_code}")
    
    @task(1)
    def test_unsupported_endpoint(self):
        """Test unsupported endpoints."""
        with self.client.post("/api/create", json={}, catch_response=True) as response:
            if response.status_code == 501:  # Not implemented
                response.success()
            else:
                response.failure(f"Expected 501, got {response.status_code}")
    
    @task(1)
    def test_nonexistent_endpoint(self):
        """Test nonexistent endpoints."""
        with self.client.get("/api/nonexistent", catch_response=True) as response:
            if response.status_code == 404:
                response.success()
            else:
                response.failure(f"Expected 404, got {response.status_code}")


# Custom load test scenarios
class BurstTestUser(OllamaProxyUser):
    """User that creates burst traffic patterns."""
    
    wait_time = between(0, 1)
    
    def on_start(self):
        """Initialize burst user."""
        super().on_start()
        self.burst_count = 0
    
    @task
    def burst_requests(self):
        """Create burst patterns."""
        # Create bursts of 5-10 requests
        burst_size = random.randint(5, 10)
        
        for _ in range(burst_size):
            self.client.get("/health")
            time.sleep(0.1)  # Small delay between burst requests
        
        # Longer pause between bursts
        time.sleep(random.uniform(2, 5))


if __name__ == "__main__":
    # This allows running the load test directly
    import subprocess
    import sys
    
    print("Starting load test...")
    print("Make sure the ollama-proxy server is running on localhost:11434")
    
    # Basic load test command
    cmd = [
        "locust",
        "--headless",
        "--users", "10",
        "--spawn-rate", "2",
        "--host", "http://localhost:11434",
        "--run-time", "60s",
        "--html", "load-test-report.html"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Load test completed. Check load-test-report.html for results.")
    except subprocess.CalledProcessError as e:
        print(f"Load test failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Locust not found. Install with: pip install locust")
        sys.exit(1)
