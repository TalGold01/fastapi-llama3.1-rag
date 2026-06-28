import requests
import time
import numpy as np

# Configuration
API_URL = "http://localhost:8000/query"
NUM_REQUESTS = 20
PAYLOAD = {"question": "What is the Nighthawk protocol rule for firewalls?"}

def run_benchmark():
    print(f"🚀 Starting RAG Latency Benchmark: {NUM_REQUESTS} requests...")
    print("Measuring Time To First Token (TTFT) - Includes FAISS Retrieval & LLM Eval")
    print("-" * 50)
    
    latencies = []
    
    for i in range(NUM_REQUESTS):
        start_time = time.time()
        
        try:
            # Increased timeout to 60s for local low-spec hardware
            response = requests.post(API_URL, json=PAYLOAD, stream=True, timeout=60)
            
            # Wait for the very first chunk of text to be yielded
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    break
            
            # CRITICAL: Close the streaming connection so we don't starve the local server
            response.close()
                    
            ttft = (time.time() - start_time) * 1000 # Convert to milliseconds
            latencies.append(ttft)
            print(f"Request {i+1:02d}/{NUM_REQUESTS} | TTFT: {ttft:.2f} ms")
            
            # Give the local CPU a 1-second breather between heavy LLM inferences
            time.sleep(1)
            
        except Exception as e:
            print(f"Request {i+1:02d} failed: {e}")

    if not latencies:
        print("All requests failed. Is the API running?")
        return

    # Calculate Enterprise Percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)

    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS (Time To First Token)")
    print("=" * 50)
    print(f"Total Requests: {len(latencies)}")
    print(f"Average (Mean): {avg:.2f} ms")
    print(f"P50 Latency:    {p50:.2f} ms")
    print(f"P95 Latency:    {p95:.2f} ms")
    print(f"P99 Latency:    {p99:.2f} ms")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()