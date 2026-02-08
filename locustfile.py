from locust import HttpUser, task, between

class BinItRightUser(HttpUser):
    # Simulates users waiting between 1 and 3 seconds between requests
    wait_time = between(1, 3)

    @task(5)
    def get_forecast(self):
        
        # This targets the actual logic that loads your pkl file
        self.client.get("/admin/forecast")

    @task(1)
    def load_openapi(self):
        """Simulates developers/tools fetching the API schema."""
        self.client.get("/python/openapi.json")
