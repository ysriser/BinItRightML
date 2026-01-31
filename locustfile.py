from locust import HttpUser, task, between

class BinItRightUser(HttpUser):
    # Users wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    @task
    def load_openapi(self):
        """Simulates users fetching the API schema."""
        self.client.get("/openapi.json")

    @task(3)
    def root_endpoint(self):
        """Simulates users hitting the main entry point."""
        self.client.get("/")
