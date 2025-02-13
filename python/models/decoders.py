class HardwareManager:
    """A class to manage hardware-related functionalities."""

    def __init__(self):
        """Initialize the HardwareManager with default settings."""
        self.hardware_status = "Initialized"
        self.resources_allocated = 0

    def check_status(self):
        """Check the current status of the hardware."""
        return self.hardware_status

    def allocate_resources(self, amount):
        """Allocate resources for hardware operations."""
        self.resources_allocated += amount
        return f"Allocated {amount} resources. Total allocated: {self.resources_allocated}"

    def get_device(self):
        """Get the current device being used for training."""
        return "cpu"

    def deallocate_resources(self, amount):
        """Deallocate resources from hardware operations."""
        if amount <= self.resources_allocated:
            self.resources_allocated -= amount
            return f"Deallocated {amount} resources. Total remaining: {self.resources_allocated}"
        else:
            return "Error: Not enough resources to deallocate."
