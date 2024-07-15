class Alert:
    def __init__(self) -> None:
        self.messages = []

    def add_alert(self, message: str) -> None:
        """
        Adds an alert message to the alert list.

        :param message: The alert message to add.
        """
        self.messages.append(message)

    def get_alerts(self) -> list:
        """
        Returns the list of alert messages.

        :return: List of alert messages.
        """
        return self.messages

    def clear_alerts(self) -> None:
        """
        Clears all alert messages.
        """
        self.messages.clear()
