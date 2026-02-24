import logging
import os

class AlertManager:
    def __init__(self, log_file="monitoring/alerts.log"):
        self.log_file = log_file
        if not os.path.exists("monitoring"):
            os.makedirs("monitoring")
            
        # Setup logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AlertManager")

    def send_alert(self, message, level="INFO"):
        """
        Sends an alert through multiple channels.
        Currently: Console and Log File.
        Future: Telegram/Email.
        """
        formatted_msg = f"[ALERT] {message}"
        print(formatted_msg)
        
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)

    def notify_trade(self, signal, price, size, sl, tp):
        msg = f"TRADE EXECUTED: {signal} {size} shares at {price:.2f} | SL: {sl:.2f}, TP: {tp:.2f}"
        self.send_alert(msg, level="INFO")

    def notify_circuit_breaker(self, reason):
        msg = f"CIRCUIT BREAKER TRIGGERED: {reason}"
        self.send_alert(msg, level="CRITICAL")

if __name__ == "__main__":
    alert = AlertManager()
    alert.send_alert("Test alert message")
