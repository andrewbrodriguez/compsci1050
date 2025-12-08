import asyncio
import datetime
from bleak import BleakScanner
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# --- TARGET CONFIGURATION ---
TARGET_MAC = "50:32:5F:A2:C2:BF"  # The specific ResMed device from your scan
TARGET_NAME = "ResMed 068051"

console = Console()

def get_distance_visual(rssi):
    """Returns a visual bar based on signal strength."""
    # RSSI ranges usually from -100 (far) to -40 (close)
    if rssi >= -50: return "[red]██████████!! (VERY CLOSE)[/red]"
    if rssi >= -60: return "[yellow]████████ (Close)[/yellow]"
    if rssi >= -70: return "[green]██████ (Nearby)[/green]"
    if rssi >= -80: return "[blue]████ (In range)[/blue]"
    return "[dim]██ (Weak signal)[/dim]"

def callback(device, advertisement_data):
    """Only triggers if the MAC matches our target."""
    if device.address == TARGET_MAC:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        
        # FIXED: Access rssi from advertisement_data, not device
        current_rssi = advertisement_data.rssi
        rssi_visual = get_distance_visual(current_rssi)
        
        # Build the alert message
        alert = Text()
        alert.append(f"\n[{now}] TARGET DETECTED\n", style="bold red")
        alert.append(f"MAC: {device.address}\n", style="bold white")
        
        # FIXED: Use current_rssi variable here too
        alert.append(f"Signal: {current_rssi} dBm  ", style="white")
        alert.append(f"{rssi_visual}\n")
        
        # Check for manufacturer specific data
        if advertisement_data.manufacturer_data:
            alert.append(f"Manuf Data: {advertisement_data.manufacturer_data}\n", style="dim cyan")
            
        console.print(alert)

async def main():
    console.print(Panel(f"Targeting Mode Engaged\nSearching for: {TARGET_NAME}\nMAC: {TARGET_MAC}", style="bold red"))
    
    # Active scanning helps find specific devices faster
    async with BleakScanner(detection_callback=callback):
        while True:
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("[bold red]Tracking Terminated.[/bold red]")
