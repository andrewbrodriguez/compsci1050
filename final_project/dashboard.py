import asyncio
import datetime
from bleak import BleakScanner
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.layout import Layout

# --- CONFIGURATION ---
# We map the ID numbers to human names
VENDOR_DB = {
    76: "Apple (iPhone/Mac)",
    117: "Samsung",
    6: "Microsoft",
    1447: "Sonos",
    89: "Nordic Semi",
    224: "Google"
}

# --- STATE ---
# This dictionary will hold all our live data
# Format: { "MAC_ADDR": { "rssi": -80, "count": 1, "vendor": "Unknown", "last_seen": timestamp } }
seen_devices = {}

console = Console()

def get_vendor(manuf_data):
    """Extracts the manufacturer ID and returns a human name."""
    if not manuf_data:
        return "-"
    
    # Get the first key (Company ID) from the dictionary
    company_id = list(manuf_data.keys())[0]
    
    # Return the name if known, otherwise the ID number
    return VENDOR_DB.get(company_id, f"ID: {company_id}")

def generate_table():
    """Creates the table for the UI."""
    table = Table(title="BLE Privacy Auditor - Live Feed", style="green")
    
    table.add_column("MAC Address", style="cyan")
    table.add_column("Vendor / Device", style="magenta")
    table.add_column("Signal (RSSI)", justify="right")
    table.add_column("Pings Seen", justify="center")
    table.add_column("Last Seen", justify="right")

    # Sort devices: Recent first, then by signal strength
    sorted_devices = sorted(
        seen_devices.items(), 
        key=lambda item: item[1]['last_seen'], 
        reverse=True
    )

    current_time = datetime.datetime.now()

    # Loop through devices and add rows
    for mac, data in sorted_devices:
        # Calculate how many seconds ago we saw it
        seconds_ago = (current_time - data['last_seen']).total_seconds()
        
        # Color code the signal strength
        rssi_color = "green" if data['rssi'] > -60 else "yellow" if data['rssi'] > -80 else "red"
        
        # Only show devices seen in the last 60 seconds (keeps the list clean)
        if seconds_ago < 60:
            table.add_row(
                mac,
                str(data['vendor']),
                f"[{rssi_color}]{data['rssi']} dBm[/{rssi_color}]",
                str(data['count']),
                f"{seconds_ago:.1f}s ago"
            )
            
    return table

def callback(device, advertisement_data):
    """Runs every time a packet is detected."""
    vendor_name = get_vendor(advertisement_data.manufacturer_data)
    
    # Determine the display name (Use the device name if available, otherwise Vendor)
    display_name = device.name if device.name else vendor_name

    # Update our database
    seen_devices[device.address] = {
        "rssi": advertisement_data.rssi,
        "count": seen_devices.get(device.address, {}).get("count", 0) + 1,
        "vendor": display_name,
        "last_seen": datetime.datetime.now()
    }

async def main():
    print("Starting Scanner...")
    
    # Create the Scanner
    scanner = BleakScanner(detection_callback=callback)
    
    # Start Scanning
    await scanner.start()

    # Start the UI Loop
    try:
        with Live(generate_table(), refresh_per_second=4) as live:
            while True:
                live.update(generate_table())
                await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        await scanner.stop()
        print("\nStopped.")

if __name__ == "__main__":
    asyncio.run(main())
