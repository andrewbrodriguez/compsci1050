import asyncio
import datetime
import csv
import os
from bleak import BleakScanner
from rich.live import Live
from rich.table import Table
from rich.console import Console

# --- CONFIGURATION ---
SCAN_DURATION = 30  # Seconds to run
VENDOR_DB = {
    76: "Apple",
    117: "Samsung",
    6: "Microsoft",
    1447: "Sonos",
    89: "Nordic Semi",
    224: "Google"
}

# --- STATE ---
# Structure: { "MAC": { "vendor": str, "rssi": int, "count": int, "first_seen": time, "last_seen": time } }
seen_devices = {}
start_time = datetime.datetime.now()
console = Console()

def get_vendor(manuf_data):
    """Identifies the vendor from the manufacturer ID."""
    if not manuf_data:
        return "Unknown"
    company_id = list(manuf_data.keys())[0]
    return VENDOR_DB.get(company_id, f"ID: {company_id}")

def callback(device, advertisement_data):
    """Updates the internal database when a packet is found."""
    vendor = get_vendor(advertisement_data.manufacturer_data)
    name = device.name if device.name else vendor
    now = datetime.datetime.now()

    if device.address not in seen_devices:
        seen_devices[device.address] = {
            "vendor": name,
            "rssi": advertisement_data.rssi,
            "count": 1,
            "first_seen": now,
            "last_seen": now
        }
    else:
        d = seen_devices[device.address]
        d["rssi"] = advertisement_data.rssi # Update to latest signal strength
        d["count"] += 1
        d["last_seen"] = now

def generate_table():
    """Generates the live UI table."""
    table = Table(title=f"BLE Scan in Progress (Time Remaining: {max(0, SCAN_DURATION - (datetime.datetime.now() - start_time).seconds)}s)", style="green")
    
    table.add_column("MAC Address", style="cyan")
    table.add_column("Vendor", style="magenta")
    table.add_column("RSSI", justify="right")
    table.add_column("Pings", justify="center")

    # Sort by most recently seen
    sorted_items = sorted(seen_devices.items(), key=lambda x: x[1]['last_seen'], reverse=True)
    
    # Show top 15 devices to keep screen clean
    for mac, data in sorted_items[:15]:
        table.add_row(
            mac,
            str(data["vendor"]),
            str(data["rssi"]),
            str(data["count"])
        )
    return table

def save_to_csv():
    """Writes the results to a CSV file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_results_{timestamp}.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["MAC Address", "Vendor/Name", "Ping Count", "Last RSSI", "First Seen", "Last Seen", "Duration Visible (s)"])
        
        for mac, data in seen_devices.items():
            duration = (data["last_seen"] - data["first_seen"]).total_seconds()
            writer.writerow([
                mac,
                data["vendor"],
                data["count"],
                data["rssi"],
                data["first_seen"].strftime("%H:%M:%S"),
                data["last_seen"].strftime("%H:%M:%S"),
                f"{duration:.2f}"
            ])
            
    return filename

async def main():
    scanner = BleakScanner(detection_callback=callback)
    await scanner.start()
    
    # Run the Live Dashboard
    with Live(generate_table(), refresh_per_second=4) as live:
        while (datetime.datetime.now() - start_time).seconds < SCAN_DURATION:
            live.update(generate_table())
            await asyncio.sleep(0.5)
            
    await scanner.stop()
    
    # Save and Exit
    filename = save_to_csv()
    console.print(f"\n[bold green]Scan Complete![/bold green]")
    console.print(f"Captured [bold cyan]{len(seen_devices)}[/bold cyan] unique devices.")
    console.print(f"Data saved to: [bold white]{os.getcwd()}/{filename}[/bold white]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Aborted.")
