import asyncio
import datetime
from bleak import BleakScanner
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.text import Text

# --- CONFIGURATION ---
TARGET_MAC = "50:32:5F:A2:C2:BF"  # <--- PASTE YOUR TARGET MAC HERE
SCAN_DURATION = 30                # Seconds to run

# --- STATE ---
latest_rssi = -100
last_seen = None
pings = 0

def get_signal_bar(rssi):
    """Creates a visual signal bar based on RSSI."""
    # RSSI typically ranges from -100 (unusable) to -30 (touching)
    normalized = max(0, min(100, (rssi + 100) * 1.5)) # Scale roughly 0-100
    
    if rssi >= -50:
        color = "bright_green"
        status = "EXTREMELY CLOSE"
    elif rssi >= -65:
        color = "green"
        status = "NEARBY"
    elif rssi >= -80:
        color = "yellow"
        status = "APPROACHING"
    else:
        color = "red"
        status = "WEAK / FAR"

    bar_length = int(normalized / 2) # Divide by 2 to fit screen width
    bar = "█" * bar_length
    return f"[{color}]{bar}[/{color}]", f"[{color}]{status}[/{color}]"

def callback(device, advertisement_data):
    """Updates state when target is found."""
    global latest_rssi, last_seen, pings
    if device.address == TARGET_MAC:
        # FIXED: Getting RSSI from advertisement_data
        latest_rssi = advertisement_data.rssi
        last_seen = datetime.datetime.now()
        pings += 1

def generate_display(time_remaining):
    """Generates the full screen UI."""
    global latest_rssi, last_seen
    
    # Calculate time since last ping
    if last_seen:
        seconds_ago = (datetime.datetime.now() - last_seen).total_seconds()
        seen_text = f"{seconds_ago:.1f}s ago"
    else:
        seen_text = "Searching..."
        
    # Get Visuals
    bar_visual, status_text = get_signal_bar(latest_rssi)
    
    # Create the Layout
    layout = Layout()
    
    # Main Content
    content = Text()
    content.append(f"\nTargeting: {TARGET_MAC}\n", style="bold white")
    content.append(f"Time Remaining: {int(time_remaining)}s\n\n", style="dim white")
    
    content.append(f"{latest_rssi} dBm\n", style="bold white" if latest_rssi > -90 else "dim white")
    content.append(f"{status_text}\n", style="bold")
    content.append(f"{bar_visual}\n\n")
    content.append(f"Last Seen: {seen_text}\n")
    content.append(f"Total Pings: {pings}", style="dim cyan")

    panel = Panel(
        Align.center(content),
        title="[bold red]SIGNAL STRENGTH METER[/bold red]",
        border_style="red" if latest_rssi < -80 else "green",
        subtitle="Walk slowly to triangulate"
    )
    
    return panel

async def main():
    start_time = datetime.datetime.now()
    
    # Start the Scanner (Active Mode for better tracking)
    scanner = BleakScanner(detection_callback=callback)
    await scanner.start()
    
    try:
        with Live(generate_display(SCAN_DURATION), refresh_per_second=10) as live:
            while True:
                elapsed = (datetime.datetime.now() - start_time).seconds
                remaining = SCAN_DURATION - elapsed
                
                if remaining <= 0:
                    break
                
                live.update(generate_display(remaining))
                await asyncio.sleep(0.1)
                
    finally:
        await scanner.stop()
        print(f"\n[Finished] Session Complete. Target pinged {pings} times.")

if __name__ == "__main__":
    asyncio.run(main())
