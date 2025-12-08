import asyncio
from bleak import BleakScanner

def callback(device, advertisement_data):
    # FIXED: We now access RSSI from 'advertisement_data', not 'device'
    print(f"[{device.address}] RSSI: {advertisement_data.rssi} dBm")
    print(f"   Name: {device.name}")
    print(f"   Data: {advertisement_data}")
    print("-" * 40)

async def main():
    print("Scanning... (Press Stop or Ctrl+C to quit)")
    
    # We use active scanning (default) to ensure compatibility with BlueZ
    async with BleakScanner(detection_callback=callback):
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
