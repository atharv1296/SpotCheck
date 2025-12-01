#!/usr/bin/env python3
"""
Create media directories for driver applications file uploads
"""

import os
from pathlib import Path

def create_media_directories():
    """Create the necessary media directories for driver applications"""
    
    # Get the base directory (where manage.py is located)
    base_dir = Path(__file__).parent
    
    # Define media directories needed
    media_dirs = [
        'media/driver_applications/photos',
        'media/driver_applications/licenses', 
        'media/driver_applications/puc',
        'media/driver_applications/insurance',
        'media/driver_applications/receipts',
        'media/driver_applications/rc',
        'media/driver_applications/permits',
        'media/driver_applications/customs',
    ]
    
    print("Creating media directories for driver applications...")
    
    for dir_path in media_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
        
        # Create a .gitkeep file to ensure the directory is tracked in git
        gitkeep_file = full_path / '.gitkeep'
        gitkeep_file.touch()
    
    print("\n✅ All media directories created successfully!")
    print("\nNote: These directories will store uploaded documents from driver applications:")
    print("- photos: Driver photos")  
    print("- licenses: Driving license documents")
    print("- puc: Pollution Under Control certificates")
    print("- insurance: Vehicle insurance documents")
    print("- receipts: Material receipts/invoices")
    print("- rc: Vehicle Registration Certificates")
    print("- permits: Goods transport permits (optional)")
    print("- customs: Customs clearance documents (optional)")

if __name__ == "__main__":
    create_media_directories()