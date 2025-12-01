#!/usr/bin/env python3
"""
Parking System Main Application
"""
import argparse
import sys
import subprocess
import os

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Parking System Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    
    # Seed command
    seed_parser = subparsers.add_parser("seed", help="Seed database with sample slots")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new vehicle")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check parking status")
    status_parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Start parking detection")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test database connection")
    
    # Setup command (init + seed)
    setup_parser = subparsers.add_parser("setup", help="Setup database (init + seed)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "init":
            print("Initializing database...")
            from database.init_db import init_database
            init_database()
            
        elif args.command == "seed":
            print("Seeding database with sample slots...")
            from database.seed_slots import seed_slots
            seed_slots()
            
        elif args.command == "register":
            from assignment.assign_slot import interactive_register
            interactive_register()
            
        elif args.command == "status":
            from utils.status_check import check_parking_status
            check_parking_status(detailed=args.detailed)
            
        elif args.command == "detect":
            print("Starting parking detection...")
            from detection.parking_detect import main as detection_main
            detection_main()
            
        elif args.command == "test":
            from utils.db_connection import test_connection
            if test_connection():
                print("✅ Database connection successful")
            else:
                print("❌ Database connection failed")
                
        elif args.command == "setup":
            print("Setting up database...")
            from database.init_db import init_database
            from database.seed_slots import seed_slots
            init_database()
            seed_slots()
            print("✅ Database setup completed successfully!")
            
        else:
            parser.print_help()
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Django is installed: pip install django mysql-connector-python")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()