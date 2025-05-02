#!/usr/bin/env python3
"""
Main entry point for TB Chest X-Ray Classification System.
This file launches the GUI application.
"""

import os
import sys
import tkinter as tk
from src.ui.app import TBXrayApp

def main():
    """Main function to initialize and run the application."""
    # Create main window
    root = tk.Tk()
    
    # Set app title and icon
    root.title("TB Chest X-ray Classification System")
    
    # Create application instance
    app = TBXrayApp(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    # Ensure proper directory structure
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run the application
    main()