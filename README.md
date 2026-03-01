# BWC-Vision-Testing-System

## Project Overview
This system is an automated testing platform for Bradford White Corporation (BWC). It utilizes a Raspberry Pi 5 to perform Optical Character Recognition (OCR) on water heater displays and physically actuates user interface buttons via a 7-servo array.

## Team 14 Members
* [Your Name]
* [Partner Name]
* [Partner Name]

## Hardware Stack
* **Controller:** Raspberry Pi 5 (8GB)
* **Vision:** Raspberry Pi Camera Module 3 (Autofocus)
* **Actuation:** 7x MG90S Metal Gear Servos
* **Driver:** PCA9685 16-Channel PWM Driver
* **Power:** Mean Well 5V 12A DC Power Supply (120V AC Input)

## Repository Structure
* `/src`: Production-ready code (Vision and Motion modules).
* `/playground`: Experimental scripts and sandbox testing ("Room for Chaos").
* `/docs`: Technical specifications and wiring diagrams.
* `/hardware`: CAD files and 3D printing STL files for the mounting bracket.

## Setup Instructions
1. Clone the repository: `git clone [URL]`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Install Tesseract Engine: `sudo apt install tesseract-ocr`
