# Robotics_MobileChallenge
# Mobile Robot Challenge 2024: Autonomous Navigation and Object Interaction

## Overview
This project presents a viable solution to address two robotics challenges: autonomous navigation and object interaction. The goal is to demonstrate the capability of our robot to autonomously find and interact with predefined objects and return to the starting position. Utilizing an RGB camera and a vector-based methodology, our robot successfully navigates and manipulates objects. However, precision remains a challenging aspect to improve, as new data points introduce cumulative errors. Despite these accuracy issues, the project provides valuable insights into sensor integration and intelligent algorithm development for autonomous robotic systems.

## How to run the tests
Prerequisites:
- RealVNC Viewer

Setup Instructions:
- Connect the computer to the robot using its IP address on RealVNC Viewer, entering the correct credentials

Running the tests:
- Challenge 1: Autonomous Return to Start

1. Position the Robot at the Starting Point

2. Place the robot at a designated starting position in the test area.

***
Run keyboards_control.py in the Raspberry terminal
***

3. Use the arrow keys to drive the robot manually. Press 'q' to stop and initiate the autonomous return.

- Challenge 2: Object Interaction and Return to Start

1. Place the robot at a designated starting position in the test area.
2. Place a 9cm × 9cm × 9cm red cube in the test area at a random location.

***
Run python color_detect.py
***

3. Use the arrow keys to drive the robot manually. Press 'q'. The robot should autonomously detect the object, interact with it, and then return to the starting position.





