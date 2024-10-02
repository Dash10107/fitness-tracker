# Fitness Tracker

## Overview

The Fitness Tracker is a Python application that leverages frequency data from a fitness watch to identify and count repetitions of various exercises. Currently, the tracker supports the following exercises:

- Squat
- Bench Press
- Overhead Press
- Deadlift
- Row

By analyzing motion data, the application determines the type of exercise being performed and accurately counts the number of repetitions in each set.

## Features

- **Exercise Detection**: Identifies five major exercises using frequency data.
- **Repetition Counting**: Counts the number of repetitions for each detected exercise.
- **User-Friendly Interface**: Easy to use and integrate with fitness watches.

## Getting Started

### Prerequisites

- Python 3.x
- Conda Environment
### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fitness-tracker.git
   cd fitness-tracker
   conda env create -f environment.yml