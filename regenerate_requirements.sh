#!/bin/bash

# Prune the default environment
hatch env prune default

# Run the update to refresh all lockfiles with proper hashes
hatch run update
