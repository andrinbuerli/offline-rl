###########################
# HELP
###########################
include scripts/*.mk

###########################
# VARIABLES
###########################
ifeq ($(OS),Windows_NT)
	VENV := .venv/Scripts
else
	VENV := .venv/bin
endif

PARAMS := 

DATA_PATH := ../data

DIR :=

FILE := 


###########################
# Data
###########################

###########################
# Project UTILS
###########################
ls_data: ##@Utils list local data
	ls -l -a $(DATA_PATH)

setup: ##@Utils setup python virtual environment and install requirements with uv
	rm -rf .venv
	uv sync

print_env: ##@Utils print environment variables
	@echo "export VENV=$(VENV)"
	@echo "export DATA_PATH=$(DATA_PATH)"
	@echo "export DIR=$(DIR)"
	@echo "export FILE=$(FILE)"
	@echo "export PARAMS=$(PARAMS)"
